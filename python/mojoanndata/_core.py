"""A focused, eager AnnData implementation for in-memory annotated slicing."""

from __future__ import annotations

import copy as _copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ._lib import addr, lib


def _frame(value: Any, n: int, axis: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame(index=pd.Index([str(i) for i in range(n)]))
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if len(frame) != n:
        raise ValueError(f"{axis} must have as many rows as X has {axis}")
    return frame


def _matrix(value: Any, name: str = "X"):
    if sparse.issparse(value):
        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        return value.copy()
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array.copy()


def _positions(index: Any, names: pd.Index, size: int) -> np.ndarray:
    """Resolve NumPy-style integer/mask selectors plus AnnData string names."""
    if isinstance(index, slice):
        return np.ascontiguousarray(np.arange(size, dtype=np.int64)[index])
    if isinstance(index, (str, bytes)):
        found = names.get_indexer([index])
        if found[0] < 0:
            raise KeyError(index)
        return np.ascontiguousarray(found.astype(np.int64))
    raw = np.asarray(index if not isinstance(index, pd.Series) else index.to_numpy())
    if raw.ndim == 0:
        value = raw.item()
        if isinstance(value, (str, bytes)):
            return _positions(value, names, size)
        raw = np.asarray([value])
    if raw.dtype.kind == "b":
        if raw.ndim != 1 or raw.size != size:
            raise IndexError(f"Boolean index has shape {raw.shape}; expected ({size},)")
        return np.ascontiguousarray(np.flatnonzero(raw).astype(np.int64))
    if raw.dtype.kind in "OUS":
        found = names.get_indexer(raw.astype(object))
        if np.any(found < 0):
            missing = raw[np.flatnonzero(found < 0)[0]]
            raise KeyError(missing)
        return np.ascontiguousarray(found.astype(np.int64))
    if raw.ndim != 1:
        raise IndexError("index must be one-dimensional")
    if raw.dtype.kind == "u":
        # Converting a uint64 greater than Int64.max would wrap it negative.
        if np.any(raw >= size):
            raise IndexError("index out of bounds")
    elif raw.dtype.kind == "f":
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise IndexError("indices must be integers")
    elif raw.dtype.kind != "i":
        raise IndexError("indices must be integers, names, or boolean masks")
    result = np.asarray(raw, dtype=np.int64)
    result = np.where(result < 0, result + size, result)
    if np.any((result < 0) | (result >= size)):
        raise IndexError("index out of bounds")
    return np.ascontiguousarray(result)


def _take_dense(x: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(x)
    if source.dtype != np.float64 or not source.flags.c_contiguous:
        return source[np.ix_(rows, cols)].copy()
    target = np.empty((rows.size, cols.size), dtype=np.float64)
    if target.size:
        lib().mad_dense_take_f64(addr(source), addr(rows), addr(cols), addr(target),
                                  source.shape[1], rows.size, cols.size)
    return target


def _take_csr(x: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray):
    if x.dtype != np.float64 or np.unique(cols).size != cols.size:
        return x[rows, :][:, cols].copy()
    source = x.copy()
    source.sum_duplicates()
    source.sort_indices()
    col_map = np.full(source.shape[1], -1, dtype=np.int64)
    col_map[cols] = np.arange(cols.size, dtype=np.int64)
    indptr = np.ascontiguousarray(source.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(source.indices, dtype=np.int64)
    # Rows may be selected repeatedly, so source.nnz is not a safe capacity.
    # This is an upper bound; column filtering can only reduce it.
    capacity = sum(int(indptr[row + 1] - indptr[row]) for row in rows)
    out_indptr = np.empty(rows.size + 1, dtype=np.int64)
    out_indices = np.empty(max(capacity, 1), dtype=np.int64)
    out_data = np.empty(max(capacity, 1), dtype=np.float64)
    if rows.size and cols.size and source.nnz:
        written = lib().mad_csr_take_f64(addr(indptr), addr(indices),
                                         addr(source.data), addr(rows), addr(col_map),
                                         addr(out_indptr), addr(out_indices), addr(out_data),
                                         rows.size)
    else:
        out_indptr.fill(0)
        written = 0
    if written < 0 or written > capacity:
        raise RuntimeError("native CSR slicer returned an invalid output length")
    result = sparse.csr_matrix((out_data[:written], out_indices[:written], out_indptr),
                               shape=(rows.size, cols.size), copy=False)
    result.sort_indices()
    return result


def _take_matrix(x: Any, rows: np.ndarray, cols: np.ndarray):
    if sparse.isspmatrix_csr(x):
        return _take_csr(x, rows, cols)
    if sparse.issparse(x):
        return x[rows, :][:, cols].copy()
    return _take_dense(np.asarray(x), rows, cols)


class AnnData:
    """In-memory annotated matrix with upstream-compatible covered slicing APIs.

    This deliberately implements eager copies, not backed files or lazy views.
    """

    def __init__(self, X=None, obs=None, var=None, uns=None, obsm=None, varm=None,
                 layers=None, raw=None, dtype=None, shape=None, filename=None,
                 filemode=None, asview=False, obsp=None, varp=None, oidx=None, vidx=None):
        if asview or oidx is not None or vidx is not None:
            raise NotImplementedError("views are not implemented; slicing returns an eager AnnData")
        if filename is not None or filemode is not None:
            raise NotImplementedError("backed .h5ad files are outside the in-memory slicing subset")
        if raw is not None:
            raise NotImplementedError("raw is outside the covered annotated-slicing subset")
        if X is None:
            if shape is None:
                raise ValueError("X or shape is required")
            X = np.zeros(shape, dtype=dtype or np.float64)
        self.X = _matrix(X)
        if dtype is not None:
            self.X = self.X.astype(dtype)
        self.obs = _frame(obs, self.n_obs, "observations")
        self.var = _frame(var, self.n_vars, "variables")
        self.uns = _copy.deepcopy(dict(uns or {}))
        self.layers = {key: _checked_layer(value, self.shape, key) for key, value in (layers or {}).items()}
        # anndata exposes X through the None layer key as well as through .X.
        self.layers.setdefault(None, self.X)
        self.obsm = {key: _checked_axis(value, self.n_obs, "obsm", key) for key, value in (obsm or {}).items()}
        self.varm = {key: _checked_axis(value, self.n_vars, "varm", key) for key, value in (varm or {}).items()}
        self.obsp = {key: _checked_pair(value, self.n_obs, "obsp", key) for key, value in (obsp or {}).items()}
        self.varp = {key: _checked_pair(value, self.n_vars, "varp", key) for key, value in (varp or {}).items()}

    @property
    def shape(self):
        return self.X.shape

    @property
    def n_obs(self):
        return self.X.shape[0]

    @property
    def n_vars(self):
        return self.X.shape[1]

    @property
    def obs_names(self):
        return self.obs.index

    @obs_names.setter
    def obs_names(self, value):
        index = pd.Index(value)
        if len(index) != self.n_obs:
            raise ValueError("obs_names length must equal n_obs")
        self.obs.index = index

    @property
    def var_names(self):
        return self.var.index

    @var_names.setter
    def var_names(self, value):
        index = pd.Index(value)
        if len(index) != self.n_vars:
            raise ValueError("var_names length must equal n_vars")
        self.var.index = index

    @property
    def is_view(self):
        return False

    def __repr__(self):
        return f"AnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}"

    def __getitem__(self, index):
        obs_index, var_index = index if isinstance(index, tuple) else (index, slice(None))
        rows = _positions(obs_index, self.obs_names, self.n_obs)
        cols = _positions(var_index, self.var_names, self.n_vars)
        result = AnnData(_take_matrix(self.X, rows, cols), obs=self.obs.iloc[rows].copy(),
                         var=self.var.iloc[cols].copy(), uns=_copy.deepcopy(self.uns),
                         layers={k: _take_matrix(v, rows, cols) for k, v in self.layers.items()
                                 if k is not None},
                         obsm={k: _take_axis(v, rows) for k, v in self.obsm.items()},
                         varm={k: _take_axis(v, cols) for k, v in self.varm.items()},
                         obsp={k: _take_matrix(v, rows, rows) for k, v in self.obsp.items()},
                         varp={k: _take_matrix(v, cols, cols) for k, v in self.varp.items()})
        return result

    def copy(self, filename=None):
        if filename is not None:
            raise NotImplementedError("backed copies are not implemented")
        return self[:, :]

    def to_memory(self, copy=False):
        return self.copy() if copy else self

    def to_df(self, layer=None):
        matrix = self.layers[layer] if layer is not None else self.X
        values = matrix.toarray() if sparse.issparse(matrix) else matrix
        return pd.DataFrame(values, index=self.obs_names, columns=self.var_names)

    def obs_vector(self, k, *, layer=None):
        matrix = self.layers[layer] if layer is not None else self.X
        if k in self.obs.columns:
            return self.obs[k].to_numpy()
        pos = _positions(k, self.var_names, self.n_vars)[0]
        return matrix[:, pos].toarray().ravel() if sparse.issparse(matrix) else matrix[:, pos].copy()

    def var_vector(self, k, *, layer=None):
        matrix = self.layers[layer] if layer is not None else self.X
        if k in self.var.columns:
            return self.var[k].to_numpy()
        pos = _positions(k, self.obs_names, self.n_obs)[0]
        return matrix[pos, :].toarray().ravel() if sparse.issparse(matrix) else matrix[pos, :].copy()

    def obs_keys(self):
        return list(self.obs.columns)

    def var_keys(self):
        return list(self.var.columns)

    def obsm_keys(self):
        return list(self.obsm)

    def varm_keys(self):
        return list(self.varm)

    def layers_keys(self):
        return list(self.layers)


def _checked_layer(value, shape, key):
    matrix = _matrix(value, f"layers[{key!r}]")
    if matrix.shape != shape:
        raise ValueError(f"layers[{key!r}] has shape {matrix.shape}, expected {shape}")
    return matrix


def _checked_axis(value, n, collection, key):
    if isinstance(value, pd.DataFrame):
        result = value.copy()
    elif sparse.issparse(value):
        result = value.copy()
    else:
        result = np.asarray(value).copy()
    if result.shape[0] != n:
        raise ValueError(f"{collection}[{key!r}] first dimension must be {n}")
    return result


def _checked_pair(value, n, collection, key):
    matrix = _matrix(value, f"{collection}[{key!r}]")
    if matrix.shape != (n, n):
        raise ValueError(f"{collection}[{key!r}] must have shape ({n}, {n})")
    return matrix


def _take_axis(value, positions):
    if isinstance(value, pd.DataFrame):
        return value.iloc[positions].copy()
    return value[positions].copy()
