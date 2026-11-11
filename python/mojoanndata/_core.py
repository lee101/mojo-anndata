"""A focused, eager AnnData implementation for in-memory annotated slicing."""

from __future__ import annotations

import copy as _copy
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ._lib import addr, lib

DENSE_PARALLEL_ELEMENTS = 2_000_000
DENSE_PARALLEL_TASKS = 8


def _frame(value: Any, n: int, axis: str) -> pd.DataFrame:
    # Matches anndata.aligned_df._gen_dataframe: an index key names the row index.
    index_keys = {"obs_names", "row_names"} if axis == "observations" else {"var_names", "col_names"}
    if value is None:
        frame = pd.DataFrame(index=pd.Index([str(i) for i in range(n)]))
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
        if frame.index.dtype.kind != "O" and not isinstance(frame.index.dtype, pd.StringDtype):
            # Matches anndata.aligned_df._gen_dataframe: a non-string index is
            # converted to str (upstream warns ImplicitModificationWarning).
            frame.index = pd.Index([str(i) for i in frame.index])
    elif isinstance(value, np.ndarray) and value.ndim < 2:
        raise ValueError(f"{axis} must be two-dimensional")
    elif isinstance(value, Mapping) and index_keys & set(value):
        index_key = next(iter(index_keys & set(value)))
        frame = pd.DataFrame({k: v for k, v in value.items() if k != index_key},
                             index=pd.Index(value[index_key]))
    else:
        frame = pd.DataFrame(value)
    if len(frame) != n:
        raise ValueError(f"{axis} must have as many rows as X has {axis}")
    if frame.columns.dtype.kind in "iu":
        # Matches anndata.AnnData._set_dim_df: `if not len(value.columns):
        # value.columns = value.columns.astype(str)`; columns become a str index,
        # so `k in adata.var` never matches integer keys.
        frame.columns = pd.Index(list(frame.columns), dtype=str, name=frame.columns.name)
    return frame


def _matrix(value: Any, name: str = "X"):
    if sparse.issparse(value):
        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        return value.copy()
    array = np.asarray(value)
    # anndata `_init_as_actual` takes `X.shape[:2]`, so X may carry trailing dims.
    if array.ndim < 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array.copy()


def _unpack_index(index):
    # Matches anndata.index.unpack_index: one optional Ellipsis, at most two axes.
    if not isinstance(index, tuple):
        return slice(None) if index is Ellipsis else index, slice(None)
    n_ellipsis = sum(item is Ellipsis for item in index)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if len(index) > 2:
        if not n_ellipsis:
            raise IndexError("Received a length 3 index without an ellipsis")
        return tuple(item for item in index if item is not Ellipsis)
    if len(index) == 2:
        return tuple(slice(None) if item is Ellipsis else item for item in index)
    if len(index) == 1:
        item = index[0]
        return (slice(None) if item is Ellipsis else item), slice(None)
    raise IndexError("invalid number of indices")


def _positions(index: Any, names: pd.Index, size: int) -> np.ndarray:
    """Resolve NumPy-style integer/mask selectors plus AnnData string names."""
    if isinstance(index, slice):
        if isinstance(index.start, str) or isinstance(index.stop, str):
            # Matches anndata.index._from_slice: label slices are inclusive.
            start, stop = index.start, index.stop
            if isinstance(start, str):
                start = names.get_loc(start)
            if isinstance(stop, str):
                stop = names.get_loc(stop) + 1
            index = slice(start, stop, index.step)
        return np.ascontiguousarray(np.arange(size, dtype=np.int64)[index])
    if isinstance(index, (str, bytes)):
        found = names.get_indexer([index])
        if found[0] < 0:
            raise KeyError(index)
        return np.ascontiguousarray(found.astype(np.int64))
    if isinstance(index, pd.api.extensions.ExtensionArray):
        raw = np.asarray(index)
    else:
        raw = np.asarray(index.to_numpy() if isinstance(index, pd.Series) else index)
    if raw.ndim == 0:
        value = raw.item()
        if isinstance(value, (str, bytes)):
            return _positions(value, names, size)
        raw = np.asarray([value])
    # Matches anndata.index._from_array: the (n, 1) / (1, n) column form is raveled.
    if raw.shape == (size, 1) or raw.shape == (1, size):
        raw = raw.ravel()
    if raw.dtype.kind == "b":
        if raw.ndim != 1 or raw.size != size:
            raise IndexError(f"Boolean index has shape {raw.shape}; expected ({size},)")
        return np.ascontiguousarray(np.flatnonzero(raw).astype(np.int64))
    if raw.dtype.kind in "OUS":
        if raw.ndim != 1:
            raise IndexError("index must be one-dimensional")
        found = names.get_indexer(raw)
        if np.any(found < 0):
            missing = raw[np.flatnonzero(found < 0)[0]]
            raise KeyError(missing)
        return np.ascontiguousarray(found.astype(np.int64))
    if raw.dtype.kind == "f":
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise IndexError("indices must be integers")
    elif raw.dtype.kind not in "iu":
        raise IndexError("indices must be integers, names, or boolean masks")
    if raw.ndim != 1:
        raise IndexError("index must be one-dimensional")
    result = np.array(raw, dtype=np.int64)
    if raw.dtype.kind == "u":
        # Matches anndata.index._from_array: unsigned arrays are viewed as signed
        # before NumPy bounds-checks them, so values below 2**63 resolve normally.
        result = result.view(np.int64)
    result = np.where(result < 0, result + size, result)
    if np.any((result < 0) | (result >= size)):
        raise IndexError("index out of bounds")
    return np.ascontiguousarray(result)


def _is_arange(cols: np.ndarray, size: int) -> bool:
    """True when `cols` is exactly `[0, 1, ..., size-1]`, i.e. the slice keeps
    every column in order and the result rows are contiguous copies."""
    if cols.size != size or cols.dtype.kind not in "iu":
        return False
    return bool(np.array_equal(cols, np.arange(size, dtype=cols.dtype)))


def _take_dense(x: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(x)
    if source.dtype != np.float64 or not source.flags.c_contiguous:
        return source[np.ix_(rows, cols)].copy()
    target = np.empty((rows.size, cols.size), dtype=np.float64)
    if target.size:
        if cols.size == source.shape[1] and _is_arange(cols, cols.size):
            # No column selection at all: the kernel copies whole rows with a
            # 2*W-wide contiguous loop.  This route is restricted to a full column
            # set because on this pinned toolchain the copy kernel miscounts rows
            # when the last vector block of a row would be partial (a non-full
            # column count with ncols % 8 != 0), so narrowing slices stay on the
            # element-wise gather below.  See README.
            kernel = lib().mad_dense_take_f64_copy
        elif (target.size >= DENSE_PARALLEL_ELEMENTS
              and rows.size >= DENSE_PARALLEL_TASKS):
            kernel = lib().mad_dense_take_f64_parallel
        else:
            kernel = lib().mad_dense_take_f64
        kernel(addr(source), addr(rows), addr(cols), addr(target), source.shape[1],
               rows.size, cols.size)
    return target


def _take_csr(x: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray):
    if x.dtype != np.float64 or np.unique(cols).size != cols.size:
        return x[rows, :][:, cols].copy()
    if x.has_canonical_format and x.has_sorted_indices:
        source = x
    else:
        source = x.copy()
        source.sum_duplicates()
        source.sort_indices()
    index_dtype = source.indices.dtype
    if index_dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        index_dtype = np.dtype(np.int64)
    indptr = np.ascontiguousarray(source.indptr, dtype=index_dtype)
    indices = np.ascontiguousarray(source.indices, dtype=index_dtype)
    col_map = np.full(source.shape[1], -1, dtype=index_dtype)
    col_map[cols] = np.arange(cols.size, dtype=index_dtype)
    # Rows may be selected repeatedly, so source.nnz is not a safe capacity.
    # This is an upper bound; column filtering can only reduce it.
    capacity = int(np.sum(indptr[rows + 1] - indptr[rows], dtype=np.int64))
    out_indptr = np.empty(rows.size + 1, dtype=index_dtype)
    out_indices = np.empty(max(capacity, 1), dtype=index_dtype)
    out_data = np.empty(max(capacity, 1), dtype=np.float64)
    if rows.size and cols.size and source.nnz:
        kernel = (lib().mad_csr_take_f64_i32 if index_dtype == np.dtype(np.int32)
                  else lib().mad_csr_take_f64)
        written = kernel(addr(indptr), addr(indices), addr(source.data), addr(rows),
                         addr(col_map), addr(out_indptr), addr(out_indices), addr(out_data),
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
                 filemode=None, asview=False, obsp=None, varp=None, oidx=None, vidx=None,
                 *, _owns=False):
        # `_owns` marks a private call whose arrays this object may hold without
        # copying: the slicing paths pass freshly gathered arrays they own.
        if asview or oidx is not None or vidx is not None:
            raise NotImplementedError("views are not implemented; slicing returns an eager AnnData")
        if filename is not None or filemode is not None:
            raise NotImplementedError("backed .h5ad files are outside the in-memory slicing subset")
        if raw is not None:
            raise NotImplementedError("raw is outside the covered annotated-slicing subset")
        if isinstance(X, AnnData):
            # Matches anndata.AnnData._init_as_actual: build from another AnnData.
            if any(item is not None for item in (obs, var, uns, obsm, varm, obsp, varp)):
                raise ValueError("If `X` is a dict no further arguments must be provided.")
            obs, var, uns = X.obs, X.var, X.uns
            obsm, varm, obsp, varp = X.obsm, X.varm, X.obsp, X.varp
            layers = dict(X.layers)
            X = X.layers.get(None)
        if layers is not None and None in layers:
            # Matches anndata.AnnData._init_as_actual: layers[None] replaces X.
            if X is not None and X is not layers[None]:
                raise ValueError("If you provide `layers[None]` and `X`, they must be identical.")
            X = layers[None]
        if X is None:
            if shape is None:
                raise ValueError("X or shape is required")
            X = np.zeros(shape, dtype=dtype or np.float64)
        if _owns:
            if dtype is not None or not isinstance(layers or {}, dict):
                raise TypeError("_owns takes no dtype conversion and a plain dict of layers")
            self.X = X if isinstance(X, np.ndarray) else _matrix(X)
        else:
            self.X = _matrix(X)
            if dtype is not None:
                self.X = self.X.astype(dtype)
            # Matches anndata.AnnData.X: an array source is stored, not copied, so
            # `layers[None] is adata.X` holds.
            if isinstance(X, np.ndarray) and dtype is None:
                self.X = X
        supplied = (obs is not None, var is not None)
        if isinstance(X, pd.DataFrame):
            # Matches anndata.AnnData._init_as_actual: a DataFrame X supplies
            # obs_names/var_names when those frames are absent or default.
            if obs is None:
                obs = pd.DataFrame(index=X.index)
            if var is None:
                var = pd.DataFrame(index=X.columns)
        self.obs = _frame(obs, self.n_obs, "observations")
        self.var = _frame(var, self.n_vars, "variables")
        if isinstance(X, pd.DataFrame):
            for attr, label, given in (("obs", "index", supplied[0]), ("var", "columns", supplied[1])):
                if given:
                    continue
                frame = getattr(self, attr)
                source = X.index if attr == "obs" else X.columns
                if (frame.index.equals(pd.Index([str(i) for i in range(len(source))]))
                        and not source.equals(frame.index)):
                    # Matches anndata.AnnData._init_as_actual: a default index takes
                    # X's labels; otherwise the labels must agree exactly.
                    frame.index = source
                elif not frame.index.equals(source):
                    raise ValueError(f"Index of {attr} must match {label} of X.")
        # Matches anndata.AnnData._init_as_actual (`self._uns = uns`): a slice
        # shares the parent's dict, while copy() rebuilds and deep-copies it.
        self.uns = _copy.deepcopy(dict(uns or {}))
        self.raw = None
        self.layers = {key: value if _owns else _checked_layer(value, self.shape, key)
                       for key, value in (layers or {}).items()}
        # Matches anndata.AnnData: `layers[None]` is the very same object as `.X`.
        self.layers[None] = self.X
        if _owns:
            self.obsm, self.varm = obsm or {}, varm or {}
        else:
            self.obsm = {key: _checked_axis(value, self.n_obs, "obsm", key) for key, value in (obsm or {}).items()}
            self.varm = {key: _checked_axis(value, self.n_vars, "varm", key) for key, value in (varm or {}).items()}
        self.obsp = {key: _matrix(value, f"obsp[{key!r}]") for key, value in (obsp or {}).items()}
        self.varp = {key: _matrix(value, f"varp[{key!r}]") for key, value in (varp or {}).items()}
        for collection, expected in (("obsp", (self.n_obs, self.n_obs)),
                                     ("varp", (self.n_vars, self.n_vars))):
            for key, value in getattr(self, collection).items():
                if value.shape != expected:
                    raise ValueError(f"{collection}[{key!r}] must have shape {expected}")
        # Matches anndata.AnnData._check_uniqueness warning via utils.warn_names_duplicates.
        for attr in ("obs", "var"):
            index = getattr(self, attr).index
            if index[~index.isna()].has_duplicates:
                names = "Observation" if attr == "obs" else "Variable"
                warnings.warn(f"{names} names are not unique. To make them unique, "
                              f"call `.{attr}_names_make_unique`.", UserWarning, stacklevel=2)

    @property
    def shape(self):
        # Matches anndata.AnnData.shape: only the first two axes count.
        return self.X.shape[:2]

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

    def __getattr__(self, name):
        # Matches anndata.AnnData: `.obsm`, `.obsp`, `.layers`, `.varm`, `.varp`,
        # `.raw` and `.X` are readable through `AnnData.<name>`; they all return
        # plain objects here because this port has no element references.
        if name in {"X", "obsm", "obsp", "layers", "varm", "varp", "raw"}:
            return object.__getattribute__(self, name)
        raise AttributeError(f"'AnnData' object has no attribute {name!r}")

    def __repr__(self):
        return f"AnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}"

    def __getitem__(self, index):
        obs_index, var_index = _unpack_index(index)
        rows = _positions(obs_index, self.obs_names, self.n_obs)
        cols = _positions(var_index, self.var_names, self.n_vars)
        # Every array below is freshly allocated by this call, so `_owns` lets the
        # result hold them instead of copying each one a second time.
        result = AnnData(_take_matrix(self.X, rows, cols), obs=self.obs.iloc[rows].copy(),
                         var=self.var.iloc[cols].copy(), uns=self.uns,
                         layers={k: _take_matrix(v, rows, cols) for k, v in self.layers.items()
                                 if k is not None},
                         obsm={k: _take_axis(v, rows) for k, v in self.obsm.items()},
                         varm={k: _take_axis(v, cols) for k, v in self.varm.items()},
                         obsp={k: _take_matrix(v, rows, rows) for k, v in self.obsp.items()},
                         varp={k: _take_matrix(v, cols, cols) for k, v in self.varp.items()},
                         _owns=True)
        # Matches anndata.AnnData._init_as_view: the view shares the parent's uns
        # dict (there are no lazy views here, only eager arrays).
        result.uns = self.uns
        return result

    def copy(self, filename=None):
        if filename is not None:
            raise NotImplementedError("backed copies are not implemented")
        # Matches anndata.AnnData._copy: independent arrays and a deep-copied uns,
        # unlike `self[:, :]`, which shares uns with its parent.
        rows = np.arange(self.n_obs, dtype=np.int64)
        cols = np.arange(self.n_vars, dtype=np.int64)
        result = AnnData(_take_matrix(self.X, rows, cols), obs=self.obs.copy(),
                         var=self.var.copy(), uns=_copy.deepcopy(self.uns),
                         layers={k: _take_matrix(v, rows, cols) for k, v in self.layers.items()
                                 if k is not None},
                         obsm={k: _take_axis(v, rows) for k, v in self.obsm.items()},
                         varm={k: _take_axis(v, cols) for k, v in self.varm.items()},
                         obsp={k: _take_matrix(v, rows, rows) for k, v in self.obsp.items()},
                         varp={k: _take_matrix(v, cols, cols) for k, v in self.varp.items()},
                         _owns=True)
        return result

    def to_memory(self, copy=False):
        return self.copy() if copy else self

    def to_df(self, layer=None):
        matrix = self.layers[layer] if layer is not None else self.X
        values = matrix.toarray() if sparse.issparse(matrix) else matrix
        return pd.DataFrame(values, index=self.obs_names, columns=self.var_names)

    def obs_vector(self, k, *, layer=None):
        # Matches anndata.index._get_vector_ambiguous for dim="obs".
        matrix = self.layers[layer] if layer is not None else self.X
        if k in self.obs.columns and k in self.var_names:
            raise ValueError(f"Key {k} could be found in both .var_names and .obs.columns")
        if k in self.obs.columns:
            return np.asarray(self.obs[k])
        if k not in self.var_names:
            raise KeyError(f"Could not find key {k} in .var_names or .obs.columns.")
        return np.asarray(matrix[:, self.var_names.get_loc(k)]).ravel()

    def var_vector(self, k, *, layer=None):
        # Matches anndata.index._get_vector_ambiguous for dim="var".
        matrix = self.layers[layer] if layer is not None else self.X
        if k in self.var.columns and k in self.obs_names:
            raise ValueError(f"Key {k} could be found in both .obs_names and .var.columns")
        if k in self.var.columns:
            return np.asarray(self.var[k])
        if k not in self.obs_names:
            raise KeyError(f"Could not find key {k} in .obs_names or .var.columns.")
        return np.asarray(matrix[self.obs_names.get_loc(k), :]).ravel()

    def obs_keys(self):
        return list(self.obs.columns)

    def var_keys(self):
        return list(self.var.columns)

    def obsm_keys(self):
        return list(self.obsm)

    def varm_keys(self):
        return list(self.varm)

    def uns_keys(self):
        # Matches anndata.AnnData.uns_keys (deprecated upstream since 0.13).
        return sorted(self.uns)

    def layers_keys(self):
        # Upstream 0.13.2 dropped this helper; it returns the `.layers` keys that
        # 0.12's `layers_keys` returned, including the `None` key for `.X`.
        return list(self.layers)


def _checked_layer(value, shape, key):
    matrix = _matrix(value, f"layers[{key!r}]")
    if matrix.shape != shape:
        raise ValueError(f"layers[{key!r}] has shape {matrix.shape}, expected {shape}")
    return matrix


def _checked_axis(value, n, collection, key):
    if isinstance(value, pd.DataFrame):
        result = value.copy()
    elif sparse.issparse(value) or value.ndim != 2:
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
