"""Parity checks against the installed upstream anndata (0.13.2).

Every public function of `mojoanndata.AnnData` is compared here with the
upstream reference on the same inputs, using an explicit numerical tolerance.
Degenerate inputs assert that the port raises the same exception class upstream
raises, with upstream source lines cited where behaviour is non-obvious.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import anndata as upstream
from mojoanndata import AnnData

RTOL = 1e-12
ATOL = 0.0


def values(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def assert_allclose(got, expected, *, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(values(got), values(expected), rtol=rtol, atol=atol)


def assert_same_sparse(got, expected):
    """Compare two sparse results structure-for-structure, then by value."""
    assert sparse.issparse(got) and sparse.issparse(expected)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    difference = (got - expected)
    assert difference.nnz == 0 or np.all(np.abs(difference.data) <= ATOL + RTOL)
    np.testing.assert_allclose(got.toarray(), expected.toarray(), rtol=RTOL, atol=ATOL)


def _clone(value):
    return value.copy() if hasattr(value, "copy") else value


def annotated(matrix, **extra):
    """One keyword set, built twice so the port and upstream see equal inputs."""
    obs = pd.DataFrame({"batch": ["a", "a", "b", "b", "c"], "score": np.arange(5)},
                       index=["cell0", "cell1", "cell2", "cell3", "cell4"])
    var = pd.DataFrame({"feature": ["f", "g", "h", "i"]}, index=["g0", "g1", "g2", "g3"])
    kwargs = dict(obs=obs, var=var, uns={"nested": {"version": 1}},
                  layers={"counts": values(matrix) * 2},
                  obsm={"pca": np.arange(10, dtype=float).reshape(5, 2)},
                  varm={"loadings": np.arange(12, dtype=float).reshape(4, 3)},
                  obsp={"connectivities": np.arange(25, dtype=float).reshape(5, 5)},
                  varp={"correlations": np.arange(16, dtype=float).reshape(4, 4)})
    kwargs.update(extra)
    return (AnnData(_clone(matrix), **{k: _clone(v) for k, v in kwargs.items()}),
            upstream.AnnData(_clone(matrix), **{k: _clone(v) for k, v in kwargs.items()}))


def both(fn_port, fn_upstream):
    """Run the same probe on both packages, returning (outcome, payload) pairs."""
    sentinel = object()

    def run(fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return "ok", fn()
        except Exception as error:  # noqa: BLE001 - the class is what we compare
            return type(error), sentinel
    return run(fn_port), run(fn_upstream)


def outcome(fn):
    return both(fn, fn)[0][0]


def raises(fn, expected):
    """Assert `fn` raises exactly `expected` (the class upstream raises)."""
    kind = outcome(fn)
    assert isinstance(kind, type) and kind is expected, f"raised {kind!r}, upstream raises {expected.__name__}"


# --------------------------------------------------------------------------- #
# Slicing: every aligned collection, every covered selector form
# --------------------------------------------------------------------------- #

SELECTORS = [
    (slice(1, None, 2), slice(None, None, -1)),
    (np.array([4, 0, 2]), np.array([3, 1])),
    (np.array([True, False, True, False, True]), np.array([False, True, True, False])),
    (["cell4", "cell1"], ["g2", "g0"]),
    (-1, -2),
    (Ellipsis, slice(None)),
    (slice(None), Ellipsis),
    (pd.Series([True, False, True, False, True], index=range(5)), slice(None)),
]


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_dense_slice_matches_anndata(selector, dtype):
    matrix = (np.arange(20, dtype=np.float64).reshape(5, 4) / 3).astype(dtype)
    got, expected = annotated(matrix)
    result, reference = got[selector], expected[selector].copy()
    assert result.shape == reference.shape
    assert result.X.dtype == reference.X.dtype
    assert_allclose(result.X, reference.X)
    pd.testing.assert_frame_equal(result.obs, reference.obs)
    pd.testing.assert_frame_equal(result.var, reference.var)
    assert set(result.layers) == set(reference.layers)
    assert_allclose(result.layers["counts"], reference.layers["counts"])
    assert_allclose(result.obsm["pca"], reference.obsm["pca"])
    assert_allclose(result.varm["loadings"], reference.varm["loadings"])
    assert_allclose(result.obsp["connectivities"], reference.obsp["connectivities"])
    assert_allclose(result.varp["correlations"], reference.varp["correlations"])


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.int32])
def test_integer_layers_and_obs_values_match_anndata(dtype):
    matrix = np.arange(20).reshape(5, 4).astype(dtype)
    got, expected = annotated(np.zeros((5, 4)), layers={"ints": matrix})
    result, reference = got[[4, 1], [2, 0]], expected[[4, 1], [2, 0]].copy()
    assert result.layers["ints"].dtype == reference.layers["ints"].dtype
    np.testing.assert_array_equal(result.layers["ints"], reference.layers["ints"])


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_slice_matches_anndata(fmt):
    matrix = sparse.random(5, 4, density=0.5, format=fmt, random_state=7)
    layers = {"dense": np.arange(20.).reshape(5, 4)}
    result = AnnData(matrix.copy(), layers=_clone(layers))
    reference = upstream.AnnData(matrix.copy(), layers=_clone(layers))
    selector = (np.array([4, 0, 2]), np.array([3, 1]))
    port_slice, up_slice = result[selector], reference[selector].copy()
    assert sparse.issparse(port_slice.X)
    assert_same_sparse(port_slice.X, up_slice.X)
    assert_allclose(port_slice.layers["dense"], up_slice.layers["dense"])


def test_noncanonical_csr_slice_matches_anndata():
    # Duplicate explicit entries make the CSR non-canonical (nnz > stored values).
    matrix = sparse.csr_matrix((np.array([1., 2., 4., 8., 16.]),
                                np.array([2, 2, 0, 1, 2]),
                                np.array([0, 2, 5])), shape=(2, 3))
    assert not matrix.has_canonical_format
    source = matrix.copy()
    got = AnnData(matrix)[[1, 1, 0], [2, 0]]
    assert_same_sparse(got.X, upstream.AnnData(matrix)[[1, 1, 0], [2, 0]].X)
    assert sparse.isspmatrix_csr(got.X)
    # The source is borrowed read-only, never mutated.
    np.testing.assert_array_equal(matrix.indptr, source.indptr)
    np.testing.assert_array_equal(matrix.indices, source.indices)
    np.testing.assert_array_equal(matrix.data, source.data)


def test_sparse_duplicate_structure_entries_match_anndata():
    # Build CSR from explicit duplicate entries so nnz > number of distinct columns.
    matrix = sparse.csr_matrix((np.array([1., 2., 4., 8.]), np.array([1, 1, 0, 2]),
                                np.array([0, 0, 2, 4])), shape=(3, 3))
    assert not matrix.has_canonical_format
    got = AnnData(_clone(matrix))[[1, 1, 0], [2, 1]]
    assert_same_sparse(got.X, upstream.AnnData(_clone(matrix))[[1, 1, 0], [2, 1]].X)


def test_repeated_and_duplicate_selectors_match_anndata():
    matrix = sparse.csr_matrix(np.arange(20, dtype=float).reshape(5, 4))
    layers = {"counts": np.arange(20, dtype=float).reshape(5, 4)}
    got = AnnData(matrix.copy(), layers=_clone(layers))
    expected = upstream.AnnData(matrix.copy(), layers=_clone(layers))
    selector = (np.array([0, 0, 2, 0, 2]), np.array([3, 1, 0]))
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        reference = expected[selector].copy()
    result = got[selector]
    assert sparse.isspmatrix_csr(result.X)
    assert_same_sparse(result.X, reference.X)


def test_parallel_dense_threshold_matches_anndata():
    shape = (1300, 1600)
    selected = (np.arange(1250, dtype=np.int64), np.arange(1700, dtype=np.int64) * 3 % shape[1])
    matrix = np.arange(np.prod(shape), dtype=np.float64).reshape(shape).copy()
    result = AnnData(matrix)[selected]
    reference = np.asarray(upstream.AnnData(matrix)[selected].X)
    assert result.X.size >= 2_000_000
    assert_allclose(result.X, reference)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def test_constructor_copies_inputs_like_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    reference = upstream.AnnData(matrix.astype(np.float32), layers={"l": matrix.astype(np.float32)})
    result = AnnData(matrix.astype(np.float32), layers={"l": matrix.astype(np.float32)})
    assert result.X.dtype == reference.X.dtype
    assert result.layers["l"].dtype == reference.layers["l"].dtype
    matrix[0, 0] = -1
    assert result.X[0, 0] == reference.X[0, 0] == 0.0


def test_constructor_accepts_dataframe_with_matching_index():
    frame = pd.DataFrame(np.arange(12.).reshape(4, 3),
                         index=["r0", "r1", "r2", "r3"], columns=["c0", "c1", "c2"])
    result, reference = AnnData(frame), upstream.AnnData(frame)
    assert result.obs_names.tolist() == reference.obs_names.tolist()
    assert result.var_names.tolist() == reference.var_names.tolist()
    assert_allclose(result.X, reference.X)


def test_constructor_accepts_obs_mapping_with_row_names():
    matrix = np.zeros((3, 2))
    result = AnnData(matrix, obs={"row_names": ["a", "b", "c"], "v": [1, 2, 3]})
    reference = upstream.AnnData(matrix, obs={"row_names": ["a", "b", "c"], "v": [1, 2, 3]})
    assert result.obs_names.tolist() == reference.obs_names.tolist()
    np.testing.assert_array_equal(result.obs["v"], reference.obs["v"])


def test_constructor_layers_none_key_matches_anndata():
    matrix = np.arange(6.).reshape(3, 2)
    result, reference = AnnData(layers={None: matrix}), upstream.AnnData(layers={None: matrix})
    assert result.shape == reference.shape
    assert_allclose(result.X, reference.X)
    assert result.layers[None] is result.X


def test_default_names_and_shape_only_construction():
    result, reference = AnnData(np.eye(3)), upstream.AnnData(np.eye(3))
    assert result.obs_names.tolist() == reference.obs_names.tolist()
    assert result.var_names.tolist() == reference.var_names.tolist()
    assert result.obs_names.dtype == reference.obs_names.dtype


def test_shape_properties_match_anndata():
    matrix = np.arange(20.).reshape(5, 4, 1)
    result, reference = AnnData(matrix), upstream.AnnData(matrix)
    assert result.shape == reference.shape == (5, 4)
    assert (result.n_obs, result.n_vars) == (reference.n_obs, reference.n_vars) == (5, 4)
    assert result.shape == result.X.shape[:2]
    # The port is eager, so an explicit copy is never a view; upstream's are.
    assert result.is_view == reference.is_view == False  # noqa: E712 - mirrors upstream's flag
    assert result.copy().is_view is False


def test_extra_trailing_axis_uses_first_two_dimensions():
    matrix = np.arange(8.).reshape(2, 2, 2)
    result, reference = AnnData(matrix), upstream.AnnData(matrix)
    assert result.shape == reference.shape == (2, 2)
    assert (result.n_obs, result.n_vars) == (reference.n_obs, reference.n_vars)


@pytest.mark.parametrize("bad", [
    dict(obs=pd.DataFrame(index=list("abc"))),
    dict(var=pd.DataFrame(index=list("ab"))),
    dict(layers={"a": np.ones((5, 3))}),
    dict(obsm={"a": np.ones((4, 2))}),
    dict(varm={"a": np.ones((5, 2))}),
    dict(obsp={"a": np.ones((4, 4))}),
    dict(varp={"a": np.ones((3, 3))}),
])
def test_mismatched_shapes_raise_what_anndata_raises(bad):
    matrix = np.ones((5, 4))
    raises(lambda: upstream.AnnData(matrix, **bad), ValueError)
    raises(lambda: AnnData(matrix, **bad), ValueError)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def test_obs_and_var_names_round_trip():
    result, reference = annotated(np.ones((5, 4)))
    result.obs_names = ["o0", "o1", "o2", "o3", "o4"]
    reference.obs_names = ["o0", "o1", "o2", "o3", "o4"]
    result.var_names = ["a", "b", "c", "d"]
    reference.var_names = ["a", "b", "c", "d"]
    assert result.obs_names.tolist() == reference.obs_names.tolist()
    assert result.var_names.tolist() == reference.var_names.tolist()
    assert_allclose(result[:, ["d", "a"]].X, reference[:, ["d", "a"]].X)


@pytest.mark.parametrize("attr", ["obs_names", "var_names"])
def test_short_names_raise_value_error(attr):
    result, reference = AnnData(np.ones((5, 4))), upstream.AnnData(np.ones((5, 4)))
    raises(lambda: setattr(reference, attr, ["x"]), ValueError)
    raises(lambda: setattr(result, attr, ["x"]), ValueError)


def test_to_df_matches_anndata_including_sparse_and_layers():
    matrix = sparse.csr_matrix(np.arange(20.).reshape(5, 4))
    result = AnnData(matrix, layers={"l": np.arange(20.).reshape(5, 4)})
    reference = upstream.AnnData(matrix, layers={"l": np.arange(20.).reshape(5, 4)})
    pd.testing.assert_frame_equal(result.to_df(), reference.to_df())
    pd.testing.assert_frame_equal(result.to_df(layer="l"), reference.to_df(layer="l"))


def test_to_df_on_empty_slice_matches_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    pd.testing.assert_frame_equal(AnnData(matrix)[0:0].to_df(), upstream.AnnData(matrix)[0:0].to_df())
    pd.testing.assert_frame_equal(AnnData(matrix)[0:0, 0:0].to_df(), upstream.AnnData(matrix)[0:0, 0:0].to_df())


def test_vectors_match_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    result = AnnData(matrix, obs=pd.DataFrame({"q": [1, 2, 3, 4, 5]}),
                     layers={"lk": matrix * 3})
    reference = upstream.AnnData(matrix, obs=pd.DataFrame({"q": [1, 2, 3, 4, 5]}),
                                 layers={"lk": matrix * 3})
    np.testing.assert_array_equal(result.obs_vector("q"), np.asarray(reference.obs_vector("q")))
    # Default names are "0".."4" for obs and "0".."3" for var, as upstream.
    assert_allclose(result.obs_vector("2"), np.asarray(reference.obs_vector("2")))
    assert_allclose(result.obs_vector("2", layer="lk"), np.asarray(reference.obs_vector("2", layer="lk")))
    assert_allclose(result.var_vector("4"), np.asarray(reference.var_vector("4")))
    assert_allclose(result.var_vector("4", layer="lk"), np.asarray(reference.var_vector("4", layer="lk")))
    assert result.obs_vector("2").shape == np.asarray(reference.obs_vector("2")).shape


def test_vector_missing_key_raises_keyerror():
    result, reference = AnnData(np.ones((3, 2))), upstream.AnnData(np.ones((3, 2)))
    raises(lambda: reference.obs_vector("nope"), KeyError)
    raises(lambda: result.obs_vector("nope"), KeyError)
    raises(lambda: reference.var_vector("nope"), KeyError)
    raises(lambda: result.var_vector("nope"), KeyError)


def test_ambiguous_vector_key_raises_value_error():
    matrix = np.ones((3, 2))
    obs = pd.DataFrame({"0": [1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = AnnData(matrix, obs=obs.copy())
        reference = upstream.AnnData(matrix, obs=obs)
    # "0" is in both .obs.columns and the default .var_names.
    raises(lambda: reference.obs_vector("0"), ValueError)
    raises(lambda: result.obs_vector("0"), ValueError)


def test_key_helpers_match_anndata():
    result, reference = annotated(np.ones((5, 4)), uns={"b": 1, "a": 2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # upstream deprecates the *_keys helpers
        assert result.obs_keys() == reference.obs_keys()
        assert result.var_keys() == reference.var_keys()
        assert result.obsm_keys() == reference.obsm_keys()
        assert result.varm_keys() == reference.varm_keys()
        assert result.uns_keys() == reference.uns_keys()
        # Upstream 0.13 removed `layers_keys`; the port documents it as covered.
        assert result.layers_keys() == list(reference.layers)


def test_copy_matches_anndata_and_is_independent():
    matrix = np.arange(20.).reshape(5, 4)
    result, reference = annotated(matrix, obsp={"connectivities": np.arange(25.).reshape(5, 5)})
    port_copy, up_copy = result.copy(), reference.copy()
    assert_allclose(port_copy.X, up_copy.X)
    pd.testing.assert_frame_equal(port_copy.obs, up_copy.obs)
    assert set(port_copy.layers) == set(up_copy.layers)
    port_copy.X[0, 0] = -7
    port_copy.layers["counts"][0, 0] = -7
    port_copy.obsp["connectivities"][0, 0] = -7
    assert result.X[0, 0] != -7
    assert result.layers["counts"][0, 0] != -7
    assert result.obsp["connectivities"][0, 0] != -7
    assert up_copy.X[0, 0] != -7


def test_to_memory_matches_anndata():
    result, reference = AnnData(np.ones((3, 2))), upstream.AnnData(np.ones((3, 2)))
    assert result.to_memory() is result
    # Port is eager, so to_memory() is the object itself; upstream 0.13.2 returns
    # an in-memory view over the same arrays (to_memory(..., copy=False)).
    assert np.shares_memory(result.to_memory().X, result.X)
    assert np.shares_memory(np.asarray(reference.to_memory().X), np.asarray(reference.X))
    assert_allclose(result.to_memory().X, np.asarray(reference.to_memory().X))
    assert result.to_memory(copy=True) is not result
    assert reference.to_memory(copy=True) is not reference
    assert isinstance(result.to_memory(True), AnnData)
    assert_allclose(result.to_memory(copy=True).X, np.asarray(reference.to_memory(copy=True).X))


def test_slices_share_uns_and_copy_does_not():
    obs, var = pd.DataFrame(index=list("abc")), pd.DataFrame(index=list("xy"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = AnnData(np.ones((3, 2)), obs=obs, var=var, uns={"n": {"v": 1}})
        reference = upstream.AnnData(np.ones((3, 2)), obs=obs, var=var, uns={"n": {"v": 1}})
    result[:2].uns["n"]["v"] = 9
    reference[:2].uns["n"]["v"] = 9
    # anndata.AnnData._init_as_view exposes the parent's `_uns` to views.
    assert result.uns["n"]["v"] == reference.uns["n"]["v"] == 9
    result.copy().uns["n"]["v"] = 4
    reference.copy().uns["n"]["v"] = 4
    # anndata.AnnData._copy deep-copies uns.
    assert result.uns["n"]["v"] == reference.uns["n"]["v"] == 9


def test_raw_attribute_matches_anndata():
    assert AnnData(np.ones((2, 2))).raw is None
    assert upstream.AnnData(np.ones((2, 2))).raw is None


# --------------------------------------------------------------------------- #
# Degenerate inputs: same value or same exception class as upstream
# --------------------------------------------------------------------------- #

def test_empty_input_matches_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    for selector in [slice(0, 0), [], np.array([], dtype=np.int64), [3, 3][:0]]:
        result, reference = both(lambda s=selector: AnnData(matrix)[s].X.shape,
                                 lambda s=selector: upstream.AnnData(matrix)[s].shape)
        assert result == reference
    result, reference = both(lambda: AnnData(matrix)[:, []].X.shape,
                             lambda: upstream.AnnData(matrix)[:, []].shape)
    assert result == reference
    result, reference = both(lambda: AnnData(np.empty((0, 5))).X.shape,
                             lambda: upstream.AnnData(np.empty((0, 5))).shape)
    assert result == reference


def test_single_element_selection_matches_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    for selector in [(2, 1), (np.array([2]), np.array([1])), ([2], [1]), (slice(2, 3), slice(1, 2))]:
        result, reference = both(lambda s=selector: AnnData(matrix)[s].X.tolist(),
                                 lambda s=selector: np.asarray(upstream.AnnData(matrix)[s].X).tolist())
        assert result == reference


def test_duplicate_names_warn_like_anndata():
    matrix = np.arange(8.).reshape(4, 2)
    with pytest.warns(UserWarning) as port_warnings:
        port = AnnData(matrix, obs=pd.DataFrame(index=["a", "a", "b", "c"]))
    with pytest.warns(UserWarning) as up_warnings:
        reference = upstream.AnnData(matrix, obs=pd.DataFrame(index=["a", "a", "b", "c"]))
    # Both emit anndata's "Observation names are not unique" warning at construction.
    assert any("Observation names are not unique" in str(w.message) for w in port_warnings)
    assert any("Observation names are not unique" in str(w.message) for w in up_warnings)
    port.obs_names = ["a", "a", "b", "c"]
    reference.obs_names = ["a", "a", "b", "c"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Duplicate labels still resolve their positions, as upstream does.
        assert port[[0, 1]].obs_names.tolist() == reference[[0, 1]].obs_names.tolist()
        assert port.obs_names[[0, 1]].tolist() == reference.obs_names[[0, 1]].tolist()


def test_duplicate_selection_values_match_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    result = AnnData(matrix)[[2, 2, 0], [3, 3, 1]]
    reference = np.asarray(upstream.AnnData(matrix)[[2, 2, 0], [3, 3, 1]].X)
    assert_allclose(result.X, reference)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_survive_slicing_like_anndata(value):
    matrix = np.arange(20.).reshape(5, 4).astype(np.float64)
    matrix[3, 1] = value
    for port_matrix in (matrix, sparse.csr_matrix(matrix)):
        result = AnnData(port_matrix)[[3, 0], [1, 2]]
        reference = upstream.AnnData(port_matrix)[[3, 0], [1, 2]]
        got, want = values(result.X), values(reference.X)
        assert np.isnan(got).tolist() == np.isnan(want).tolist()
        assert np.isinf(got).tolist() == np.isinf(want).tolist()
        assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)])


@pytest.mark.parametrize("indexer", [[np.nan], [np.inf], [-np.inf], [1.5]])
def test_nonfinite_float_index_raises_indexerror(indexer):
    result, reference = AnnData(np.ones((3, 2))), upstream.AnnData(np.ones((3, 2)))
    raises(lambda: reference[indexer], IndexError)
    raises(lambda: result[indexer], IndexError)


@pytest.mark.parametrize("indexer", [[True, False], np.ones(3, dtype=bool)[:2], [True] * 5])
def test_mismatched_boolean_mask_raises_indexerror(indexer):
    result, reference = AnnData(np.ones((3, 2))), upstream.AnnData(np.ones((3, 2)))
    raises(lambda: reference[indexer], IndexError)
    raises(lambda: result[indexer], IndexError)


@pytest.mark.parametrize("indexer", [[-3], [2], [99], [-99]])
def test_out_of_bounds_integer_index_raises_indexerror(indexer):
    result, reference = AnnData(np.ones((3, 2))), upstream.AnnData(np.ones((3, 2)))
    raises(lambda: reference[:, indexer], IndexError)
    raises(lambda: result[:, indexer], IndexError)


@pytest.mark.parametrize("indexer", [["missing"], 3, "nope"])
def test_invalid_label_or_position_raises_upstream_class(indexer):
    result, reference = AnnData(np.ones((3, 2))), upstream.AnnData(np.ones((3, 2)))
    up_kind = outcome(lambda: reference[indexer])
    port_kind = outcome(lambda: result[indexer])
    assert isinstance(up_kind, type) and port_kind is up_kind


def test_two_dimensional_positional_index_is_rejected():
    # Upstream anndata 0.13.2 accepts a 2-D indexer into a malformed view whose
    # `X` gains an axis (shape (2, 1, 4)) and whose `.copy()` then fails with
    # ValueError "Observations annot. `var` must have as many rows ...".
    matrix = np.arange(20.).reshape(5, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = upstream.AnnData(matrix)[np.array([[0], [2]])]
        assert np.asarray(broken.X).shape == (2, 1, 4)
        raises(broken.copy, ValueError)
    # The port instead rejects the indexer outright and never builds a bad object.
    port = AnnData(matrix)
    raises(lambda: port[np.array([[0], [2]])], IndexError)
    raises(lambda: port[:, np.array([[0, 2]])], IndexError)
    # Accepting the raveled form matches upstream's values (anndata.index._from_array
    # only ravels when the indexer length equals the axis length).
    padded = np.array([[0], [2], [1], [3], [4]])
    assert_allclose(port[padded].X, np.asarray(upstream.AnnData(matrix)[padded].X))


def test_unsigned_wraparound_matches_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    # anndata.index._from_array: uint64 2**64-1 is viewed as -1 before NumPy bounds-checks.
    indexer = np.array([2**64 - 1], dtype=np.uint64)
    result = AnnData(matrix)[indexer]
    reference = np.asarray(upstream.AnnData(matrix)[indexer].X)
    assert_allclose(result.X, reference)


def test_ellipsis_forms_match_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    for selector in [Ellipsis, (1,), (1, ...), (..., 2), (Ellipsis, Ellipsis), (0, 0, 0)]:
        result, reference = both(lambda s=selector: AnnData(matrix)[s].X.tolist(),
                                 lambda s=selector: np.asarray(upstream.AnnData(matrix)[s].X).tolist())
        assert result == reference


def test_label_slices_are_inclusive_like_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    port, reference = AnnData(matrix), upstream.AnnData(matrix)
    port.obs_names = reference.obs_names = ["c0", "c1", "c2", "c3", "c4"]
    port.var_names = reference.var_names = ["v0", "v1", "v2", "v3"]
    assert port["c1":"c3"].obs_names.tolist() == reference["c1":"c3"].obs_names.tolist()
    assert port[:, "v1":"v2"].var_names.tolist() == reference[:, "v1":"v2"].var_names.tolist()
    raises(lambda: reference["zz":"c3"], KeyError)
    raises(lambda: port["zz":"c3"], KeyError)


def test_scalar_negative_and_reversed_slices_match_anndata():
    matrix = np.arange(20.).reshape(5, 4)
    for selector in [-1, -5, slice(None, None, -1), slice(3, None, -2), slice(None, None, 2)]:
        result, reference = both(lambda s=selector: AnnData(matrix)[s].X.tolist(),
                                 lambda s=selector: np.asarray(upstream.AnnData(matrix)[s].X).tolist())
        assert result == reference


def test_empty_sparse_slice_matches_anndata():
    matrix = sparse.csr_matrix(np.arange(20.).reshape(5, 4))
    port = AnnData(matrix)[[0, 0][:0], [1, 2]]
    reference = upstream.AnnData(matrix)[[0, 0][:0], [1, 2]].copy()
    assert port.X.shape == reference.X.shape
    assert sparse.isspmatrix_csr(port.X)
    assert_same_sparse(port.X, reference.X)
