from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import anndata as upstream
from mojoanndata import AnnData


def fixture(matrix):
    obs = pd.DataFrame({"batch": ["a", "a", "b", "b", "c"], "score": np.arange(5)},
                       index=["cell0", "cell1", "cell2", "cell3", "cell4"])
    var = pd.DataFrame({"feature": ["f", "g", "h", "i"]}, index=["g0", "g1", "g2", "g3"])
    kwargs = dict(obs=obs, var=var, uns={"nested": {"version": 1}},
                  layers={"counts": matrix * 2},
                  obsm={"pca": np.arange(10, dtype=float).reshape(5, 2)},
                  varm={"loadings": np.arange(12, dtype=float).reshape(4, 3)},
                  obsp={"connectivities": np.arange(25, dtype=float).reshape(5, 5)},
                  varp={"correlations": np.arange(16, dtype=float).reshape(4, 4)})
    return AnnData(matrix, **kwargs), upstream.AnnData(matrix, **kwargs)


def values(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def assert_same(got, expected):
    assert got.shape == expected.shape
    np.testing.assert_allclose(values(got.X), values(expected.X))
    pd.testing.assert_frame_equal(got.obs, expected.obs)
    pd.testing.assert_frame_equal(got.var, expected.var)
    assert set(got.layers) == set(expected.layers)
    np.testing.assert_allclose(values(got.layers["counts"]), values(expected.layers["counts"]))
    np.testing.assert_allclose(values(got.obsm["pca"]), values(expected.obsm["pca"]))
    np.testing.assert_allclose(values(got.varm["loadings"]), values(expected.varm["loadings"]))
    np.testing.assert_allclose(values(got.obsp["connectivities"]), values(expected.obsp["connectivities"]))
    np.testing.assert_allclose(values(got.varp["correlations"]), values(expected.varp["correlations"]))


@pytest.mark.parametrize("selector", [
    (slice(1, None, 2), slice(None, None, -1)),
    (np.array([4, 0, 2]), np.array([3, 1])),
    (np.array([True, False, True, False, True]), np.array([False, True, True, False])),
    (["cell4", "cell1"], ["g2", "g0"]),
    (-1, -2),
])
def test_dense_slicing_matches_anndata(selector):
    matrix = np.arange(20, dtype=np.float64).reshape(5, 4) / 3
    got, expected = fixture(matrix)
    assert_same(got[selector], expected[selector].copy())


def test_csr_slicing_matches_anndata_for_masks_and_fancy_indices():
    matrix = sparse.csr_matrix(np.array([
        [0., 2., 0., 4.], [5., 0., 7., 0.], [0., 0., 9., 10.],
        [11., 12., 0., 0.], [0., 14., 15., 0.],
    ]))
    got, expected = fixture(matrix)
    selector = (np.array([True, False, True, True, False]), np.array([3, 1, 2]))
    result, reference = got[selector], expected[selector].copy()
    assert sparse.isspmatrix_csr(result.X)
    assert_same(result, reference)


def test_csr_duplicate_columns_fall_back_and_match_anndata():
    matrix = sparse.csr_matrix(np.arange(20, dtype=float).reshape(5, 4))
    got, expected = fixture(matrix)
    with pytest.warns(UserWarning, match="Variable names are not unique"):
        reference = expected[[3, 1], [2, 2, 0]].copy()
    assert_same(got[[3, 1], [2, 2, 0]], reference)


def test_csr_repeated_rows_do_not_overrun_output_and_match_anndata():
    matrix = sparse.csr_matrix(np.arange(20, dtype=float).reshape(5, 4))
    got, expected = fixture(matrix)
    selector = (np.array([0, 0, 2, 0, 2]), np.array([3, 1, 0]))
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        reference = expected[selector].copy()
    assert_same(got[selector], reference)


@pytest.mark.parametrize(("shape", "selected_cols"), [((33, 17), 15), ((1500, 1501), 1499)])
def test_dense_simd_tail_and_parallel_threshold_match_anndata(shape, selected_cols):
    matrix = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    rows = np.arange(shape[0], dtype=np.int64)
    cols = np.arange(selected_cols, dtype=np.int64)
    got = AnnData(matrix)[rows, cols]
    expected = upstream.AnnData(matrix)[rows, cols].copy()
    np.testing.assert_allclose(got.X, expected.X)


def test_csr_int64_indices_match_anndata():
    matrix = sparse.csr_matrix(np.arange(30, dtype=np.float64).reshape(6, 5))
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)
    rows = np.array([5, 1, 1, 3], dtype=np.int64)
    cols = np.array([4, 0, 2], dtype=np.int64)
    got = AnnData(matrix)[rows, cols]
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        expected = upstream.AnnData(matrix)[rows, cols].copy()
    np.testing.assert_allclose(got.X.toarray(), expected.X.toarray())


def test_noncanonical_csr_is_canonicalized_without_mutating_source():
    matrix = sparse.csr_matrix((np.array([2., 3., 5., 7., 11.]),
                                np.array([2, 0, 2, 1, 0], dtype=np.int32),
                                np.array([0, 3, 5], dtype=np.int32)), shape=(2, 3))
    adata = AnnData(matrix)
    original_indices = adata.X.indices.copy()
    got = adata[[1, 0], [2, 0]]
    expected = upstream.AnnData(matrix)[[1, 0], [2, 0]].copy()
    np.testing.assert_allclose(got.X.toarray(), expected.X.toarray())
    np.testing.assert_array_equal(adata.X.indices, original_indices)


def test_default_axis_names_and_named_indexing():
    adata = AnnData(np.eye(3))
    assert (adata.shape, adata.n_obs, adata.n_vars) == ((3, 3), 3, 3)
    assert adata.obs_names.tolist() == ["0", "1", "2"]
    adata.obs_names = ["o0", "o1", "o2"]
    assert adata.obs_names.tolist() == ["o0", "o1", "o2"]
    adata.var_names = ["a", "b", "c"]
    np.testing.assert_allclose(adata[:, ["c", "a"]].X, np.eye(3)[:, [2, 0]])


def test_aligned_annotations_are_independent_and_uns_is_copied():
    adata, _ = fixture(np.ones((5, 4)))
    subset = adata[[1, 3], [0, 2]]
    assert subset.layers[None] is subset.X
    subset.obs.iloc[0, 0] = "changed"
    subset.uns["nested"]["version"] = 2
    assert adata.obs.iloc[1, 0] == "a"
    assert adata.uns["nested"]["version"] == 1


def test_vectors_and_to_df_match_anndata():
    matrix = np.arange(20, dtype=float).reshape(5, 4)
    got, expected = fixture(matrix)
    with pytest.warns(FutureWarning):
        reference_obs = expected.obs_vector("g2")
    with pytest.warns(FutureWarning):
        reference_var = expected.var_vector("cell3")
    np.testing.assert_allclose(got.obs_vector("g2"), reference_obs)
    np.testing.assert_allclose(got.var_vector("cell3"), reference_var)
    pd.testing.assert_frame_equal(got.to_df(), expected.to_df())


def test_key_helpers_and_to_memory():
    adata, _ = fixture(np.ones((5, 4)))
    assert adata.obs_keys() == ["batch", "score"]
    assert adata.var_keys() == ["feature"]
    assert adata.layers_keys() == ["counts", None]
    assert adata.obsm_keys() == ["pca"]
    assert adata.varm_keys() == ["loadings"]
    assert adata.to_memory() is adata
    assert adata.to_memory(copy=True) is not adata


@pytest.mark.parametrize("selector", [([True, False], slice(None)), (["missing"], slice(None)),
                                      (slice(None), [99])])
def test_invalid_indices_raise(selector):
    adata = AnnData(np.ones((3, 2)))
    with pytest.raises((IndexError, KeyError)):
        _ = adata[selector]


@pytest.mark.parametrize("selector", [([1.5], slice(None)), ([np.uint64(2**63)], slice(None))])
def test_non_integral_or_unrepresentable_integer_indices_raise(selector):
    adata = AnnData(np.ones((3, 2)))
    with pytest.raises(IndexError):
        _ = adata[selector]


def test_shape_constructor_and_copy():
    adata = AnnData(shape=(2, 3), dtype=np.float32)
    assert adata.shape == (2, 3)
    assert adata.X.dtype == np.float32
    copied = adata.copy()
    copied.X[0, 0] = 7
    assert adata.X[0, 0] == 0
