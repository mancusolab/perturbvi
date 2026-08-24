import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scipy import sparse as scipy_sparse

from jax.experimental import sparse as jax_sparse

from perturbvi import load_screen


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 48, 20
    expression = rng.normal(size=(n_cells, n_genes))
    conditions = np.resize(["control", "IRF1", "STAT1", "CEBPA"], n_cells)
    obs = pd.DataFrame(
        {
            "condition": conditions,
            "batch": np.resize(["run_1", "run_2"], n_cells),
            "depth": rng.uniform(1_000, 20_000, n_cells),
            "guide_A": np.resize([0, 1, 0, 1], n_cells),
            "guide_B": np.resize([0, 0, 1, 1], n_cells),
        },
        index=[f"cell_{index}" for index in range(n_cells)],
    )
    result = ad.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(n_genes)]),
    )
    result.layers["transformed"] = expression * 2
    result.obsm["G"] = obs["condition"].astype("string").str.get_dummies().astype(int)
    return result


@pytest.mark.parametrize("storage", ["object", "h5ad", "zarr"])
def test_key_driven_load_has_the_same_behavior_across_anndata_storage(adata, tmp_path, storage):
    source = adata
    if storage == "h5ad":
        source = tmp_path / "screen.h5ad"
        adata.write_h5ad(source)
    elif storage == "zarr":
        source = tmp_path / "screen.zarr"
        adata.write_zarr(source)

    data = load_screen(source, control="control")

    assert data.X.shape == adata.shape
    assert data.G.shape == (adata.n_obs, 3)
    assert data.gene_names == tuple(adata.var_names)
    assert data.perturbation_names == ("CEBPA", "IRF1", "STAT1")
    control_rows = adata.obs["condition"].eq("control").to_numpy()
    np.testing.assert_array_equal(np.asarray(data.G)[control_rows], 0)


def test_baseline_free_g_passes_through_without_control(adata):
    G = pd.DataFrame(
        {
            "guide_A": adata.obs["guide_A"].astype(int),
            "guide_B": adata.obs["guide_B"].astype(int),
        },
        index=adata.obs_names,
    )
    adata.obsm["G"] = G

    data = load_screen(adata)

    assert data.perturbation_names == ("guide_A", "guide_B")
    np.testing.assert_array_equal(np.asarray(data.G)[0], [0, 0])
    np.testing.assert_array_equal(np.asarray(data.G)[3], [1, 1])


def test_control_column_is_dropped_before_fitting(adata):
    data = load_screen(adata, control="control")

    assert "control" not in data.perturbation_names
    assert data.perturbation_names == ("CEBPA", "IRF1", "STAT1")


def test_missing_control_column_raises(adata):
    with pytest.raises(ValueError, match="not present"):
        load_screen(adata, control="missing")


def test_control_none_passes_stored_frame_through_unchanged(adata):
    # control=None declares G baseline-free; the loader trusts that claim and
    # cannot guess that "control" is the reference.
    data = load_screen(adata)
    assert "control" in data.perturbation_names


def test_perturbation_names_preserve_stored_column_order(adata):
    # Deliberately non-alphabetical order: names must label the exact columns
    # of the stored binary matrix, in the same order.
    G = pd.DataFrame(
        {
            "z_target": np.resize([0, 1, 0], adata.n_obs),
            "a_target": np.resize([1, 0, 0], adata.n_obs),
            "m_target": np.resize([0, 0, 1], adata.n_obs),
        },
        index=adata.obs_names,
    )
    adata.obsm["G"] = G

    data = load_screen(adata)

    assert data.perturbation_names == ("z_target", "a_target", "m_target")
    stored = G.to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(data.G), stored)
    # Each name corresponds to the matching column of the stored matrix.
    for index, name in enumerate(data.perturbation_names):
        np.testing.assert_array_equal(np.asarray(data.G)[:, index], stored[:, index])


def test_obsm_entry_must_be_a_named_dataframe(adata):
    adata.obsm["G"] = np.zeros((adata.n_obs, 2))
    with pytest.raises(ValueError, match="named pandas DataFrame"):
        load_screen(adata)


def test_missing_g_key_raises_with_guidance(adata):
    del adata.obsm["G"]
    with pytest.raises(ValueError, match="does not exist"):
        load_screen(adata)


def test_custom_g_key_is_honored(adata):
    adata.obsm["perturbations"] = adata.obsm["G"].copy()
    del adata.obsm["G"]
    data = load_screen(adata, g_key="perturbations", control="control")
    assert data.perturbation_names == ("CEBPA", "IRF1", "STAT1")


def test_named_layer_is_explicit(adata):
    default = load_screen(adata, control="control")
    selected = load_screen(adata, control="control", x_key="transformed")
    np.testing.assert_allclose(np.asarray(selected.X), np.asarray(default.X) * 2)


def test_missing_x_key_raises(adata):
    with pytest.raises(KeyError, match="does not exist"):
        load_screen(adata, x_key="missing")


def test_covariates_keep_values_and_allow_string_categories(adata):
    data = load_screen(adata, control="control", covariates=["batch", "depth"])
    pd.testing.assert_frame_equal(data.covariates, adata.obs[["batch", "depth"]])


def test_loader_fails_on_missing_covariates(adata):
    with pytest.raises(KeyError, match="missing"):
        load_screen(adata, control="control", covariates=["missing"])


def test_all_zero_g_column_is_rejected(adata):
    G = adata.obsm["G"].copy()
    G["empty"] = 0
    adata.obsm["G"] = G
    with pytest.raises(ValueError, match="all-zero"):
        load_screen(adata)


def test_non_binary_g_is_rejected(adata):
    G = adata.obsm["G"].copy()
    G.iloc[0, 0] = 2
    adata.obsm["G"] = G
    with pytest.raises(ValueError, match="binary"):
        load_screen(adata)


def test_sparse_expression_remains_sparse_at_realistic_screen_dimensions():
    rng = np.random.default_rng(8)
    n_cells, n_genes, n_perturbations = 2_000, 5_000, 64
    expression = scipy_sparse.random(
        n_cells,
        n_genes,
        density=0.002,
        random_state=rng,
        format="csr",
    )
    assignments = np.arange(n_cells) % n_perturbations
    obs = pd.DataFrame(
        {"condition": ["control" if index % 9 == 0 else f"target_{assignments[index]}" for index in range(n_cells)]},
        index=[f"cell_{index}" for index in range(n_cells)],
    )
    adata = ad.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(n_genes)]),
    )
    adata.obsm["G"] = obs["condition"].astype("string").str.get_dummies().astype(int)

    data = load_screen(adata, control="control")

    assert isinstance(data.X, jax_sparse.JAXSparse)
    assert data.X.shape == (n_cells, n_genes)
    assert data.G.shape[0] == n_cells
    assert len(data.perturbation_names) == n_perturbations


def test_obsm_g_dataframe_round_trips_through_h5ad(tmp_path):
    rng = np.random.default_rng(3)
    cells = [f"cell_{index}" for index in range(12)]
    adata = ad.AnnData(
        X=rng.normal(size=(12, 5)),
        obs=pd.DataFrame({"condition": ["control", "a", "b"] * 4}, index=cells),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(5)]),
    )
    adata.obsm["G"] = (
        adata.obs["condition"].astype("string").str.get_dummies().astype(int)
    )
    path = tmp_path / "screen.h5ad"
    adata.write_h5ad(path)

    default = load_screen(path, control="control")
    custom = load_screen(path, g_key="G", control="control")

    assert default.perturbation_names == ("a", "b")
    assert default.perturbation_names == custom.perturbation_names
    np.testing.assert_allclose(np.asarray(default.G), np.asarray(custom.G))


def test_non_anndata_paths_are_not_implicitly_guessed(tmp_path):
    path = tmp_path / "expression.csv"
    path.write_text("cell,gene\nc1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected an AnnData"):
        load_screen(path)
