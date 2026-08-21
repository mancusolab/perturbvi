import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scipy import sparse as scipy_sparse

from jax.experimental import sparse as jax_sparse


@pytest.fixture
def h5ad_path(tmp_path):
    rng = np.random.default_rng(0)
    n, g = 30, 10
    X = rng.poisson(5, (n, g)).astype(float)
    obs = pd.DataFrame(
        {
            "perturbation": rng.choice(["geneA", "geneB", "non-targeting"], n),
            "batch": rng.choice(["A", "B"], n),
            "n_counts": X.sum(axis=1),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(g)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X * 2
    path = tmp_path / "screen.h5ad"
    adata.write_h5ad(path)
    return path


def test_load_h5ad_guide_key(h5ad_path):
    from perturbvi import load_screen

    screen = load_screen(str(h5ad_path), guide_key="perturbation", control_label="non-targeting")
    assert screen.X.shape[0] == 30
    assert screen.X.shape[1] == 10
    assert screen.G.shape[0] == 30
    assert screen.gene_names is not None
    assert screen.cell_names is not None
    assert screen.perturbation_names is not None


def test_load_anndata_object(h5ad_path):
    from perturbvi import load_screen

    adata = ad.read_h5ad(h5ad_path)
    screen = load_screen(
        adata,
        format="anndata",
        guide_key="perturbation",
        control_label="non-targeting",
        covariates=["batch", "n_counts"],
    )
    assert screen.X.shape == adata.shape
    assert screen.cell_names == adata.obs_names.astype(str).tolist()
    assert screen.source["format"] == "anndata"
    assert screen.source["path"] is None


def test_load_anndata_object_rejects_file_format(h5ad_path):
    from perturbvi import load_screen

    adata = ad.read_h5ad(h5ad_path)
    with pytest.raises(ValueError, match="incompatible"):
        load_screen(adata, format="csv", guide_key="perturbation")


def test_load_h5ad_control_label_dropped(h5ad_path):
    from perturbvi import load_screen

    screen = load_screen(str(h5ad_path), guide_key="perturbation", control_label="non-targeting")
    assert "non-targeting" not in screen.perturbation_names


def test_load_h5ad_requires_control_label_for_categorical_assignments(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="control_label is required"):
        load_screen(str(h5ad_path), guide_key="perturbation")


def test_load_h5ad_named_layer(h5ad_path):
    from perturbvi import load_screen

    screen = load_screen(
        str(h5ad_path), guide_key="perturbation", control_label="non-targeting", layer="counts"
    )
    # counts layer was X * 2, so values should differ from X layer
    screen_x = load_screen(str(h5ad_path), guide_key="perturbation", control_label="non-targeting", layer="X")
    assert not np.allclose(screen.X, screen_x.X)


def test_load_h5ad_missing_layer(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="Layer 'bad_layer' not found"):
        load_screen(
            str(h5ad_path), guide_key="perturbation", control_label="non-targeting", layer="bad_layer"
        )


def test_load_h5ad_missing_guide_key(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="guide_key 'does_not_exist' not found"):
        load_screen(str(h5ad_path), guide_key="does_not_exist")


def test_load_h5ad_guide_obsm(h5ad_path):
    import scanpy as sc

    from perturbvi import load_screen

    adata = sc.read_h5ad(h5ad_path)
    rng = np.random.default_rng(1)
    G = (rng.random((adata.n_obs, 3)) > 0.5).astype(float)
    adata.obsm["guide_matrix"] = G
    adata.write_h5ad(h5ad_path)

    screen = load_screen(str(h5ad_path), guide_obsm="guide_matrix")
    assert screen.G.shape == (adata.n_obs, 3)


def test_load_h5ad_rejects_mismatched_guide_obsm(h5ad_path, monkeypatch):
    import scanpy as sc

    from perturbvi import load_screen

    adata = sc.read_h5ad(h5ad_path)

    class FakeAnnData:
        X = adata.X
        obs = adata.obs
        var_names = adata.var_names
        obs_names = adata.obs_names
        layers = adata.layers
        obsm = {"bad_guides": np.ones((adata.n_obs - 1, 2))}
        n_obs = adata.n_obs
        n_vars = adata.n_vars

    monkeypatch.setattr(sc, "read_h5ad", lambda path: FakeAnnData())
    with pytest.raises(ValueError, match="same number of rows"):
        load_screen(str(h5ad_path), guide_obsm="bad_guides")


def test_load_h5ad_missing_guide_obsm(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="guide_obsm 'missing' not found"):
        load_screen(str(h5ad_path), guide_obsm="missing")


def test_load_h5ad_covariates(h5ad_path):
    from perturbvi import load_screen

    screen = load_screen(
        str(h5ad_path),
        guide_key="perturbation",
        control_label="non-targeting",
        covariates=["batch", "n_counts"],
    )
    assert screen.covariates is not None
    assert screen.covariates.shape == (30, 2)
    assert screen.covariate_names == ["batch", "n_counts"]


def test_load_h5ad_missing_covariate(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="covariates not found in adata.obs"):
        load_screen(
            str(h5ad_path),
            guide_key="perturbation",
            control_label="non-targeting",
            covariates=["does_not_exist"],
        )


def test_load_h5ad_auto_format(h5ad_path):
    from perturbvi import load_screen

    # auto format should detect .h5ad extension
    screen = load_screen(
        str(h5ad_path), guide_key="perturbation", control_label="non-targeting", format="auto"
    )
    assert screen.X.shape[1] == 10


def test_load_h5ad_metadata_path_raises(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="metadata_path is not supported for h5ad"):
        load_screen(
            str(h5ad_path),
            guide_key="perturbation",
            control_label="non-targeting",
            metadata_path="some_file.csv",
        )


def test_load_h5ad_rejects_missing_guide_labels(tmp_path):
    from perturbvi import load_screen

    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame({"guide": ["g1", None, "g2"]}, index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["x", "y"]),
    )
    path = tmp_path / "missing.h5ad"
    adata.write_h5ad(path)

    with pytest.raises(ValueError, match="Perturbation labels are missing"):
        load_screen(str(path), guide_key="guide", control_label="g1")

    screen = load_screen(str(path), guide_key="guide", control_label="g1", missing_guide="unassigned")
    np.testing.assert_array_equal(np.asarray(screen.G)[1], np.zeros(1))


def test_load_h5ad_rejects_missing_categorical_covariate(tmp_path):
    from perturbvi import load_screen

    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {"guide": ["g1", "g1", "g2"], "batch": ["A", None, "B"]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["x", "y"]),
    )
    path = tmp_path / "missing_covariate.h5ad"
    adata.write_h5ad(path)

    with pytest.raises(ValueError, match="missing values"):
        load_screen(str(path), guide_key="guide", control_label="g1", covariates=["batch"])


def test_load_h5ad_rejects_unknown_control_label(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="control_label"):
        load_screen(str(h5ad_path), guide_key="perturbation", control_label="typo")


def test_load_h5ad_preserves_sparse_expression(tmp_path):
    from perturbvi import load_screen

    adata = ad.AnnData(
        X=scipy_sparse.csr_matrix(np.eye(4)),
        obs=pd.DataFrame({"guide": ["g1", "g1", "g2", "g2"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"x{i}" for i in range(4)]),
    )
    path = tmp_path / "sparse.h5ad"
    adata.write_h5ad(path)
    screen = load_screen(str(path), guide_key="guide", control_label="g1")
    assert isinstance(screen.X, jax_sparse.JAXSparse)
