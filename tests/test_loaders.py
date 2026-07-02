import anndata as ad
import numpy as np
import pandas as pd
import pytest


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

    screen = load_screen(str(h5ad_path), guide_key="perturbation")
    assert screen.X.shape[0] == 30
    assert screen.X.shape[1] == 10
    assert screen.G.shape[0] == 30
    assert screen.gene_names is not None
    assert screen.cell_names is not None
    assert screen.perturbation_names is not None


def test_load_h5ad_control_label_dropped(h5ad_path):
    from perturbvi import load_screen

    screen = load_screen(str(h5ad_path), guide_key="perturbation", control_label="non-targeting")
    assert "non-targeting" not in screen.perturbation_names


def test_load_h5ad_named_layer(h5ad_path):
    from perturbvi import load_screen

    screen = load_screen(str(h5ad_path), guide_key="perturbation", layer="counts")
    # counts layer was X * 2, so values should differ from X layer
    screen_x = load_screen(str(h5ad_path), guide_key="perturbation", layer="X")
    assert not np.allclose(screen.X, screen_x.X)


def test_load_h5ad_missing_layer(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="Layer 'bad_layer' not found"):
        load_screen(str(h5ad_path), guide_key="perturbation", layer="bad_layer")


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


def test_load_h5ad_missing_guide_obsm(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="guide_obsm 'missing' not found"):
        load_screen(str(h5ad_path), guide_obsm="missing")


def test_load_h5ad_covariates(h5ad_path):
    from perturbvi import load_screen

    screen = load_screen(str(h5ad_path), guide_key="perturbation", covariates=["batch", "n_counts"])
    assert screen.covariates is not None
    assert screen.covariates.shape == (30, 2)
    assert screen.covariate_names == ["batch", "n_counts"]


def test_load_h5ad_missing_covariate(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="covariates not found in adata.obs"):
        load_screen(str(h5ad_path), guide_key="perturbation", covariates=["does_not_exist"])


def test_load_h5ad_auto_format(h5ad_path):
    from perturbvi import load_screen

    # auto format should detect .h5ad extension
    screen = load_screen(str(h5ad_path), guide_key="perturbation", format="auto")
    assert screen.X.shape[1] == 10


def test_load_h5ad_covariate_file_raises(h5ad_path):
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="covariate_file= is not supported for h5ad"):
        load_screen(str(h5ad_path), guide_key="perturbation", covariate_file="some_file.csv")
