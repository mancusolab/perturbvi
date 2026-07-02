import numpy as np
import pytest

from perturbvi.preprocess import _build_design_matrix, _qr_residualize, residualize_screen
from perturbvi.screen import ScreenData


def _make_screen(n=50, g=10, p=3, seed=0, covariates=None, covariate_names=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, g))
    G = (rng.random((n, p)) > 0.5).astype(float)
    # ensure no empty guide columns
    for col in range(p):
        if G[:, col].sum() == 0:
            G[0, col] = 1.0
    return ScreenData(
        X=X,
        G=G,
        gene_names=None,
        perturbation_names=None,
        cell_names=None,
        source={},
        covariates=covariates,
        covariate_names=covariate_names,
    )


def test_residualize_numeric_orthogonal():
    rng = np.random.default_rng(0)
    n, g = 100, 20
    depth = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, g)) + depth.reshape(-1, 1) * 2

    screen = _make_screen(n=n, g=g, covariates=depth.reshape(-1, 1), covariate_names=["depth"])
    screen = screen._replace(X=X)

    resid = residualize_screen(screen)
    # residuals should be orthogonal to depth
    assert np.abs(depth @ np.asarray(resid.X)).max() < 1e-8


def test_residualize_categorical_orthogonal():
    rng = np.random.default_rng(1)
    n, g = 60, 10
    batch = rng.choice([0, 1, 2], n).astype(float)
    X = rng.normal(0, 1, (n, g)) + batch.reshape(-1, 1)

    screen = _make_screen(n=n, g=g, covariates=batch.reshape(-1, 1), covariate_names=["batch"])
    screen = screen._replace(X=X)

    resid = residualize_screen(screen, categoricals=["batch"])
    # residuals should be orthogonal to all dummy columns of batch
    C = _build_design_matrix(batch.reshape(-1, 1), ["batch"], ["batch"])
    for col in range(C.shape[1]):
        assert np.abs(C[:, col] @ np.asarray(resid.X)).max() < 1e-8


def test_residualize_rank_deficient_no_crash():
    rng = np.random.default_rng(2)
    n, g = 40, 8
    c = rng.normal(0, 1, n)
    # two identical columns → rank deficient design matrix
    covariates = np.column_stack([c, c])
    screen = _make_screen(n=n, g=g, covariates=covariates, covariate_names=["c1", "c2"])
    resid = residualize_screen(screen)
    assert resid.X.shape == (n, g)


def test_residualize_clears_covariates():
    rng = np.random.default_rng(3)
    n = 30
    covariates = rng.normal(0, 1, (n, 2))
    screen = _make_screen(n=n, covariates=covariates, covariate_names=["a", "b"])
    resid = residualize_screen(screen)
    assert resid.covariates is None
    assert resid.covariate_names is None


def test_residualize_records_metadata():
    rng = np.random.default_rng(4)
    n = 30
    covariates = rng.normal(0, 1, (n, 1))
    screen = _make_screen(n=n, covariates=covariates, covariate_names=["depth"])
    resid = residualize_screen(screen, categoricals=[])
    assert resid.source["residualized"] is True
    assert resid.source["cov_names"] == ["depth"]
    assert resid.source["categoricals"] == []


def test_residualize_no_covariates_raises():
    screen = _make_screen()
    with pytest.raises(ValueError, match="screen.covariates is None"):
        residualize_screen(screen)


def test_residualize_unknown_categorical_raises():
    rng = np.random.default_rng(5)
    n = 20
    covariates = rng.normal(0, 1, (n, 1))
    screen = _make_screen(n=n, covariates=covariates, covariate_names=["depth"])
    with pytest.raises(ValueError, match="not found in covariate_names"):
        residualize_screen(screen, categoricals=["batch"])


def test_qr_residualize_shape():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (50, 15))
    C = np.column_stack([np.ones(50), rng.normal(0, 1, 50)])
    R = _qr_residualize(X, C)
    assert R.shape == X.shape


def test_build_design_matrix_intercept():
    covariates = np.array([[1.0], [2.0], [3.0]])
    C = _build_design_matrix(covariates, ["x"], categoricals=None)
    # first column should be all ones (intercept)
    assert np.all(C[:, 0] == 1.0)
    assert C.shape == (3, 2)


def test_build_design_matrix_categorical_drops_reference():
    # 3 categories → 2 dummy columns (reference dropped)
    cats = np.array([0, 1, 2, 0, 1, 2], dtype=float)
    C = _build_design_matrix(cats.reshape(-1, 1), ["cat"], categoricals=["cat"])
    # intercept + 2 dummy columns (not 3)
    assert C.shape == (6, 3)
