from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scipy import sparse

from perturbvi._defaults import DEFAULT_BLOCK_SIZE
from perturbvi.preprocess import (
    residualize_screen,
)
from perturbvi.screen import PerturbData
from tests.helpers import build_design_matrix


def _make_screen(
    n=50,
    g=10,
    p=3,
    seed=0,
    covariates=None,
):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, g))
    G = (rng.random((n, p)) > 0.5).astype(float)
    # ensure no empty guide columns
    for col in range(p):
        if G[:, col].sum() == 0:
            G[0, col] = 1.0
    return PerturbData(
        X=X,
        G=G,
        gene_names=[f"gene_{index}" for index in range(g)],
        perturbation_names=[f"pert_{index}" for index in range(p)],
        covariates=covariates,
    )


def test_residualize_numeric_orthogonal():
    rng = np.random.default_rng(0)
    n, g = 100, 20
    depth = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, g)) + depth.reshape(-1, 1) * 2

    screen = _make_screen(n=n, g=g, covariates=pd.DataFrame({"depth": depth}))
    screen = replace(screen, X=X)

    resid = residualize_screen(screen)
    # residuals should be orthogonal to depth
    assert np.abs(depth @ np.asarray(resid.X)).max() < 1e-4


def test_residualize_categorical_orthogonal():
    rng = np.random.default_rng(1)
    n, g = 60, 10
    batch = rng.choice([0, 1, 2], n).astype(float)
    X = rng.normal(0, 1, (n, g)) + batch.reshape(-1, 1)

    frame = pd.DataFrame({"batch": pd.Categorical(batch)})
    screen = _make_screen(n=n, g=g, covariates=frame)
    screen = replace(screen, X=X)

    resid = residualize_screen(screen)
    # residuals should be orthogonal to all dummy columns of batch
    C = build_design_matrix(frame)
    for col in range(C.shape[1]):
        assert np.abs(C[:, col] @ np.asarray(resid.X)).max() < 1e-4


def test_residualize_rank_deficient_no_crash():
    rng = np.random.default_rng(2)
    n, g = 40, 8
    c = rng.normal(0, 1, n)
    # Two identical columns produce a rank-deficient design matrix.
    covariates = pd.DataFrame({"c1": c, "c2": c})
    screen = _make_screen(n=n, g=g, covariates=covariates)
    resid = residualize_screen(screen)
    assert resid.X.shape == (n, g)
    design = build_design_matrix(covariates)
    expected = screen.X - design @ np.linalg.lstsq(design, screen.X, rcond=None)[0]
    np.testing.assert_allclose(resid.X, expected, atol=1e-5)


def test_residualize_clears_covariates():
    rng = np.random.default_rng(3)
    n = 30
    values = rng.normal(0, 1, (n, 2))
    covariates = pd.DataFrame(values, columns=["a", "b"])
    screen = _make_screen(n=n, covariates=covariates)
    resid = residualize_screen(screen)
    assert resid.covariates is None


def test_residualize_does_not_mutate_input():
    rng = np.random.default_rng(4)
    n = 30
    covariates = pd.DataFrame({"depth": rng.normal(0, 1, n)})
    screen = _make_screen(n=n, covariates=covariates)
    resid = residualize_screen(screen)
    assert screen.covariates is covariates
    assert resid.covariates is None


def test_residualize_no_covariates_raises():
    screen = _make_screen()
    with pytest.raises(ValueError, match="screen.covariates is None"):
        residualize_screen(screen)


def test_residualize_infers_plain_text_as_categorical():
    screen = _make_screen(
        n=20,
        covariates=pd.DataFrame({"batch": ["A", "B"] * 10}),
    )
    residualized = residualize_screen(screen)
    design = build_design_matrix(screen.covariates)
    np.testing.assert_allclose(design.T @ residualized.X, 0, atol=1e-4)


def test_build_design_matrix_intercept():
    covariates = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    C = build_design_matrix(covariates)
    # first column should be all ones (intercept)
    assert np.all(C[:, 0] == 1.0)
    assert C.shape == (3, 2)


def test_build_design_matrix_categorical_drops_reference():
    # Three categories produce two dummy columns after dropping the reference.
    cats = pd.Categorical([0, 1, 2, 0, 1, 2])
    C = build_design_matrix(pd.DataFrame({"cat": cats}))
    # intercept + 2 dummy columns (not 3)
    assert C.shape == (6, 3)


def test_constant_covariate_is_a_harmless_intercept_only_design():
    screen = _make_screen(
        n=20,
        covariates=pd.DataFrame({"constant": np.ones(20)}),
    )
    residualized = residualize_screen(screen)
    np.testing.assert_allclose(residualized.X.mean(axis=0), 0, atol=1e-6)


def test_residualize_uses_stored_numeric_and_categorical_schema():
    rng = np.random.default_rng(8)
    n = 30
    depth = rng.normal(size=n)
    batch = rng.choice(["A", "B"], size=n)
    screen = _make_screen(
        n=n,
        covariates=pd.DataFrame({"depth": depth, "batch": pd.Categorical(batch)}),
    )

    residualized = residualize_screen(screen)

    assert residualized.X.shape == screen.X.shape
    assert np.isfinite(residualized.X).all()
    assert residualized.covariates is None


def test_residualize_sparse_matches_ols():
    rng = np.random.default_rng(5)
    n, g = 80, 15
    depth = rng.normal(size=n)
    batch = pd.Categorical(rng.choice(["A", "B"], n))
    covariates = pd.DataFrame({"depth": depth, "batch": batch})
    X = sparse.random(n, g, density=0.3, format="csr", random_state=rng)

    screen = _make_screen(n=n, g=g, covariates=covariates)
    screen = replace(screen, X=X)
    residualized = residualize_screen(screen)

    assert isinstance(residualized.X, np.ndarray)
    assert np.isfinite(residualized.X).all()
    design = build_design_matrix(covariates)
    expected = X.toarray() - design @ np.linalg.lstsq(design, X.toarray(), rcond=None)[0]
    np.testing.assert_allclose(residualized.X, expected, atol=1e-5)


def test_residualize_memmap_matches_in_memory(tmp_path):
    rng = np.random.default_rng(6)
    n, g = 40, 10
    covariates = pd.DataFrame({"depth": rng.normal(size=n)})
    screen = _make_screen(n=n, g=g, covariates=covariates)

    out = tmp_path / "residualized.dat"
    memmap_result = residualize_screen(screen, output=out)
    in_memory = residualize_screen(screen)

    assert isinstance(memmap_result.X, np.memmap)
    assert memmap_result.X.dtype == np.float32
    assert out.exists()
    np.testing.assert_allclose(np.asarray(memmap_result.X), in_memory.X, atol=1e-6)


def test_residualize_block_size_invariance():
    rng = np.random.default_rng(7)
    n, g = 60, 30
    covariates = pd.DataFrame(
        {
            "depth": rng.normal(size=n),
            "batch": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    screen = _make_screen(n=n, g=g, covariates=covariates)
    reference = residualize_screen(screen)

    for block_size in (1, 8, 64, 10_000):
        got = residualize_screen(screen, block_size=block_size)
        np.testing.assert_allclose(got.X, reference.X, atol=1e-5)


def test_residualize_is_idempotent():
    rng = np.random.default_rng(8)
    n, g = 50, 12
    covariates = pd.DataFrame({"depth": rng.normal(size=n)})
    screen = _make_screen(n=n, g=g, covariates=covariates)

    once = residualize_screen(screen)
    again = _make_screen(n=n, g=g, covariates=covariates)
    again = replace(again, X=np.asarray(once.X))
    twice = residualize_screen(again)

    np.testing.assert_allclose(twice.X, once.X, atol=1e-5)


def test_residualize_float32_output_precision_budget():
    rng = np.random.default_rng(9)
    n, g = 80, 20
    depth = rng.normal(size=n)
    covariates = pd.DataFrame({"depth": depth})
    X = rng.normal(size=(n, g))
    screen = _make_screen(n=n, g=g, covariates=covariates)
    screen = replace(screen, X=X)

    residualized = residualize_screen(screen)
    assert residualized.X.dtype == np.float32

    design = build_design_matrix(covariates)
    expected = X - design @ np.linalg.lstsq(design, X, rcond=None)[0]
    scale = max(1.0, np.max(np.abs(expected)))
    np.testing.assert_allclose(residualized.X, expected, atol=1e-5 * scale)


def test_default_block_size_wired_from_defaults():
    import inspect

    assert isinstance(DEFAULT_BLOCK_SIZE, int)
    assert DEFAULT_BLOCK_SIZE > 0
    signature_default = inspect.signature(residualize_screen).parameters["block_size"].default
    assert signature_default == DEFAULT_BLOCK_SIZE


def test_residualize_invalid_block_size_raises():
    screen = _make_screen(covariates=pd.DataFrame({"depth": np.ones(50)}))
    with pytest.raises(ValueError, match="block_size must be positive"):
        residualize_screen(screen, block_size=0)
