"""Retained numerical and precision checks for the current result interfaces."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import jax

from perturbvi import estimate_lfsr, FitResults, infer, save_results
from perturbvi._results import _posterior


@pytest.fixture
def fitted(tmp_path):
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_perturbations = 80, 30, 6
    X = rng.normal(size=(n_cells, n_genes))
    G = np.zeros((n_cells, n_perturbations))
    G[np.arange(n_cells), np.arange(n_cells) % n_perturbations] = 1
    inference = infer(X, G, z_dim=3, l_dim=5, tau=20, max_iter=2, seed=0, verbose=False)
    fit = FitResults(
        inference=inference,
        gene_names=tuple(f"gene_{index}" for index in range(n_genes)),
        perturbation_names=tuple(f"target_{index}" for index in range(n_perturbations)),
    )
    output = tmp_path / "fit"
    save_results(fit, output)
    return fit, output


def test_weighting_and_total_effect_are_applied_once():
    W = np.array([[2., -3.], [4., 5.]])
    beta = np.array([[3., 2.], [-1., 4.]])
    probability = np.array([[.5, .25], [.75, .5]])
    inference = SimpleNamespace(
        params=SimpleNamespace(W=W, mean_w=W[None], mean_beta=beta, p_hat=probability.T),
    )
    fit = FitResults(inference, ("001", "NA"), ("drug", "guide"))
    expected_B = np.array([[1.5, .5], [-.75, 2.]])
    np.testing.assert_array_equal(fit.W, W)
    np.testing.assert_array_equal(fit.B, expected_B)
    np.testing.assert_array_equal(fit.BW, expected_B @ W)


def test_saved_fit_preserves_precision_and_lfsr_reproducibility(fitted):
    fit, output = fitted
    params, genes, perturbations = _posterior(output)
    left = estimate_lfsr(output, draws=12, seed=19)
    right = estimate_lfsr(output, draws=12, seed=19)
    assert jax.config.jax_enable_x64
    assert params.mean_w.dtype == fit.params.mean_w.dtype
    np.testing.assert_array_equal(params.mean_w, fit.params.mean_w)
    assert genes == fit.gene_names
    assert perturbations == fit.perturbation_names
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert np.isfinite(left).all().all()
    assert left.min().min() >= 0 and left.max().max() <= 1


def test_float64_saved_fit_round_trip(tmp_path):
    rng = np.random.default_rng(94)
    output = tmp_path / "float64"
    inference = infer(rng.normal(size=(24, 7)), np.eye(2)[np.arange(24) % 2],
                      z_dim=2, l_dim=3, tau=10, max_iter=2, verbose=False)
    assert inference.params.mean_w.dtype == np.dtype("float64")
    fit = FitResults(inference, tuple(f"g{i}" for i in range(7)), ("A", "B"))
    expected = estimate_lfsr(fit, draws=11, seed=7)
    save_results(fit, output)
    saved = {
        path.name: pd.read_csv(path, index_col=0, float_precision="round_trip")
        for path in output.glob("*.csv")
    }
    actual = estimate_lfsr(output, draws=11, seed=7)
    save_results(output)
    assert jax.config.jax_enable_x64
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    for name, table in saved.items():
        regenerated = pd.read_csv(output / name, index_col=0, float_precision="round_trip")
        pd.testing.assert_frame_equal(table, regenerated, check_exact=True)
