import importlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax.experimental import sparse

from perturbvi.annotation import FixedPrior
from perturbvi.guide import SparseGuideModel
from perturbvi.infer import _init_params
from perturbvi.sparse import CenteredSparseMatrix
from perturbvi.utils import prob_pca


@pytest.mark.parametrize("sparse_input", [False, True])
def test_pca_returns_ordered_scaled_axes_and_consistent_reconstruction(sparse_input):
    rng = np.random.default_rng(72)
    X = rng.normal(size=(35, 3)) @ rng.normal(size=(3, 12))
    X -= X.mean(axis=0)
    data = CenteredSparseMatrix(sparse.BCOO.fromdense(jnp.asarray(X))) if sparse_input else jnp.asarray(X)
    Z, W = map(np.asarray, prob_pca(jax.random.PRNGKey(0), data, 3))
    np.testing.assert_allclose(Z.T @ Z, len(X) * np.eye(3), atol=1e-4)
    np.testing.assert_allclose(Z @ W, X, atol=1e-4)
    np.testing.assert_allclose(W @ W.T, np.diag(np.sum(W**2, axis=1)), atol=1e-4)
    assert np.all(np.diff(np.sum(W**2, axis=1)) <= 0)


def test_pca_rank_deficient_input_remains_finite():
    X = jnp.arange(8.0)[:, None] * jnp.ones((1, 5))
    X -= X.mean(axis=0)
    Z, W = prob_pca(jax.random.PRNGKey(2), X, 4, max_iter=20)
    assert np.isfinite(Z).all() and np.isfinite(W).all()
    np.testing.assert_allclose(Z @ W, X, atol=1e-4)


def test_pca_recovers_leading_components_in_full_rank_data():
    rng = np.random.default_rng(14)
    left = rng.normal(size=(40, 8))
    left -= left.mean(axis=0)
    left, _ = np.linalg.qr(left)
    right, _ = np.linalg.qr(rng.normal(size=(12, 8)))
    spectrum = np.array([9., 7., 5., 2., 1., .8, .5, .2])
    X = (left * spectrum) @ right.T
    expected = (left[:, :3] * spectrum[:3]) @ right[:, :3].T
    Z, W = prob_pca(jax.random.PRNGKey(5), jnp.asarray(X), 3, tol=1e-4)
    np.testing.assert_allclose(Z @ W, expected, atol=1e-4)


def test_initial_variances_do_not_introduce_random_precisions():
    X = jnp.asarray(np.random.default_rng(0).normal(size=(30, 8)))
    guide = SparseGuideModel(jnp.tile(jnp.eye(3), (10, 1)))
    for init in ("pca", "random"):
        for seed in (0, 1):
            params = _init_params(jax.random.PRNGKey(seed), 3, 4, X, guide, FixedPrior(), init=init)
            np.testing.assert_array_equal(params.var_w, 1 / params.tau_0)
            np.testing.assert_array_equal(params.var_beta, np.ones((3, 3)))
            assert np.linalg.eigvalsh(params.var_z).min() > 0
            if init == "pca":
                np.testing.assert_allclose(np.sum(params.mean_z**2, axis=0), 30, atol=1e-4)


@pytest.mark.parametrize("bad_elbo", [float("nan"), float("inf"), -float("inf")])
def test_infer_rejects_nonfinite_objective(monkeypatch, bad_elbo):
    module = importlib.import_module("perturbvi.infer")
    from perturbvi.common import ELBOResults

    def bad_step(X, guide, factors, loadings, annotation, params):
        return ELBOResults(bad_elbo, 0., 0., 0., 0.), params

    monkeypatch.setattr(module, "_inner_loop", bad_step)
    with pytest.raises(FloatingPointError, match="Nonfinite ELBO at iteration 1"):
        module.infer(jnp.eye(4), jnp.ones((4, 1)), z_dim=1, l_dim=1, max_iter=1, init="random")


def test_small_objective_decrease_does_not_count_as_convergence(monkeypatch):
    module = importlib.import_module("perturbvi.infer")
    from perturbvi.common import ELBOResults

    scores = iter([1., 1. - 1e-4, 2., 2.])
    observed = []

    def step(X, guide, factors, loadings, annotation, params):
        score = next(scores)
        observed.append(score)
        return ELBOResults(score, 0., 0., 0., 0.), params

    monkeypatch.setattr(module, "_inner_loop", step)
    module.infer(jnp.eye(4), jnp.ones((4, 1)), z_dim=1, l_dim=1, max_iter=4, init="random")
    assert len(observed) == 4
