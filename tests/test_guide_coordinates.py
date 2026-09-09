import numpy as np
import pytest

from scipy.special import expit, logit

import equinox as eqx
import jax.numpy as jnp

from jax import random
from jax.experimental import sparse

from perturbvi.annotation import FixedPrior
from perturbvi.factorloadings import FactorModel
from perturbvi.guide import SparseGuideModel
from perturbvi.infer import _init_params
from perturbvi.sparse import SparseMatrix


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_coordinate_sweeps_match_current_residual_reference_and_improve_objective(sparse_input, overlap):
    G = np.vstack([np.ones((9, 3)), np.eye(3)]) if overlap else np.tile(np.eye(3), (4, 1))
    data = jnp.asarray(G)
    if sparse_input:
        data = SparseMatrix(sparse.BCOO.fromdense(data))
    guide = SparseGuideModel(data)
    factors = FactorModel()
    params = _init_params(random.PRNGKey(0), 2, 1, jnp.ones((12, 2)), guide, FixedPrior(), init="random")
    params = params._replace(mean_z=jnp.asarray(G @ np.array([[2., -1.], [1., 3.], [2., 1.]])))
    update = eqx.filter_jit(guide.update)
    previous = -float(factors.kl_divergence(guide, params) + guide.kl_divergence(params))
    for _ in range(10):
        mu, var, q = map(lambda x: np.array(x, copy=True), (params.mean_beta, params.var_beta, params.p_hat))
        for g in range(G.shape[1]):
            residual = np.asarray(params.mean_z) - G @ (mu * q.T) + G[:, g, None] * mu[g] * q[:, g]
            var[g] = 1 / (np.asarray(params.tau_beta) + G[:, g] @ G[:, g])
            mu[g] = (G[:, g] @ residual) * var[g]
            bf = .5 * (np.log(var[g] * params.tau_beta) + mu[g]**2 / var[g])
            q[:, g] = np.clip(expit(logit(float(params.p[g])) + bf), 1e-8, 1 - 1e-8)
        params = update(params)
        np.testing.assert_allclose(params.mean_beta, mu, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(params.var_beta, var, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(params.p_hat, q, atol=2e-5, rtol=2e-5)
        objective = -float(factors.kl_divergence(guide, params) + guide.kl_divergence(params))
        assert objective >= previous - 1e-4
        previous = objective
