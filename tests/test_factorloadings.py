# pattern: Functional Core

import jax.numpy as jnp

from perturbvi.common import ModelParams
from perturbvi.factorloadings import FactorModel


class _Guide:
    def predict(self, params):
        return jnp.zeros_like(params.mean_z)


class _Loadings:
    def moments(self, params):
        mean_w = jnp.array([[1.0, 0.5], [0.25, 1.5]])
        mean_ww = mean_w @ mean_w.T + jnp.eye(2) * 0.1
        return mean_w, mean_ww


def _params():
    return ModelParams(
        x_ssq=jnp.array(1.0),
        mean_z=jnp.zeros((3, 2)),
        var_z=jnp.eye(2),
        mean_w=jnp.zeros((1, 2, 2)),
        var_w=jnp.ones((1, 2)),
        alpha=jnp.ones((1, 2, 2)) / 2,
        tau=jnp.array(2.0),
        tau_0=jnp.ones((1, 2)),
        theta=None,
        pi=jnp.ones((2, 2)) / 2,
        ann_state=None,
        mean_beta=jnp.zeros((2, 2)),
        var_beta=jnp.ones((2, 2)),
        tau_beta=jnp.ones((2,)),
        p=jnp.ones((2,)) / 2,
        p_hat=jnp.ones((2, 2)) / 2,
    )


def test_factor_update_does_not_call_explicit_inverse(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("explicit inverse should not be used for factor covariance update")

    monkeypatch.setattr("perturbvi.factorloadings.jnp.linalg.inv", fail_if_called)

    updated = FactorModel().update(
        data=jnp.array(
            [
                [1.0, -1.0],
                [0.5, 2.0],
                [-1.5, 0.25],
            ]
        ),
        guide=_Guide(),
        loadings=_Loadings(),
        params=_params(),
    )

    assert updated.mean_z.shape == (3, 2)
    assert updated.var_z.shape == (2, 2)
    assert jnp.all(jnp.isfinite(updated.mean_z))
    assert jnp.all(jnp.isfinite(updated.var_z))
