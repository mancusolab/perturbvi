# pattern: Functional Core

import jax.numpy as jnp

from jax import random

from perturbvi.common import ModelParams
from perturbvi.utils import compute_lfsr


def _zero_effect_params():
    return ModelParams(
        x_ssq=jnp.array(0.0),
        mean_z=jnp.zeros((1, 1)),
        var_z=jnp.eye(1),
        mean_w=jnp.zeros((1, 1, 1)),
        var_w=jnp.zeros((1, 1)),
        alpha=jnp.ones((1, 1, 1)),
        tau=jnp.array(1.0),
        tau_0=jnp.ones((1, 1)),
        theta=None,
        pi=jnp.ones((1, 1)),
        ann_state=None,
        mean_beta=jnp.zeros((1, 1)),
        var_beta=jnp.zeros((1, 1)),
        tau_beta=jnp.ones((1,)),
        p=jnp.ones((1,)),
        p_hat=jnp.ones((1, 1)),
    )


def test_lfsr_gives_maximum_uncertainty_for_exact_zero_effects():
    lfsr = compute_lfsr(random.PRNGKey(0), _zero_effect_params(), iters=3)

    assert jnp.array_equal(lfsr, jnp.ones((1, 1)))


def test_lfsr_runs_one_jitted_scan_per_chunk(monkeypatch):
    calls = []

    def fake_lfsr_step(key, params, iters):
        calls.append(iters)
        return jnp.full((1, 1), iters), jnp.zeros((1, 1))

    monkeypatch.setattr("perturbvi.utils._compute_lfsr_step", fake_lfsr_step)

    lfsr = compute_lfsr(random.PRNGKey(0), object(), iters=205)

    assert calls == [100, 100, 5]
    assert jnp.array_equal(lfsr, jnp.zeros((1, 1)))
