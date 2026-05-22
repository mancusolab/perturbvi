# pattern: Functional Core

import jax.numpy as jnp
import jax.scipy.special as jspec

from jax.experimental import sparse

from perturbvi.guide import SparseGuideModel
from perturbvi.sparse import SparseMatrix
from perturbvi.utils import kl_bernoulli
from tests.helpers import make_model_params


def _params_for_eta_kl():
    return make_model_params(
        x_ssq=jnp.array(0.0),
        mean_z=jnp.zeros((1, 1)),
        var_z=jnp.eye(1),
        mean_w=jnp.zeros((1, 1, 1)),
        var_w=jnp.ones((1, 1)),
        alpha=jnp.ones((1, 1, 1)),
        tau=jnp.array(1.0),
        tau_0=jnp.ones((1, 1)),
        theta=None,
        pi=jnp.ones((1, 1)),
        ann_state=None,
        mean_beta=jnp.zeros((2, 1)),
        var_beta=jnp.ones((2, 1)),
        tau_beta=jnp.ones((1,)),
        p=jnp.array([0.5, 0.25]),
        p_hat=jnp.array([[0.2, 0.8]]),
    )


def test_kl_bernoulli_includes_inclusion_and_exclusion_states():
    q = jnp.array([[0.2, 0.8]])
    p = jnp.array([0.5, 0.25])

    expected = jnp.sum(q * jnp.log(q / p) + (1.0 - q) * jnp.log((1.0 - q) / (1.0 - p)))

    assert jnp.allclose(kl_bernoulli(q, p), expected)


def test_kl_bernoulli_matches_two_state_discrete_kl():
    q = jnp.array([[0.2, 0.8], [0.35, 0.6]])
    p = jnp.array([0.5, 0.25])

    q_states = jnp.stack([q, 1.0 - q], axis=-1)
    p_states = jnp.stack([p, 1.0 - p], axis=-1)
    expected = jnp.sum(jspec.xlogy(q_states, q_states) - jspec.xlogy(q_states, p_states))

    assert jnp.allclose(kl_bernoulli(q, p), expected)


def test_kl_bernoulli_is_finite_at_boundaries():
    q = jnp.array([[0.0, 1.0]])
    p = jnp.array([0.0, 1.0])

    assert jnp.isfinite(kl_bernoulli(q, p))


def test_sparse_guide_kl_uses_full_bernoulli_kl_for_eta():
    params = _params_for_eta_kl()

    assert jnp.allclose(SparseGuideModel.kl_divergence(params), kl_bernoulli(params.p_hat, params.p))


def test_sparse_guide_weighted_sumsq_includes_cross_terms_for_overlapping_guides():
    guide_data = jnp.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    params = _params_for_eta_kl()._replace(
        mean_beta=jnp.array(
            [
                [1.5, -0.5],
                [0.25, 2.0],
            ]
        ),
        var_beta=jnp.array(
            [
                [0.4, 0.1],
                [0.2, 0.3],
            ]
        ),
        p_hat=jnp.array(
            [
                [0.8, 0.3],
                [0.6, 0.9],
            ]
        ),
    )

    mean_b = params.mean_beta * params.p_hat.T
    second_b = (params.mean_beta**2 + params.var_beta) * params.p_hat.T
    var_b = second_b - mean_b**2
    expected = jnp.sum((guide_data @ mean_b) ** 2) + jnp.sum(jnp.sum(guide_data**2, axis=0)[:, None] * var_b)

    assert jnp.allclose(SparseGuideModel(guide_data).weighted_sumsq(params), expected)


def test_sparse_guide_weighted_sumsq_matches_explicit_full_second_moment():
    guide_data = jnp.array(
        [
            [1.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.5],
        ]
    )
    params = _params_for_eta_kl()._replace(
        mean_beta=jnp.array(
            [
                [1.5, -0.5],
                [0.25, 2.0],
                [-1.0, 0.75],
            ]
        ),
        var_beta=jnp.array(
            [
                [0.4, 0.1],
                [0.2, 0.3],
                [0.5, 0.7],
            ]
        ),
        p=jnp.array([0.5, 0.25, 0.8]),
        p_hat=jnp.array(
            [
                [0.8, 0.3, 0.7],
                [0.6, 0.9, 0.4],
            ]
        ),
    )

    mean_b = params.mean_beta * params.p_hat.T
    second_b = (params.mean_beta**2 + params.var_beta) * params.p_hat.T
    var_b = second_b - mean_b**2
    gram = guide_data.T @ guide_data

    expected = 0.0
    for k in range(params.mean_beta.shape[1]):
        full_second = mean_b[:, k, None] @ mean_b[:, k, None].T + jnp.diag(var_b[:, k])
        expected = expected + jnp.trace(gram @ full_second)

    assert jnp.allclose(SparseGuideModel(guide_data).weighted_sumsq(params), expected)


def test_sparse_guide_weighted_sumsq_reduces_to_diagonal_shortcut_for_one_hot_guides():
    guide_data = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    params = _params_for_eta_kl()._replace(
        mean_beta=jnp.array(
            [
                [1.5, -0.5],
                [0.25, 2.0],
            ]
        ),
        var_beta=jnp.array(
            [
                [0.4, 0.1],
                [0.2, 0.3],
            ]
        ),
        p_hat=jnp.array(
            [
                [0.8, 0.3],
                [0.6, 0.9],
            ]
        ),
    )

    second_b = (params.mean_beta**2 + params.var_beta) * params.p_hat.T
    expected = jnp.sum(jnp.sum(guide_data**2, axis=0)[:, None] * second_b)

    assert jnp.allclose(SparseGuideModel(guide_data).weighted_sumsq(params), expected)


def test_sparse_guide_weighted_sumsq_dense_and_sparse_inputs_match():
    guide_data = jnp.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    params = _params_for_eta_kl()._replace(
        mean_beta=jnp.array(
            [
                [1.5, -0.5],
                [0.25, 2.0],
            ]
        ),
        var_beta=jnp.array(
            [
                [0.4, 0.1],
                [0.2, 0.3],
            ]
        ),
        p_hat=jnp.array(
            [
                [0.8, 0.3],
                [0.6, 0.9],
            ]
        ),
    )

    dense_value = SparseGuideModel(guide_data).weighted_sumsq(params)
    sparse_value = SparseGuideModel(SparseMatrix(sparse.BCOO.fromdense(guide_data))).weighted_sumsq(params)

    assert jnp.allclose(sparse_value, dense_value)
