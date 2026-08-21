import numpy as np

from perturbvi import generate_sim
from perturbvi.sim import generate_sim_with_control


def test_generate_sim_uses_distinct_reproducible_loading_draws():
    kwargs = dict(seed=7, l_dim=2, n_dim=8, p_dim=6, z_dim=3, g_dim=3)

    first = generate_sim(**kwargs)
    repeated = generate_sim(**kwargs)
    loading_blocks = np.stack(
        [np.asarray(first.W)[factor, factor * kwargs["l_dim"] : (factor + 1) * kwargs["l_dim"]]
         for factor in range(kwargs["z_dim"])]
    )

    np.testing.assert_array_equal(first.W, repeated.W)
    assert np.unique(loading_blocks, axis=0).shape[0] == kwargs["z_dim"]


def test_generate_sim_with_control_preserves_guide_effect_alignment():
    n_dim = 10
    g_dim = 3
    control_fraction = 0.3
    result = generate_sim_with_control(
        seed=7,
        l_dim=1,
        n_dim=n_dim,
        p_dim=6,
        z_dim=2,
        g_dim=g_dim,
        control_fraction=control_fraction,
        b_sparsity=1.0,
    )

    control_size = int(n_dim * control_fraction)
    case_size = n_dim - control_size

    assert result.X.shape == (n_dim, 6)
    assert result.Z.shape == (n_dim, 2)
    assert result.G.shape == (n_dim, g_dim)
    assert result.beta.shape == (g_dim, 2)
    np.testing.assert_array_equal(result.G[:g_dim], np.eye(g_dim))
    np.testing.assert_allclose(result.G[:g_dim] @ result.beta, result.beta)
    np.testing.assert_array_equal(result.G[case_size:], np.zeros((control_size, g_dim)))
