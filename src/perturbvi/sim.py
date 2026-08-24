from typing import NamedTuple, Optional

import jax.numpy as jnp

from jax import random
from jax.typing import ArrayLike


__all__ = [
    "SimulatedData",
    "generate_sim",
]


class SimulatedData(NamedTuple):
    """the object contain simulated data components.

    Args:
        Z: simulated factor
        W: simulated loadings
        X: simulated data set
        G: design matrix
        beta: perturbation effect matrix

    """

    Z: ArrayLike
    W: ArrayLike
    X: ArrayLike
    G: Optional[ArrayLike]
    beta: Optional[ArrayLike]


# Define the function to generate design matrix.
# Ensure each perturbation is at least assigned to a cell
def create_design_matrix(key, n_dim, g_dim):
    """Create a design matrix given N and G

    Args:
    key : JAX PRNG key used to assign the remaining cells.
    n_dim : Sample size
    g_dim : Perturbation dimension

    Returns:
    G : The design matrix.
    """
    if g_dim > n_dim:
        raise ValueError(f"g_dim should be less than n: received g_dim = {g_dim}, n = {n_dim}")
    # Initialize the matrix with zeros
    # Assign each perturbation exactly once
    G = jnp.identity(g_dim)

    # Randomly assign remaining perturbations using the caller's JAX key
    indices = random.choice(key, g_dim, shape=(n_dim - g_dim,), replace=True)
    G_left = jnp.zeros((n_dim - g_dim, g_dim), dtype=int)
    G_left = G_left.at[jnp.arange(n_dim - g_dim), indices].set(1)

    G = jnp.concatenate((G, G_left), axis=0)
    return G


def _validate_sim_inputs(seed, l_dim, n_dim, p_dim, z_dim, g_dim, b_sparsity, effect_size):
    """Reject invalid simulation arguments before any arithmetic or allocation."""
    if isinstance(seed, int) is False:
        raise ValueError(f"seed should be an interger: received seed = {seed}")

    for name, value in (("n_dim", n_dim), ("p_dim", p_dim), ("z_dim", z_dim), ("g_dim", g_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} should be a positive integer: received {name} = {value}")

    if isinstance(l_dim, bool) or not isinstance(l_dim, int) or l_dim <= 0:
        raise ValueError(f"l_dim should be positive: received l_dim = {l_dim}")

    # Integer form of the old "l_dim < p_dim / z_dim" rule; safe for z_dim = 0
    # because positivity is already enforced above.
    if l_dim * z_dim > p_dim:
        raise ValueError(
            f"l_dim should be less than p_dim/z_dim: received l_dim = {l_dim}, "
            f"z_dim = {z_dim}, p_dim = {p_dim}"
        )

    if g_dim > n_dim:
        raise ValueError(f"g_dim should be less than n: received g_dim = {g_dim}, n = {n_dim}")

    if not 0 <= b_sparsity <= 1:
        raise ValueError(f"b_sparsity should be between 0 and 1: received b_sparsity = {b_sparsity}")

    if effect_size <= 0:
        raise ValueError(f"effect size should be positive: received effect_size = {effect_size}")


def _random_loadings(key, z_dim, p_dim, l_dim, effect_size):
    """Build the block-sparse loading matrix W from the given key."""
    W = jnp.zeros(shape=(z_dim, p_dim))
    loading_values = effect_size * random.normal(key, shape=(z_dim, l_dim))

    for k in range(z_dim):
        W = W.at[k, (k * l_dim) : ((k + 1) * l_dim)].set(loading_values[k])

    return W


def _random_beta(key, z_dim, g_dim, non_zero_num):
    """Build the sparse perturbation effect matrix beta from the given key."""
    beta = jnp.zeros(shape=(g_dim, z_dim))

    for col in range(z_dim):
        key, subkey = random.split(key)
        # Select `non_zero_num` rows without replacement, then fill their values.
        indices = random.choice(key, g_dim, shape=(non_zero_num,), replace=False)
        values = random.normal(subkey, (non_zero_num,))
        beta = beta.at[indices, col].set(values)

    return beta


def generate_sim(
    seed: int,
    l_dim: int,
    n_dim: int,
    p_dim: int,
    z_dim: int,
    g_dim: int,
    b_sparsity: float = 0.2,
    effect_size: float = 1.0,
) -> SimulatedData:
    """Create the function to generate a sparse data for PerturbVI.
       Please make sure l_dim < p_dim/z_dim
       when generate simulation data using this function.

    Args:
        seed: Seed for random initialization
        l_dim: Number of single effects in each factor
        n_dim: Number of sample in the data
        p_dim: Number of feature in the data
        z_dim: Number of Latent dimensions
        g_dim: perturbation dimensions
        b_sparsity: sparsity of perturbation effects.
        effect_size: The effect size of features contributing to the factor.
                      (default = 1).

    Returns:
        SimulatedData: Tuple that contains simulated factors (`N x K`),
    """

    _validate_sim_inputs(seed, l_dim, n_dim, p_dim, z_dim, g_dim, b_sparsity, effect_size)

    rng_key = random.PRNGKey(seed)
    rng_key, b_key, beta_key, s_key, var_key, obs_key = random.split(rng_key, 6)

    W = _random_loadings(b_key, z_dim, p_dim, l_dim, effect_size)
    beta = _random_beta(beta_key, z_dim, g_dim, int(b_sparsity * g_dim))

    G = create_design_matrix(s_key, n_dim, g_dim)

    Z = G @ beta + random.normal(var_key, shape=(n_dim, z_dim))
    # Latent factor model
    m = Z @ W
    X = m + random.normal(obs_key, shape=(n_dim, p_dim))

    return SimulatedData(Z, W, X, G, beta)


def generate_sim_with_control(
    seed: int,
    l_dim: int,
    n_dim: int,
    p_dim: int,
    z_dim: int,
    g_dim: int,
    control_fraction: float = 0.2,
    b_sparsity: float = 0.2,
    effect_size: float = 1.0,
) -> SimulatedData:
    """Create the function to generate a sparse data for PerturbVI.
       Please make sure l_dim < p_dim/z_dim
       when generate simulation data using this function.

    Args:
        seed: Seed for random initialization
        l_dim: Number of single effects in each factor
        n_dim: Number of sample in the data
        p_dim: Number of feature in the data
        z_dim: Number of Latent dimensions
        g_dim: perturbation dimensions
        control_fraction: fraction of negative control guide
        b_sparsity: sparsity of perturbation effects.
        effect_size: The effect size of features contributing to the factor.
                      (default = 1).

    Returns:
        SimulatedData: Tuple that contains simulated factors (`N x K`),
    """

    _validate_sim_inputs(seed, l_dim, n_dim, p_dim, z_dim, g_dim, b_sparsity, effect_size)

    if not 0 <= control_fraction < 1:
        raise ValueError(f"control_fraction should be between 0 and 1: received {control_fraction}")

    # Split the requested total sample count into perturbed and control cells.
    control_size = int(n_dim * control_fraction)
    case_size = n_dim - control_size
    if case_size < g_dim:
        raise ValueError(
            f"Not enough perturbed cells to assign every perturbation: "
            f"received {case_size} perturbed cells and g_dim = {g_dim}"
        )

    rng_key = random.PRNGKey(seed)
    rng_key, b_key, beta_key, s_key, var_key, obs_key = random.split(rng_key, 6)

    W = _random_loadings(b_key, z_dim, p_dim, l_dim, effect_size)
    beta = _random_beta(beta_key, z_dim, g_dim, int(b_sparsity * g_dim))

    # Negative-control cells have no perturbation assignment and therefore use
    # all-zero guide rows. G columns remain aligned one-to-one with beta rows.
    G_case = create_design_matrix(s_key, case_size, g_dim)
    G_control = jnp.zeros(shape=(control_size, g_dim))
    G = jnp.vstack((G_case, G_control))

    Z = G @ beta + random.normal(var_key, shape=(n_dim, z_dim))
    # Latent factor model
    m = Z @ W
    X = m + random.normal(obs_key, shape=(n_dim, p_dim))

    return SimulatedData(Z, W, X, G, beta)
