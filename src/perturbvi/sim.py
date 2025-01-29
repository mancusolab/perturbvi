from typing import NamedTuple, Optional

import numpy as np

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
    key : Random seed for np.random.
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

    # Randomly assign remaining perturbations
    np.random.seed(key)
    indices = np.random.choice(g_dim, size=n_dim - g_dim)
    G_left = np.zeros((n_dim - g_dim, g_dim), dtype=int)
    G_left[np.arange(n_dim - g_dim), indices] = 1

    G = jnp.concatenate((G, G_left), axis=0)
    return G


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
    """Create the function to generate a sparse data for perturbVI.
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

    # interger seed
    if isinstance(seed, int) is False:
        raise ValueError(f"seed should be an interger: received seed = {seed}")

    rng_key = random.PRNGKey(seed)
    rng_key, b_key, beta_key, s_key, var_key, obs_key = random.split(rng_key, 6)

    # dimension check
    if l_dim > p_dim:
        raise ValueError(f"l_dim should be less than p: received l_dim = {l_dim}, p = {p_dim}")
    if l_dim > p_dim / z_dim:
        raise ValueError(
            f"""l_dim is smaller than p_dim/z_dim,
            please make sure each component has {l_dim} single effects"""
        )

    if l_dim <= 0:
        raise ValueError(f"l_dim should be positive: received l_dim = {l_dim}")

    if z_dim > p_dim:
        raise ValueError(f"z_dim should be less than p: received z_dim = {z_dim}, p = {p_dim}")
    if z_dim > n_dim:
        raise ValueError(f"z_dim should be less than n: received z_dim = {z_dim}, n = {n_dim}")
    if z_dim <= 0:
        raise ValueError(f"z_dim should be positive: received z_dim = {z_dim}")

    if effect_size <= 0:
        raise ValueError(f"effect size should be positive: received effect_size = {effect_size}")

    # random W
    W = jnp.zeros(shape=(z_dim, p_dim))

    for k in range(z_dim):
        W = W.at[k, (k * l_dim) : ((k + 1) * l_dim)].set(effect_size * random.normal(b_key, shape=(l_dim,)))

    # linear function to generate Z
    # perturbation effects
    beta = jnp.zeros(shape=(g_dim, z_dim))
    # number of non-zero entries in each component
    non_zero_num = int(b_sparsity * g_dim)

    for col in range(z_dim):
        beta_key, beta_key_2 = random.split(beta_key)
        # Step 2: Randomly select `non_zero_num` indices without replacement
        indices = random.choice(beta_key, g_dim, shape=(non_zero_num,), replace=False)

        # Step 3: Generate `non_zero_num` random values from a normal distribution
        random_values = random.normal(beta_key_2, (non_zero_num,))

        # Update the selected entries in the column with random values
        beta = beta.at[indices, col].set(random_values)

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
    """Create the function to generate a sparse data for perturbVI.
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

    # interger seed
    if isinstance(seed, int) is False:
        raise ValueError(f"seed should be an interger: received seed = {seed}")

    rng_key = random.PRNGKey(seed)
    rng_key, b_key, beta_key, s_key, var_key, obs_key, beta_key_2 = random.split(rng_key, 7)

    # dimension check
    if l_dim > p_dim:
        raise ValueError(f"l_dim should be less than p: received l_dim = {l_dim}, p = {p_dim}")
    if l_dim > p_dim / z_dim:
        raise ValueError(
            f"""l_dim is smaller than p_dim/z_dim,
            please make sure each component has {l_dim} single effects"""
        )

    if l_dim <= 0:
        raise ValueError(f"l_dim should be positive: received l_dim = {l_dim}")

    if z_dim > p_dim:
        raise ValueError(f"z_dim should be less than p: received z_dim = {z_dim}, p = {p_dim}")
    if z_dim > n_dim:
        raise ValueError(f"z_dim should be less than n: received z_dim = {z_dim}, n = {n_dim}")
    if z_dim <= 0:
        raise ValueError(f"z_dim should be positive: received z_dim = {z_dim}")

    if effect_size <= 0:
        raise ValueError(f"effect size should be positive: received effect_size = {effect_size}")

    # control data
    control_size = int(n_dim * control_fraction)

    # random W
    W = jnp.zeros(shape=(z_dim, p_dim))

    for k in range(z_dim):
        W = W.at[k, (k * l_dim) : ((k + 1) * l_dim)].set(effect_size * random.normal(b_key, shape=(l_dim,)))

    # linear function to generate Z
    # perturbation effects
    beta = jnp.zeros(shape=(g_dim, z_dim))
    # effects should be sparse
    non_zero_num = int(b_sparsity * g_dim)

    for col in range(z_dim):
        beta_key, beta_key_2 = random.split(beta_key)
        # Step 2: Randomly select `non_zero_num` indices without replacement
        indices = random.choice(beta_key, g_dim, shape=(non_zero_num,), replace=False)

        # Step 3: Generate `non_zero_num` random values from a normal distribution
        random_values = random.normal(beta_key_2, (non_zero_num,))

        # Update the selected entries in the column with random values
        beta = beta.at[indices, col].set(random_values)

    # compose new G with control cell
    G_case = create_design_matrix(s_key, n_dim, g_dim)
    G_add_zero_col = jnp.insert(G_case, 0, values=0, axis=1)
    G_control = jnp.hstack((jnp.zeros(shape=(control_size, g_dim)), jnp.ones(shape=(control_size, 1))))
    G = jnp.vstack((G_add_zero_col, G_control))
    beta_add = jnp.vstack((beta, 0.5 * random.normal(beta_key_2, shape=(1, z_dim))))
    # beta_add_zero_row = jnp.insert(beta, beta.shape[0], values=0, axis=0)

    Z = G @ beta_add + random.normal(var_key, shape=(n_dim + control_size, z_dim))
    # Latent factor model
    m = Z @ W
    X = m + random.normal(obs_key, shape=(n_dim + control_size, p_dim))

    return SimulatedData(Z, W, X, G, beta_add)
