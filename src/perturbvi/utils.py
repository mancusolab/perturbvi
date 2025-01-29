from functools import partial

import numpy as np
import pandas as pd

import equinox as eqx
import jax.scipy.special as jspec
import lineax as lx

from jax import jit, lax, numpy as jnp, random as rdm
from jaxtyping import Array


multi_linear_solve = eqx.filter_vmap(lx.linear_solve, in_axes=(None, 1, None))

_add_ufunc = jnp.frompyfunc(jnp.add, nin=2, nout=1, identity=0)
outer_add = _add_ufunc.outer


def logdet(A: Array) -> Array:
    _, ldet = jnp.linalg.slogdet(A)
    return ldet


def kl_discrete(alpha: Array, pi: Array) -> Array:
    """A function that calculates the Kullback-Leibler divergence for multinomial distributions

    **Arguments:**

    -`alpha` [`Array`]: An array representing the first discrete distribution.

    -`pi` [`Array`]: An array representing the second discrete distribution.

    **Returns:**

    The Kullback-Leibler divergence between the two distributions.
    """
    return jnp.sum(jspec.xlogy(alpha, alpha) - jspec.xlogy(alpha, pi))


@partial(jit, static_argnums=(2, 3, 4))
def prob_pca(rng_key, X, k, max_iter=1000, tol=1e-3):
    """Probabilistic PCA algorithm to initialize latent factors.

    **Arguments:**

    -`rng_key` [`PRNGKey`]: Random key generator.

    -`X` [`Array`]: The observed data.

    -`k` [`int`]: The latent dimension.

    -`max_iter` [`int`]: The maximum number of iterations, default is 1000.

    -`tol` [`float`]: The convergence tolerance, default is 1e-3.

    **Returns:**

    - `Z` [`Array`]: The estimated latent factors.

    -`W` [`Array`]: The estimated loadings.
    """

    n_dim, p_dim = X.shape

    # initial guess for W
    w_key, z_key = rdm.split(rng_key, 2)

    # good enough for initialization
    solver = lx.Cholesky()

    # check if reach the max_iter, or met the norm criterion every 100 iteration
    def _condition(carry):
        i, _, Z, old_Z = carry
        iter_check = i < max_iter
        tol_check = jnp.linalg.norm(Z - old_Z) > tol
        return iter_check & tol_check

    # EM algorithm for PPCA
    def _step(carry):
        i, W, Z, _ = carry

        # E step
        W_op = lx.MatrixLinearOperator(W @ W.T, tags=lx.positive_semidefinite_tag)
        Z_new = multi_linear_solve(W_op, W @ X.T, solver).value

        # M step
        Z_op = lx.MatrixLinearOperator(Z_new.T @ Z_new, tags=lx.positive_semidefinite_tag)
        W = multi_linear_solve(Z_op, Z_new.T @ X, solver).value.T

        return i + 1, W, Z_new, Z

    W = rdm.normal(w_key, shape=(k, p_dim))
    Z = rdm.normal(z_key, shape=(n_dim, k))
    Z_zero = jnp.zeros_like(Z)
    initial_carry = 0, W, Z, Z_zero

    _, W, Z, _ = lax.while_loop(_condition, _step, initial_carry)
    Z, _ = jnp.linalg.qr(Z)

    return Z, W


# Create function to evaluate Local False Sign Rate
# First Create a function to sample single effect matrix based on params.alpha
def bern_sample(alpha):
    """Sample from a Bernoulli distribution with probability alpha.

    **Arguments:**

    - `alpha` [`Array`]: The probability of each row in the L x K matrix.

    **Returns:**

    - `efficient_result_matrix` [`Array`]: The sampled matrix.

    """
    l_dim, z_dim, _ = alpha.shape
    # Generate random numbers for each row in the L x K matrix
    # These random numbers are used as indices for selecting features
    random_indices = np.random.rand(l_dim, z_dim)
    # Calculate the cumulative sum of probabilities along the P dimension
    cumulative_probabilities = np.cumsum(alpha, axis=2)

    # Determine the indices in P where the cumulative probability exceeds the random index
    # This effectively samples from the probability distribution
    feature_indices = np.argmax(cumulative_probabilities > random_indices[..., np.newaxis], axis=2)
    # Initialize the result matrix with zeros
    efficient_result_matrix = np.zeros_like(alpha)

    # Use advanced indexing to set the selected features to 1
    efficient_result_matrix[np.arange(l_dim)[:, np.newaxis], np.arange(z_dim), feature_indices] = 1

    return efficient_result_matrix


def compute_lfsr(params, iters=2000):
    """Compute the LFSR (Local False Sign Rate) using the given parameters.

    **Arguments:**

    - `params` [`ModelParams`]: The parameters of the model.

    - `iters` [`int`]: The number of iterations to run the algorithm. Default is 2000.

    **Returns:**

    - `lfsr` [`Array`]: The LFSR for each of `L` single effects.

    """
    _, _, p_dim = params.alpha.shape
    g_dim, _ = params.mean_beta.shape
    # Reshaping the var_w to (L by K) such that each value in var_w repeats P times
    reshaped_var_w = np.repeat(params.var_w[:, :, np.newaxis], p_dim, axis=2)
    # Initialize count matrix
    total_pos_zero = jnp.zeros(shape=(g_dim, p_dim))
    total_neg_zero = jnp.zeros(shape=(g_dim, p_dim))

    for i in range(iters):
        sample_w = np.random.normal(loc=params.mean_w, scale=np.sqrt(reshaped_var_w))
        sample_alpha = bern_sample(params.alpha)
        sample_W = np.sum(sample_w * sample_alpha, axis=0)

        sample_eta = np.random.binomial(1, params.p_hat.T)
        sample_beta = np.random.normal(loc=params.mean_beta, scale=np.sqrt(params.var_beta))
        sample_B = sample_beta * sample_eta

        sample_oe = sample_B @ sample_W

        ind_pos = (sample_oe >= 0).astype(int)
        ind_neg = (sample_oe <= 0).astype(int)
        total_pos_zero = total_pos_zero + ind_pos
        total_neg_zero = total_neg_zero + ind_neg
        if (i + 1) % 100 == 0:
            print(f"{i - 99}-{i + 1} iters complete")

    lfsr = np.minimum(total_pos_zero, total_neg_zero) / iters
    return lfsr


def pip_analysis(pip: jnp.ndarray, rho=0.9, rho_prime=0.05):
    """Create a function to give a quick summary of PIPs

    Args:
        pip:the pip matrix, a ndarray from results object returned by
        infer.perturbvi

    """
    z_dim, p_dim = pip.shape
    results = []

    print(f"Of {p_dim} features from the data, SuSiE PCA identifies:")
    for k in range(z_dim):
        num_signal = jnp.where(pip[k, :] >= rho)[0].shape[0]
        num_zero = jnp.where(pip[k, :] < rho_prime)[0].shape[0]
        print(f"Component {k} has {num_signal} features with pip>{rho}; and {num_zero} features with pip<{rho_prime}")
        results.append([num_signal, num_zero])

    df = pd.DataFrame(results, columns=["num_signal", "num_zero"])

    # Calculate and print mean and standard deviation for each column
    mean_signal = df["num_signal"].mean()
    std_signal = df["num_signal"].std()
    mean_zero = df["num_zero"].mean()
    std_zero = df["num_zero"].std()

    print(f"Mean and standard deviation for num_signal: {mean_signal}, {std_signal}")
    print(f"Mean and standard deviation for num_zero: {mean_zero}, {std_zero}")

    return df
