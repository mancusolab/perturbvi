# pattern: Functional Core

import logging
import numbers

from datetime import datetime
from functools import partial

import numpy as np

import equinox as eqx
import jax.scipy.special as jspec
import lineax as lx

from jax import jit, lax, numpy as jnp, random as rdm
from jaxtyping import Array

from .common import ModelParams
from .log import get_logger


log = get_logger("perturbvi")
log.setLevel(logging.INFO)

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


def kl_bernoulli(q: Array, p: Array, eps: float = 1e-8) -> Array:
    """Calculate the KL divergence between Bernoulli distributions."""
    q = jnp.clip(q, eps, 1.0 - eps)
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.sum(jspec.xlogy(q, q) - jspec.xlogy(q, p) + jspec.xlog1py(1.0 - q, -q) - jspec.xlog1py(1.0 - q, -p))


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
        # scaled_tol_check = tol_check / n_dim > tol
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


def bern_sample_jax(key, alpha):
    """JAX version of bern_sample function.

    Arguments:
        key: JAX random key
        alpha: probability matrix of shape (l_dim, z_dim, p_dim)
    """
    random_values = rdm.uniform(key, shape=alpha.shape[:-1])
    cumsum = jnp.cumsum(alpha, axis=-1)
    return jnp.eye(alpha.shape[-1])[jnp.argmax(cumsum > random_values[..., None], axis=-1)]


@partial(jit, static_argnums=(2,))
def _compute_lfsr_step(key, params, iters):
    """Jitted inner loop of LFSR computation"""
    l_dim, z_dim, p_dim = params.alpha.shape
    g_dim, _ = params.mean_beta.shape
    reshaped_var_w = jnp.repeat(params.var_w[:, :, jnp.newaxis], p_dim, axis=2)

    def _inner_loop(carry, i):  # Modified to accept iteration index
        key, total_pos, total_neg = carry

        # Split keys for different random operations
        key, w_key, a_key, e_key, b_key = rdm.split(key, 5)

        # Sample W
        sample_w = params.mean_w + jnp.sqrt(reshaped_var_w) * rdm.normal(w_key, shape=params.mean_w.shape)
        sample_alpha = bern_sample_jax(a_key, params.alpha)
        sample_W = jnp.sum(sample_w * sample_alpha, axis=0)

        # Sample B
        sample_eta = rdm.bernoulli(e_key, params.p_hat.T)
        sample_beta = params.mean_beta + jnp.sqrt(params.var_beta) * rdm.normal(b_key, shape=params.mean_beta.shape)
        sample_B = sample_beta * sample_eta

        # Compute outer product
        sample_oe = sample_B @ sample_W
        ind_pos = sample_oe >= 0
        ind_neg = sample_oe <= 0

        return (key, total_pos + ind_pos, total_neg + ind_neg), None

    # Initialize
    total_pos_zero = jnp.zeros((g_dim, p_dim))
    total_neg_zero = jnp.zeros((g_dim, p_dim))
    init_carry = (key, total_pos_zero, total_neg_zero)

    # Run the loop
    (_, total_pos_zero, total_neg_zero), _ = lax.scan(_inner_loop, init_carry, jnp.arange(iters))

    return total_pos_zero, total_neg_zero


def compute_lfsr(key: Array, params: ModelParams, iters: int = 2000) -> Array:
    """Compute the LFSR (Local False Sign Rate) using the given parameters.

    Arguments:
        key: JAX random key
        params: The parameters of the model
        iters: Positive number of iterations (default=2000)
    """
    if isinstance(iters, bool) or not isinstance(iters, numbers.Integral) or iters <= 0:
        raise ValueError(f"iters must be a positive integer; received {iters!r}")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"Start computing LFSR at {current_time}")

    # Split computation into chunks to show progress
    chunk_size = 100
    num_chunks = iters // chunk_size
    remaining = iters % chunk_size

    total_pos = 0
    total_neg = 0

    for i in range(num_chunks):
        iter_key = rdm.fold_in(key, i * chunk_size)
        pos_chunk, neg_chunk = _compute_lfsr_step(iter_key, params, chunk_size)
        total_pos += pos_chunk
        total_neg += neg_chunk
        log.info(f"Completed {(i + 1) * chunk_size}/{iters} iterations")

    # Handle remaining iterations if any
    if remaining > 0:
        iter_key = rdm.fold_in(key, num_chunks * chunk_size)
        pos_rem, neg_rem = _compute_lfsr_step(iter_key, params, remaining)
        total_pos += pos_rem
        total_neg += neg_rem
        log.info(f"Completed {iters}/{iters} iterations")

    # Compute final LFSR
    lfsr = jnp.minimum(total_pos, total_neg) / iters

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"Finished computing LFSR at {current_time}")

    return lfsr
