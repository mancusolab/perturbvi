from functools import partial

import equinox as eqx
import jax.scipy.special as jspec
import lineax as lx

from jax import jit, lax, numpy as jnp, random as rdm
from jaxtyping import Array


multi_linear_solve = eqx.filter_vmap(lx.linear_solve, in_axes=(None, 1, None))

_add_ufunc = jnp.frompyfunc(jnp.add, nin=2, nout=1, identity=0)
outer_add = _add_ufunc.outer


def logdet(A: Array) -> Array:
    sign, ldet = jnp.linalg.slogdet(A)
    return ldet


def kl_discrete(alpha: Array, pi: Array) -> Array:
    return jnp.sum(jspec.xlogy(alpha, alpha) - jspec.xlogy(alpha, pi))


@partial(jit, static_argnums=(2, 3, 4))
def prob_pca(rng_key, X, k, max_iter=1000, tol=1e-3):
    n_dim, p_dim = X.shape

    # initial guess for W
    w_key, z_key = rdm.split(rng_key, 2)

    # good enough for initialization
    solver = lx.NormalCG(rtol=1e-3, atol=1e-3)

    # check if reach the max_iter, or met the norm criterion every 100 iteration
    def _condition(carry):
        i, W, Z, old_Z = carry
        iter_check = i < max_iter
        tol_check = jnp.linalg.norm(Z - old_Z) > tol
        return iter_check & tol_check

    # EM algorithm for PPCA
    def _step(carry):
        i, W, Z, old_Z = carry

        # E step
        W_op = lx.MatrixLinearOperator(W)
        Z_new = multi_linear_solve(W_op, X.T, solver).value.T

        # M step
        Z_op = lx.MatrixLinearOperator(Z_new.T)
        W = multi_linear_solve(Z_op, X, solver).value

        return i + 1, W, Z_new, Z

    W = rdm.normal(w_key, shape=(p_dim, k))
    Z = rdm.normal(z_key, shape=(k, n_dim))
    Z_zero = jnp.zeros_like(Z)
    initial_carry = 0, W, Z, Z_zero

    _, W, Z, _ = lax.while_loop(_condition, _step, initial_carry)

    return Z.T, W.T
