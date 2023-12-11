from functools import partial
from typing import get_args, Literal, NamedTuple, Optional, Tuple

from .common import (
    compute_pip,
    compute_pve,
    ELBOResults_Design,
    ModelParams_Design,
    SuSiEPCAResults_Design,
)
from plum import dispatch
from .sparse import SparseMatrix

# from memory_profiler import profile
import equinox as eqx
import optax

from jax import Array, grad, jit, lax, nn, numpy as jnp, random
from jax.experimental import sparse
from jax.scipy import special as spec
from jax.typing import ArrayLike


__all__ = [
    "compute_elbo",
    "susie_pca",
]

_init_type = Literal["pca", "random"]


@dispatch
def _is_valid(X: ArrayLike):
    return jnp.all(jnp.isfinite(X))


@dispatch
def _is_valid(X: sparse.JAXSparse):
    return jnp.all(jnp.isfinite(X.data))


@dispatch
def _is_valid(G: ArrayLike):
    return jnp.all(jnp.isfinite(G))


@dispatch
def _is_valid(G: sparse.JAXSparse):
    return jnp.all(jnp.isfinite(G.data))


# define EM algorithm for PPCA to extract PPCA initialization
@partial(jit, static_argnums=(2, 3, 4))
def prob_pca(rng_key, X, k, max_iter=1000, tol=1e-3):
    n_dim, p_dim = X.shape

    # initial guess for W
    w_key, z_key = random.split(rng_key, 2)

    # check if reach the max_iter, or met the norm criterion every 100 iteration
    def _condition(carry):
        i, W, Z, old_Z = carry
        tol_check = jnp.linalg.norm(Z - old_Z) > tol
        return (i < max_iter) & ((i % 100 != 0) | tol_check)

    # EM algorithm for PPCA
    def _step(carry):
        i, W, Z, old_Z = carry

        # E step
        Z_new = jnp.linalg.solve(W.T @ W, W.T @ X.T)

        # M step
        W_new = jnp.linalg.solve(Z_new @ Z_new.T, Z_new @ X).T

        return i + 1, W_new, Z_new, Z

    W = random.normal(w_key, shape=(p_dim, k))
    Z = random.normal(z_key, shape=(k, n_dim))
    Z_zero = jnp.zeros_like(Z)
    initial_carry = 0, W, Z, Z_zero

    _, W, Z, _ = lax.while_loop(_condition, _step, initial_carry)
    # lets return transpose to match SuSiE-PCA
    return Z.T, W


def _logdet(A: ArrayLike) -> float:
    sign, ldet = jnp.linalg.slogdet(A)
    return ldet


def _compute_pi(A: Array, theta: Array) -> Array:
    return nn.softmax(A @ theta, axis=0).T


def _kl_gamma(alpha: Array, pi: Array) -> Array:
    return jnp.sum(spec.xlogy(alpha, alpha) - spec.xlogy(alpha, pi))


def _compute_w_moment(params: ModelParams_Design) -> Tuple[Array, Array]:
    trace_var = jnp.sum(
        params.var_w[:, :, jnp.newaxis] * params.alpha + (params.mu_w**2 * params.alpha * (1 - params.alpha)),
        axis=(-1, 0),
    )

    E_W = params.W
    E_WW = E_W @ E_W.T + jnp.diag(trace_var)

    return E_W, E_WW


# Update posterior mean and variance W
def _update_w(RtZk: Array, E_zzk: Array, params: ModelParams_Design, kdx: int, ldx: int) -> ModelParams_Design:
    # n_dim, z_dim = params.mu_z.shape

    # calculate update_var_w as the new V[w | gamma]
    # suppose indep between w_k
    update_var_wkl = jnp.reciprocal(params.tau * E_zzk + params.tau_0[ldx, kdx])

    # calculate update_mu_w as the new E[w | gamma]
    update_mu_wkl = params.tau * update_var_wkl * RtZk

    return params._replace(
        mu_w=params.mu_w.at[ldx, kdx].set(update_mu_wkl),
        var_w=params.var_w.at[ldx, kdx].set(update_var_wkl),
    )


def _update_beta(
    ZkG: ArrayLike, G: SparseMatrix, GtG_diag: ArrayLike, params: ModelParams_Design, kdx: int
) -> ModelParams_Design:
    update_var_beta = jnp.reciprocal(params.tau_beta[kdx] + GtG_diag)
    update_mu_beta = update_var_beta * ZkG

    return params._replace(
        mu_beta=params.mu_beta.at[:, kdx].set(update_mu_beta),
        var_beta=params.var_beta.at[:, kdx].set(update_var_beta),
    )


def _update_p_hat(params: ModelParams_Design, kdx: int):
    log_bf = (
        jnp.log(params.p)
        - jnp.log(1 - params.p)
        + 0.5
        * (
            jnp.log(params.var_beta[:, kdx])
            + jnp.log(params.tau_beta[kdx])
            + (params.mu_beta[:, kdx] ** 2 / params.var_beta[:, kdx])
        )
    )
    # note that p_hat is k by g
    p_hat = nn.sigmoid(log_bf)

    return params._replace(
        p_hat=params.p_hat.at[kdx, :].set(p_hat),
    )


# Compute log of Bayes factor
def _log_bf_np(z: ArrayLike, s2: ArrayLike, s0: ArrayLike):
    return 0.5 * (jnp.log(s2) - jnp.log(s2 + 1 / s0)) + 0.5 * z**2 * ((1 / s0) / (s2 + 1 / s0))


# Update posterior of alpha using Bayes factor
def _update_alpha_bf(
    RtZk: ArrayLike, E_zzk: ArrayLike, params: ModelParams_Design, kdx: int, ldx: int
) -> ModelParams_Design:
    Z_s = (RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)
    s20_s = params.tau_0[ldx, kdx]

    log_bf = _log_bf_np(Z_s, s2_s, s20_s)
    log_alpha = jnp.log(params.pi) + log_bf
    alpha_kl = nn.softmax(log_alpha)

    params = params._replace(
        alpha=params.alpha.at[ldx, kdx].set(alpha_kl),
    )

    return params


# Update posterior of alpha using Bayes factor in SuSiE PCA+Anno
def _update_alpha_bf_annotation(
    RtZk: ArrayLike, E_zzk: ArrayLike, params: ModelParams_Design, kdx: int, ldx: int
) -> ModelParams_Design:
    Z_s = (RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)
    s20_s = params.tau_0[ldx, kdx]

    log_bf = _log_bf_np(Z_s, s2_s, s20_s)
    log_alpha = jnp.log(params.pi[kdx, :]) + log_bf
    alpha_kl = nn.softmax(log_alpha)

    params = params._replace(
        alpha=params.alpha.at[ldx, kdx].set(alpha_kl),
    )

    return params


# Update Posterior mean and variance of Z
def _update_z(
    X: ArrayLike | SparseMatrix, G: ArrayLike | SparseMatrix, params: ModelParams_Design
) -> ModelParams_Design:
    E_W, E_WW = _compute_w_moment(params)
    z_dim, _ = E_W.shape

    update_var_z = jnp.linalg.inv(params.tau * E_WW + jnp.identity(z_dim))
    update_mu_z = (params.tau * X @ E_W.T + G @ params.B) @ update_var_z

    return params._replace(mu_z=update_mu_z, var_z=update_var_z)


# Update tau_0 based on MLE
def _update_tau0_mle(params: ModelParams_Design) -> ModelParams_Design:
    # l_dim, z_dim, p_dim = params.mu_w.shape

    est_varw = params.mu_w**2 + params.var_w[:, :, jnp.newaxis]

    u_tau_0 = jnp.sum(params.alpha, axis=-1) / jnp.sum(est_varw * params.alpha, axis=-1)

    return params._replace(tau_0=u_tau_0)


# Update tau_beta based on MLE
def _update_tau_beta_mle(params: ModelParams_Design) -> ModelParams_Design:
    # l_dim, z_dim, p_dim = params.mu_w.shape

    est_var_beta = params.mu_beta**2 + params.var_beta

    u_tau_beta = jnp.sum(params.p_hat, axis=-1) / jnp.sum(est_var_beta * params.p_hat.T, axis=0)

    return params._replace(tau_beta=u_tau_beta)


# Update tau based on MLE
def _update_tau(X: Array | SparseMatrix, params: ModelParams_Design) -> ModelParams_Design:
    n_dim, z_dim = params.mu_z.shape
    l_dim, z_dim, p_dim = params.mu_w.shape

    # calculate second moment of Z; (k x k) matrix
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # calculate moment of W
    E_W, E_WW = _compute_w_moment(params)

    # expectation of log likelihood
    # E_ss = params.ssq - 2 * jnp.trace(E_W @ X.T @ params.mu_z) + jnp.trace(E_ZZ @ E_WW)
    E_ss = jnp.sum(X * X) - 2 * jnp.trace(E_W @ X.T @ params.mu_z) + jnp.trace(E_ZZ @ E_WW)
    u_tau = (n_dim * p_dim) / E_ss

    return params._replace(tau=u_tau)


def _update_theta(
    params: ModelParams_Design,
    A: Array,
    lr: float = 1e-2,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> ModelParams_Design:
    optimizer = optax.adam(lr)
    theta = params.theta
    old_theta = jnp.zeros_like(params.theta)
    opt_state = optimizer.init(params.theta)

    def _loss(theta_i: Array) -> float:
        pi = _compute_pi(A, theta_i)
        return _kl_gamma(params.alpha, pi)

    def body_fun(inputs):
        old_theta, theta, idx, opt_state = inputs
        grads = grad(_loss)(theta)
        updates, new_optimizer_state = optimizer.update(grads, opt_state)
        new_theta = optax.apply_updates(theta, updates)
        old_theta = theta
        return old_theta, new_theta, idx + 1, new_optimizer_state

    # define a function to check the stopping criterion
    def cond_fn(inputs):
        old_theta, theta, idx, _ = inputs
        tol_check = jnp.linalg.norm(theta - old_theta) > tol
        iter_check = idx < max_iter
        return jnp.logical_and(tol_check, iter_check)

    # use jax.lax.while_loop until the change in parameters is less than a given tolerance
    old_theta, theta, idx_count, opt_state = lax.while_loop(
        cond_fn,
        body_fun,
        (old_theta, theta, 0, opt_state),
    )

    return params._replace(theta=theta, pi=_compute_pi(A, theta))


def compute_elbo(
    X: ArrayLike | SparseMatrix, G: ArrayLike | SparseMatrix, GtG_diag: ArrayLike, params: ModelParams_Design
) -> ELBOResults_Design:
    """Create function to compute evidence lower bound (ELBO)

    Args:
        X: the observed data, an N by P ndarray
        G: secondary data.
        GtG_diag: diagonal element of GTG.
        params: the dictionary contains all the infered parameters

    Returns:
        ELBOResult_Design: the object contains all components in ELBO

    """
    n_dim, z_dim = params.mu_z.shape
    l_dim, z_dim, p_dim = params.mu_w.shape

    # calculate second moment of Z along k, (k x k) matrix
    # E[Z'Z] = V_k[Z] * tr(I_n) + E[Z]'E[Z] = V_k[Z] * n + E[Z]'E[Z]
    E_ZZ = n_dim * params.var_z + params.mu_z.T @ params.mu_z
    # calculate moment of B, length k vector (in the diagonal)
    E_BB = jnp.sum((params.mu_beta**2 + params.var_beta) * params.p_hat.T, axis=1)

    # calculate moment of W
    E_W, E_WW = _compute_w_moment(params)

    # expectation of log likelihood
    # calculation tip: tr(A @ A.T) = tr(A.T @ A) = sum(A ** 2)
    # (X.T @ E[Z] @ E[W]) is p x p (big!); compute (E[W] @ X.T @ E[Z]) (k x k)
    E_ll = (-0.5 * params.tau) * (
        jnp.sum(X**2)
        - 2 * jnp.einsum("kp,np,nk->", E_W, X, params.mu_z)  # tr(E[W] @ X.T @ E[Z])
        + jnp.einsum("ij,ji->", E_ZZ, E_WW)  # tr(E[Z.T @ Z] @ E[W @ W.T])
    ) + 0.5 * n_dim * p_dim * jnp.log(params.tau)

    # neg-KL for Z
    Z_pred = G @ params.B
    negKL_z = -0.5 * (
        jnp.trace(E_ZZ)
        - 2 * jnp.trace(params.mu_z.T @ Z_pred)
        + jnp.sum(GtG_diag * E_BB)
        - n_dim * z_dim
        - n_dim * _logdet(params.var_z)
    )
    # neg-KL for w
    # awkward indexing to get broadcast working
    klw_term1 = params.tau_0[:, :, jnp.newaxis] * (params.var_w[:, :, jnp.newaxis] + params.mu_w**2)
    klw_term2 = klw_term1 - 1.0 - (jnp.log(params.tau_0) + jnp.log(params.var_w))[:, :, jnp.newaxis]
    negKL_w = -0.5 * jnp.sum(params.alpha * klw_term2)

    # neg-KL for gamma
    negKL_gamma = -_kl_gamma(params.alpha, params.pi)

    # neg-KL for beta
    negKL_beta = -0.5 * jnp.sum(
        (params.mu_beta**2 + params.var_beta) * params.tau_beta
        - 1
        - jnp.log(params.var_beta)
        - jnp.log(params.tau_beta)
    )
    # neg-KL for eta
    negKL_eta = -_kl_gamma(params.p_hat, params.p)

    elbo = E_ll + negKL_z + negKL_w + negKL_gamma + negKL_beta + negKL_eta

    result = ELBOResults_Design(elbo, E_ll, negKL_z, negKL_w, negKL_gamma, negKL_beta, negKL_eta)

    return result


class _FactorLoopResults(NamedTuple):
    X: Array | SparseMatrix
    W: Array
    EZZ: Array
    params: ModelParams_Design


class _EffectLoopResults(NamedTuple):
    E_zzk: Array
    RtZk: Array
    Wk: Array
    k: int
    params: ModelParams_Design


class _BetaLoopResults(NamedTuple):
    G: ArrayLike | SparseMatrix
    GtG_diag: ArrayLike
    params: ModelParams_Design


def _factor_loop(kdx: int, loop_params: _FactorLoopResults) -> _FactorLoopResults:
    X, W, E_ZZ, params = loop_params

    l_dim, z_dim, p_dim = params.mu_w.shape

    # sufficient stats for inferring downstream w_kl/alpha_kl
    not_kdx = jnp.where(jnp.arange(z_dim) != kdx, size=z_dim - 1)
    E_zpzk = E_ZZ[kdx][not_kdx]
    E_zzk = E_ZZ[kdx, kdx]
    Wk = W[kdx, :]
    Wnk = W[not_kdx]
    RtZk = params.mu_z[:, kdx] @ X - Wnk.T @ E_zpzk

    # update over each of L effects
    init_loop_param = _EffectLoopResults(E_zzk, RtZk, Wk, kdx, params)
    _, _, Wk, _, params = lax.fori_loop(
        0,
        l_dim,
        _effect_loop,
        init_loop_param,
    )

    return loop_params._replace(W=W.at[kdx].set(Wk), params=params)


def _effect_loop(ldx: int, effect_params: _EffectLoopResults) -> _EffectLoopResults:
    E_zzk, RtZk, Wk, kdx, params = effect_params

    # remove current kl'th effect and update its expected residual
    Wkl = Wk - (params.mu_w[ldx, kdx] * params.alpha[ldx, kdx])
    E_RtZk = RtZk - E_zzk * Wkl

    # update conditional w_kl and alpha_kl based on current residual
    params = _update_w(E_RtZk, E_zzk, params, kdx, ldx)
    params = _update_alpha_bf(E_RtZk, E_zzk, params, kdx, ldx)

    # update marginal w_kl
    Wk = Wkl + (params.mu_w[ldx, kdx] * params.alpha[ldx, kdx])

    return effect_params._replace(Wk=Wk, params=params)


def _beta_loop(kdx, beta_params: _BetaLoopResults) -> _BetaLoopResults:
    G, GtG_diag, params = beta_params
    # compute E[Z'k]G: remove the g-th effect
    # however notice that only intercept in non-zero in GtG off-diag
    # ZkG = params.mu_z[:,kdx].T @ G - GtG_diag * params.B[-1,kdx]
    # without intercept
    ZkG = params.mu_z[:, kdx].T @ G

    params = _update_beta(ZkG, G, GtG_diag, params, kdx)
    params = _update_p_hat(params, kdx)

    return beta_params._replace(params=params)


# loop function for annotation model
def _factor_loop_annotation(kdx: int, loop_params: _FactorLoopResults) -> _FactorLoopResults:
    X, W, E_ZZ, params = loop_params

    l_dim, z_dim, p_dim = params.mu_w.shape

    # sufficient stats for inferring downstream w_kl/alpha_kl
    not_kdx = jnp.where(jnp.arange(z_dim) != kdx, size=z_dim - 1)
    E_zpzk = E_ZZ[kdx][not_kdx]
    E_zzk = E_ZZ[kdx, kdx]
    Wk = W[kdx, :]
    Wnk = W[not_kdx]
    RtZk = params.mu_z[:, kdx] @ X - Wnk.T @ E_zpzk

    # update over each of L effects
    init_loop_param = _EffectLoopResults(E_zzk, RtZk, Wk, kdx, params)
    _, _, Wk, _, params = lax.fori_loop(
        0,
        l_dim,
        _effect_loop_annotation,
        init_loop_param,
    )

    return loop_params._replace(W=W.at[kdx].set(Wk), params=params)


def _effect_loop_annotation(ldx: int, effect_params: _EffectLoopResults) -> _EffectLoopResults:
    E_zzk, RtZk, Wk, kdx, params = effect_params

    # remove current kl'th effect and update its expected residual
    Wkl = Wk - (params.mu_w[ldx, kdx] * params.alpha[ldx, kdx])
    E_RtZk = RtZk - E_zzk * Wkl

    # update conditional w_kl and alpha_kl based on current residual
    params = _update_w(E_RtZk, E_zzk, params, kdx, ldx)
    params = _update_alpha_bf_annotation(E_RtZk, E_zzk, params, kdx, ldx)

    # update marginal w_kl
    Wk = Wkl + (params.mu_w[ldx, kdx] * params.alpha[ldx, kdx])

    return effect_params._replace(Wk=Wk, params=params)


@eqx.filter_jit
def _inner_loop(
    X: ArrayLike | SparseMatrix,
    G: ArrayLike | SparseMatrix,
    GtG_diag: ArrayLike,
    params: ModelParams_Design,
):
    n_dim, z_dim = params.mu_z.shape
    l_dim, _, _ = params.mu_w.shape

    # compute expected residuals
    # use posterior mean of Z, W, and Alpha to calculate residuals
    W = params.W
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # update effect precision via MLE
    params = _update_tau0_mle(params)

    # update locals (W, alpha)
    init_loop_param = _FactorLoopResults(X, W, E_ZZ, params)
    _, W, _, params = lax.fori_loop(0, z_dim, _factor_loop, init_loop_param)

    # update factor parameters
    params = _update_z(X, G, params)

    # update beta and p_hat
    init_beta_params = _BetaLoopResults(G, GtG_diag, params)
    _, _, params = lax.fori_loop(0, z_dim, _beta_loop, init_beta_params)
    # update precision paramter for beta
    params = _update_tau_beta_mle(params)

    # update precision parameters via MLE
    params = _update_tau(X, params)

    # compute elbo
    elbo_res = compute_elbo(X, G, GtG_diag, params)

    return elbo_res, params


@eqx.filter_jit
def _annotation_inner_loop(
    X: ArrayLike | SparseMatrix,
    G: ArrayLike | SparseMatrix,
    GtG_diag: ArrayLike,
    A: ArrayLike,
    params: ModelParams_Design,
):
    n_dim, z_dim = params.mu_z.shape
    l_dim, _, _ = params.mu_w.shape

    # compute expected residuals
    # use posterior mean of Z, W, and Alpha to calculate residuals
    W = params.W
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # perform MLE inference before variational inference
    # update effect precision via MLE
    params = _update_tau0_mle(params)

    # update theta via MLE
    params = _update_theta(params, A)

    # update locals (W, alpha)
    init_loop_param = _FactorLoopResults(X, W, E_ZZ, params)
    _, W, _, params = lax.fori_loop(0, z_dim, _factor_loop_annotation, init_loop_param)

    # update factor parameters
    params = _update_z(X, G, params)

    # update beta and p_hat
    init_beta_params = _BetaLoopResults(G, GtG_diag, params)
    _, _, params = lax.fori_loop(0, z_dim, _beta_loop, init_beta_params)
    # update precision paramter for beta
    params = _update_tau_beta_mle(params)

    # update precision parameters via MLE
    params = _update_tau(X, params)

    # compute elbo
    elbo_res = compute_elbo(X, G, GtG_diag, params)

    return elbo_res, params


def _reorder_factors_by_pve(
    A: ArrayLike, params: ModelParams_Design, pve: ArrayLike
) -> Tuple[ModelParams_Design, Array]:
    sorted_indices = jnp.argsort(pve)[::-1]
    pve = pve[sorted_indices]
    sorted_mu_z = params.mu_z[:, sorted_indices]
    sorted_var_z = params.var_z[sorted_indices, sorted_indices]
    sorted_mu_beta = params.mu_beta[:, sorted_indices]
    sorted_var_beta = params.var_beta[:, sorted_indices]
    sorted_p_hat = params.p_hat[sorted_indices, :]
    sorted_mu_w = params.mu_w[:, sorted_indices, :]
    sorted_var_w = params.var_w[:, sorted_indices]
    sorted_tau_beta = params.tau_beta[sorted_indices]
    sorted_alpha = params.alpha[:, sorted_indices, :]
    sorted_tau_0 = params.tau_0[:, sorted_indices]
    if A is not None:
        sorted_theta = params.theta[:, sorted_indices]
        sorted_pi = _compute_pi(A, sorted_theta)
    else:
        sorted_theta = None
        sorted_pi = params.pi

    params = ModelParams_Design(
        sorted_mu_z,
        sorted_var_z,
        sorted_mu_w,
        sorted_var_w,
        sorted_alpha,
        params.tau,
        sorted_tau_0,
        sorted_theta,
        sorted_pi,
        sorted_mu_beta,
        sorted_var_beta,
        sorted_tau_beta,
        params.p,
        sorted_p_hat,
    )

    return params, pve


def _init_params(
    rng_key: random.PRNGKey,
    X: ArrayLike | SparseMatrix,
    G: ArrayLike | SparseMatrix,
    z_dim: int,
    l_dim: int,
    A: Optional[ArrayLike] = None,
    p_prior: float = 0.5,
    tau: float = 1.0,
    init: _init_type = "pca",
) -> ModelParams_Design:
    """Initialize parameters for SuSiE PCA.

    Args:
        rng_key: Random number generator seed
        X: Input data. Should be an array-like
        z_dim: Latent factor dimension (K)
        l_dim: Number of single-effects comprising each factor (L)
        p_prior: Prior probability for each perturbation being non-zero.
        init: How to initialize the variational mean parameters for latent factors.
            Either "pca" or "random" (default = "pca")
        tau: initial value of residual precision

    Returns:
        ModelParams_Design: initialized set of model parameters.

    Raises:
        ValueError: Invalid initialization scheme.
    """

    tau_0 = jnp.ones((l_dim, z_dim))

    n_dim, p_dim = X.shape

    (
        rng_key,
        svd_key,
        mu_key,
        var_key,
        muw_key,
        varw_key,
        alpha_key,
        beta_key,
        var_beta_key,
        p_key,
        theta_key,
    ) = random.split(rng_key, 11)

    # pull type options for init
    type_options = get_args(_init_type)

    if init == "pca":
        # run PCA and extract weights and latent
        init_mu_z, _ = prob_pca(svd_key, X, k=z_dim)
    elif init == "random":
        # random initialization
        init_mu_z = random.normal(mu_key, shape=(n_dim, z_dim))
    else:
        raise ValueError(f"Unknown initialization provided '{init}'; Choices: {type_options}")

    init_var_z = jnp.diag(random.normal(var_key, shape=(z_dim,)) ** 2)

    # each w_kl has a specific variance term
    init_mu_w = random.normal(muw_key, shape=(l_dim, z_dim, p_dim)) * 1e-3
    init_var_w = (1 / tau_0) * (random.normal(varw_key, shape=(l_dim, z_dim))) ** 2

    init_alpha = random.dirichlet(alpha_key, alpha=jnp.ones(p_dim), shape=(l_dim, z_dim))
    if A is not None:
        p_dim, m = A.shape
        theta = random.normal(theta_key, shape=(m, z_dim))
        pi = _compute_pi(A, theta)
    else:
        theta = None
        pi = jnp.ones(p_dim) / p_dim

    # Initialization for perturbation effects
    n_dim, g_dim = G.shape
    # Notice: shape maybe (z_dim,): same for all feature in each component
    tau_beta = jnp.ones((z_dim,))
    init_mu_beta = random.normal(beta_key, shape=(g_dim, z_dim)) * 1e-3
    init_var_beta = (1 / tau_beta) * random.normal(var_beta_key, shape=(g_dim, z_dim)) ** 2
    # uniform prior for eta
    p = p_prior * jnp.ones(g_dim)
    # Initialization of variational params for eta
    # p_hat is in shape of (z_dim,g_dim)
    p_hat = 0.5 * jnp.ones(shape=(z_dim, g_dim))

    return ModelParams_Design(
        init_mu_z,
        init_var_z,
        init_mu_w,
        init_var_w,
        init_alpha,
        tau,
        tau_0,
        theta=theta,
        pi=pi,
        mu_beta=init_mu_beta,
        var_beta=init_var_beta,
        tau_beta=tau_beta,
        p=p,
        p_hat=p_hat,
    )


def _check_args(X: ArrayLike, A: Optional[ArrayLike], z_dim: int, l_dim: int, init: _init_type):
    # pull type options for init
    type_options = get_args(_init_type)

    if len(X.shape) != 2:
        raise ValueError(f"Shape of X = {X.shape}; Expected 2-dim matrix")

    # should we check for n < p?
    n_dim, p_dim = X.shape

    # dim checks
    if l_dim > p_dim:
        raise ValueError(f"l_dim should be less than p: received l_dim = {l_dim}, p = {p_dim}")
    if l_dim <= 0:
        raise ValueError(f"l_dim should be positive: received l_dim = {l_dim}")
    if z_dim > p_dim:
        raise ValueError(f"z_dim should be less than p: received z_dim = {z_dim}, p = {p_dim}")
    if z_dim > n_dim:
        raise ValueError(f"z_dim should be less than n: received z_dim = {z_dim}, n = {n_dim}")
    if z_dim <= 0:
        raise ValueError(f"z_dim should be positive: received z_dim = {z_dim}")
    # quality checks
    if not _is_valid(X):
        raise ValueError("X contains 'nan/inf'. Please check input data for correctness or missingness")

    if A is not None:
        if len(A.shape) != 2:
            raise ValueError(f"Dimension of annotation matrix A should be 2: received {len(A.shape)}")
        a_p_dim, _ = A.shape
        if a_p_dim != p_dim:
            raise ValueError(
                f"Leading dimension of annotation matrix A should match feature dimension {p_dim}: received {a_p_dim}"
            )
        if not _is_valid(A):
            raise ValueError("A contains 'nan/inf'. Please check input data for correctness or missingness")
    # type check for init

    if init not in type_options:
        raise ValueError(f"Unknown initialization provided '{init}'; Choices: {type_options}")

    return


def susie_pca(
    X: ArrayLike | sparse.JAXSparse,
    z_dim: int,
    l_dim: int,
    G: ArrayLike | sparse.JAXSparse,
    A: Optional[ArrayLike] = None,
    p_prior: float = 0.5,
    tau: float = 1.0,
    standardize: bool = False,
    init: _init_type = "pca",
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-3,
    verbose: bool = True,
) -> SuSiEPCAResults_Design:
    """The main inference function for SuSiE PCA.

    Args:
        X: Input data. Should be an array-like
        z_dim: Latent factor dimension (int; K)
        l_dim: Number of single-effects comprising each factor (int; L)
        G: Secondary information. Should be an array-like or sparse JAX matrix.
        A: Annotation matrix to use in parameterized-prior mode. If not `None`, leading dimension
            should match the feature dimension of X.
        p_prior: Prior probability for each perturbation being non-zero.
        tau: initial value of residual precision (default = 1)
        standardize: Whether to center and scale the input data with mean 0
            and variance 1 (default = False)
        init: How to initialize the variational mean parameters for latent factors.
            Either "pca" or "random" (default = "pca")
        seed: Seed for "random" initialization (int)
        max_iter: Maximum number of iterations for inference (int)
        tol: Numerical tolerance for ELBO convergence (float)
        verbose: Flag to indicate displaying log information (ELBO value) in each
            iteration

    Returns:
        :py:obj:`SuSiEPCAResults_Design`: tuple that has member variables for learned
        parameters (:py:obj:`ModelParams_Design`), evidence lower bound (ELBO) results
        (:py:obj:`ELBOResults`) from the last iteration, the percent of variance
        explained (PVE) for each of the `K` factors (:py:obj:`jax.numpy.ndarray`),
        the posterior inclusion probabilities (PIPs) for each of the `K` factors
        and `P` features (:py:obj:`jax.numpy.ndarray`).

    Raises:
        ValueError: Invalid `l_dim` or `z_dim` values. Invalid initialization scheme.
        Data `X` contains `inf` or `nan`. If annotation matrix `A` is not `None`, raises
        if `A` contains `inf`, `nan` or does not match feature dimension with `X`.
    """

    # sanity check arguments
    _check_args(X, A, z_dim, l_dim, init)

    # cast to jax array
    if isinstance(X, ArrayLike):
        X = jnp.asarray(X)
        # option to center the data
        X -= jnp.mean(X, axis=0)
        if standardize:
            X /= jnp.std(X, axis=0)
    elif isinstance(X, sparse.JAXSparse):
        X = SparseMatrix(X, scale=standardize)
    if isinstance(G, ArrayLike):
        G_sp = jnp.asarray(G)
        # extract diagonal of G.T @ G
        GtG_diag = jnp.sum(G**2, axis=0)
    elif isinstance(G, sparse.JAXSparse):
        G_sp = SparseMatrix(G)
        # extract diagonal of G.T @ G when G is a Binary matrix
        GtG_diag = sparse.bcoo_reduce_sum(G, axes=(0,)).todense()

    # initialize PRNGkey and params
    rng_key = random.PRNGKey(seed)
    params = _init_params(rng_key, X, G_sp, z_dim, l_dim, A, p_prior,tau, init)

    #  core loop for inference
    elbo = -5e25
    for idx in range(1, max_iter + 1):
        if A is not None:
            elbo_res, params = _annotation_inner_loop(X, G_sp, GtG_diag, A, params)
        else:
            elbo_res, params = _inner_loop(X, G_sp, GtG_diag, params)

        if verbose:
            print(f"Iter [{idx}] | {elbo_res}")

        diff = elbo_res.elbo - elbo
        if diff < 0 and verbose:
            print(f"Alert! Diff between elbo[{idx - 1}] and elbo[{idx}] = {diff}")
        if jnp.fabs(diff) < tol:
            if verbose:
                print(f"Elbo diff tolerance reached at iteration {idx}")
            break

        elbo = elbo_res.elbo

    # compute PVE and reorder in descending value
    pve = compute_pve(params)
    params, pve = _reorder_factors_by_pve(A, params, pve)

    # compute PIPs
    pip = compute_pip(params)

    return SuSiEPCAResults_Design(params, elbo_res, pve, pip)
