from typing import NamedTuple

import equinox as eqx

from jax import lax as lax, nn as nn, numpy as jnp
from jaxtyping import Array

from .common import DataMatrix, ModelParams
from .guide import GuideModel
from .utils import kl_discrete, logdet


class FactorParams(NamedTuple):
    mean_z: Array
    covar_z: Array


class FactorMoments(NamedTuple):
    mean_z: Array
    mean_zz: Array


class FactorModel(eqx.Module):
    n: int
    z_dim: int

    @property
    def shape(self):
        return self.n, self.z_dim

    def update(self, data: DataMatrix, guide: GuideModel, loadings: "LoadingModel", params: ModelParams) -> ModelParams:
        mean_w, mean_ww = loadings.moments(params)
        n_dim, z_dim = self.shape

        update_var_z = jnp.linalg.inv(params.tau * mean_ww + jnp.identity(z_dim))
        update_mu_z = (params.tau * (data @ mean_w.T) + guide.predict(params)) @ update_var_z

        return params._replace(
            mean_z=update_mu_z,
            var_z=update_var_z,
        )

    def moments(self, params: ModelParams) -> FactorMoments:
        n_dim, z_dim = self.shape

        # compute expected residuals
        # use posterior mean of Z, W, and Alpha to calculate residuals
        mean_z = params.mean_z
        mean_zz = mean_z.T @ mean_z + n_dim * params.var_z

        moments_ = FactorMoments(
            mean_z=mean_z,
            mean_zz=mean_zz,
        )
        return moments_

    def kl_divergence(self, guide: GuideModel, params: ModelParams) -> Array:
        n_dim, z_dim = self.shape
        mean_z, var_z = params.mean_z, params.var_z
        pred_z = guide.predict(params)
        # tr(mean_zz) = tr(mean_z' mean_z) + tr(n * var_z)
        #  = sum(mean_z ** 2) + n * tr(var_z)
        # NB: tr(E_q[Z]' M E_prior[Z]) = sum(E_q[Z] * (M E_prior[Z])); saves factor of n
        # guide.weighted_sumsq(params) = tr(M'E[BB']M); can change depending on guide model
        kl_d_ = 0.5 * (
            jnp.sum(mean_z**2)
            + n_dim * jnp.trace(var_z)
            - 2 * jnp.sum(mean_z * pred_z)
            + guide.weighted_sumsq(params)
            - n_dim * z_dim
            - n_dim * logdet(params.var_z)
        )
        return kl_d_


class LoadingMoments(NamedTuple):
    mean_w: Array
    mean_ww: Array


def _log_bf_np(z: Array, s2: Array, s0: Array) -> Array:
    s0_inv = 1.0 / s0
    s2ps0inv = s2 + s0_inv
    return 0.5 * (jnp.log(s2) - jnp.log(s2ps0inv) + z**2 * (s0_inv / s2ps0inv))


class _EffectLoopResults(NamedTuple):
    E_zzk: Array
    RtZk: Array
    Wk: Array
    k: int
    params: ModelParams


def _update_susie_effect(ldx: int, effect_params: _EffectLoopResults) -> _EffectLoopResults:
    E_zzk, RtZk, Wk, kdx, params = effect_params

    # remove current kl'th effect and update its expected residual
    Wkl = Wk - (params.mean_w[ldx, kdx] * params.alpha[ldx, kdx])
    E_RtZk = RtZk - E_zzk * Wkl

    # calculate update_var_w as the new V[w | gamma]
    # suppose indep between w_k
    update_var_wkl = jnp.reciprocal(params.tau * E_zzk + params.tau_0[ldx, kdx])

    # calculate update_mu_w as the new E[w | gamma]
    update_mean_wkl = params.tau * update_var_wkl * E_RtZk

    Z_s = (E_RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)
    s20_s = params.tau_0[ldx, kdx]
    log_bf = _log_bf_np(Z_s, s2_s, s20_s)
    # notice that pi is 1-D array for model without annotation
    # changed in the init_params
    log_alpha = jnp.log(params.pi[kdx, :]) + log_bf
    alpha_kl = nn.softmax(log_alpha)

    # update marginal w_kl
    Wk = Wkl + (update_mean_wkl * alpha_kl)
    params = params._replace(
        mean_w=params.mean_w.at[ldx, kdx].set(update_mean_wkl),
        var_w=params.var_w.at[ldx, kdx].set(update_var_wkl),
        alpha=params.alpha.at[ldx, kdx].set(alpha_kl),
    )

    return effect_params._replace(Wk=Wk, params=params)


class _FactorLoopResults(NamedTuple):
    X: DataMatrix
    W: Array
    EZZ: Array
    params: ModelParams


def _loop_factors(kdx: int, loop_params: _FactorLoopResults) -> _FactorLoopResults:
    data, W, mean_zz, params = loop_params
    l_dim, z_dim, p_dim = params.mean_w.shape

    # sufficient stats for inferring downstream w_kl/alpha_kl
    not_kdx = jnp.where(jnp.arange(z_dim) != kdx, size=z_dim - 1)
    E_zpzk = mean_zz[kdx][not_kdx]
    E_zzk = mean_zz[kdx, kdx]
    Wk = W[kdx, :]
    Wnk = W[not_kdx]
    RtZk = params.mean_z[:, kdx] @ data - Wnk.T @ E_zpzk

    # update over each of L effects
    init_loop_param = _EffectLoopResults(E_zzk, RtZk, Wk, kdx, params)
    _, _, Wk, _, params = lax.fori_loop(
        0,
        l_dim,
        _update_susie_effect,
        init_loop_param,
    )

    return loop_params._replace(W=W.at[kdx].set(Wk), params=params)


class LoadingModel(eqx.Module):
    p_dim: int
    z_dim: int
    l_dim: int

    @property
    def shape(self):
        return self.l_dim, self.z_dim, self.p_dim

    def update(self, data: DataMatrix, factors: FactorModel, params: ModelParams) -> ModelParams:
        l_dim, z_dim, p_dim = self.shape
        mean_z, mean_zz = factors.moments(params)
        mean_w, mean_ww = self.moments(params)

        # update locals (W, alpha)
        init_loop_param = _FactorLoopResults(data, mean_w, mean_zz, params)
        _, _, _, params = lax.fori_loop(0, z_dim, _loop_factors, init_loop_param)
        return params

    @staticmethod
    def update_hyperparam(params: ModelParams) -> ModelParams:
        est_varw = params.mean_w**2 + params.var_w[:, :, jnp.newaxis]

        u_tau_0 = jnp.sum(params.alpha, axis=-1) / jnp.sum(est_varw * params.alpha, axis=-1)

        return params._replace(tau_0=u_tau_0)

    def moments(self, params: ModelParams) -> LoadingMoments:
        trace_var = jnp.sum(
            params.var_w[:, :, jnp.newaxis] * params.alpha + (params.mean_w**2 * params.alpha * (1 - params.alpha)),
            axis=(-1, 0),
        )
        mu_w = jnp.sum(params.mean_w * params.alpha, axis=0)
        moments_ = LoadingMoments(
            mean_w=mu_w,
            mean_ww=mu_w @ mu_w.T + jnp.diag(trace_var),
        )

        return moments_

    @staticmethod
    def kl_divergence(params: ModelParams) -> Array:
        # technically this depends on the annotation model, but we usually flatten its predictions into the `pi`
        # member of `params`

        # awkward indexing to get broadcast working
        # KL for W variables
        klw_term1 = params.tau_0[:, :, jnp.newaxis] * (params.var_w[:, :, jnp.newaxis] + params.mean_w**2)
        klw_term2 = klw_term1 - 1.0 - (jnp.log(params.tau_0) + jnp.log(params.var_w))[:, :, jnp.newaxis]

        # weighted KL by E_q[gamma] variables
        kl_w_ = 0.5 * jnp.sum(params.alpha * klw_term2)

        # KL for gamma variables
        kl_gamma_ = kl_discrete(params.alpha, params.pi)

        return kl_w_ + kl_gamma_
