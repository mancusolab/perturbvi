from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jaxtyping import Array

from .common import DataMatrix, ModelParams
from .guide import GuideModel
from .loading import LoadingModel
from .utils import logdet


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

    def update(self, data: DataMatrix, guide: GuideModel, loadings: LoadingModel, params: ModelParams) -> ModelParams:
        mean_w, mean_ww = loadings.moments(params)
        n_dim, z_dim = self.shape

        update_var_z = jnp.linalg.inv(params.tau * mean_ww + jnp.identity(z_dim))
        update_mu_z = (params.tau * (data @ mean_w.T) + guide.predict(params)) @ update_var_z
        z_params = FactorParams(
            mean_z=update_mu_z,
            covar_z=update_var_z,
        )
        return params._replace(z_params=z_params)

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
        #  = sum(mean_z ** 2) + n * k * tr(var_z)
        # NB: tr(E_q[Z]' M E_prior[Z]) = sum(E_q[Z] * (M E_prior[Z])); saves factor of n
        # guide.weighted_sumsq(params) = tr(M'E[BB']M); can change depending on guide model
        kl_d_ = 0.5 * (
            jnp.sum(mean_z**2)
            + n_dim * z_dim * jnp.trace(var_z)
            - 2 * jnp.sum(mean_z * pred_z)
            + guide.weighted_sumsq(params)
            - n_dim * z_dim
            - n_dim * logdet(params.var_z)
        )
        return kl_d_
