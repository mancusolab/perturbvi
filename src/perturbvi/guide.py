from abc import abstractmethod
from dataclasses import field

from plum import dispatch

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.nn as nn
import jax.numpy as jnp
import lineax as lx

from jaxtyping import Array

from .common import ModelParams
from .sparse import SparseMatrix
from .utils import kl_discrete, outer_add


@dispatch
def _get_diag(G: Array) -> Array:
    return jnp.sum(G**2, axis=0)


@dispatch
def _get_diag(G: SparseMatrix) -> Array:
    return jsparse.sparsify(jnp.sum)(G.matrix**2, axis=0).todense()  # type: ignore


_multi_linear_solve = eqx.filter_vmap(lx.linear_solve, in_axes=(None, 1, None))


@dispatch
def _update_dense_beta(G: Array, params: ModelParams) -> ModelParams:
    # Create linear operator on G
    G_op = lx.MatrixLinearOperator(G)

    # Use lineax's CG solver
    solver = lx.NormalCG(rtol=1e-6, atol=1e-6)
    out = _multi_linear_solve(G_op, params.mean_z, solver)

    # Updated beta
    updated_beta = out.value.T

    return params._replace(mean_beta=updated_beta)


@dispatch
def _update_dense_beta(G: SparseMatrix, params: ModelParams) -> ModelParams:
    # Use lineax's CG solver
    solver = lx.NormalCG(rtol=1e-6, atol=1e-6)

    out = jax.vmap(lambda b: lx.linear_solve(G, b, solver), in_axes=1)(params.mean_z)

    # Updated beta
    updated_beta = out.value.T
    return params._replace(mean_beta=updated_beta)


class GuideModel(eqx.Module):
    guide_data: Array | SparseMatrix
    gsq_diag: Array = field(init=False)

    def __post_init__(self):
        self.gsq_diag = _get_diag(self.guide_data)  # type: ignore

    @property
    def shape(self):
        return self.guide_data.shape

    @abstractmethod
    def weighted_sumsq(self, params: ModelParams) -> Array:
        ...

    @abstractmethod
    def predict(self, params: ModelParams) -> Array:
        ...

    @abstractmethod
    def update(self, params: ModelParams) -> ModelParams:
        ...

    @staticmethod
    @abstractmethod
    def update_hyperparam(params: ModelParams) -> ModelParams:
        ...

    @staticmethod
    @abstractmethod
    def kl_divergence(params: ModelParams) -> Array:
        ...


class SparseGuideModel(GuideModel):
    def predict(self, params: ModelParams) -> Array:
        return self.guide_data @ (params.mean_beta * params.p_hat.T)

    def weighted_sumsq(self, params: ModelParams) -> Array:
        mean_bb = jnp.sum((params.mean_beta**2 + params.var_beta) * params.p_hat.T, axis=1)
        return jnp.sum(self.gsq_diag * mean_bb)

    def update(self, params: ModelParams) -> ModelParams:
        # compute E[Z'k]G: remove the g-th effect
        # however notice that only intercept in non-zero in GtG off-diag
        # ZkG = params.mean_z[:,kdx].T @ G - GtG_diag * params.B[-1,kdx]
        # without intercept
        ZkG = params.mean_z.T @ self.guide_data

        # if we don't need to add/subtract we can do it all in one go
        var_beta = jnp.reciprocal(outer_add(params.tau_beta, self.gsq_diag))
        mean_beta = ZkG * var_beta
        log_bf = (
            jnp.log(params.p)
            - jnp.log1p(params.p)
            + 0.5 * (jnp.log(var_beta) + jnp.log(params.tau_beta[:, jnp.newaxis]) + (mean_beta**2 / var_beta))
        )
        p_hat = nn.sigmoid(log_bf)
        return params._replace(
            mean_beta=mean_beta.T,
            var_beta=var_beta.T,
            p_hat=p_hat,
        )

    @staticmethod
    def update_hyperparam(params: ModelParams) -> ModelParams:
        est_var_beta = params.mean_beta**2 + params.var_beta
        u_tau_beta = jnp.sum(params.p_hat, axis=-1) / jnp.sum(est_var_beta * params.p_hat.T, axis=0)

        return params._replace(tau_beta=u_tau_beta)

    @staticmethod
    def kl_divergence(params: ModelParams) -> Array:
        # KL for beta
        # TODO: should be weighted by p_hat ?
        kl_beta_ = 0.5 * jnp.sum(
            (params.mean_beta**2 + params.var_beta) * params.tau_beta
            - 1
            - jnp.log(params.var_beta)
            - jnp.log(params.tau_beta)
        )
        # KL for eta selection variables
        kl_eta_ = kl_discrete(params.p_hat, params.p)
        return kl_beta_ + kl_eta_


class DenseGuideModel(GuideModel):
    def predict(self, params: ModelParams) -> Array:
        return self.guide_data @ params.mean_beta

    def weighted_sumsq(self, params: ModelParams) -> Array:
        pred_z = self.predict(params)
        return jnp.sum(pred_z**2)

    def update(self, params: ModelParams) -> ModelParams:
        return _update_dense_beta(self.guide_data, params)

    @staticmethod
    def update_hyperparam(params: ModelParams) -> ModelParams:
        return params

    @staticmethod
    def kl_divergence(params: ModelParams) -> Array:
        return jnp.asarray(0.0)
