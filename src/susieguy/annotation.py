from abc import abstractmethod
from dataclasses import field

import equinox as eqx
import jax
import jax.nn as nn
import optimistix as optx

from jaxtyping import Array

from .common import DataMatrix, ModelParams
from .utils import kl_discrete


def _compute_pi(A: DataMatrix, theta: Array) -> Array:
    return nn.softmax(A @ theta, axis=0).T


def _loss(theta: Array, args) -> Array:
    A, alpha = args
    pi = _compute_pi(A, theta)
    return kl_discrete(alpha, pi)


class PriorModel(eqx.Module):
    @abstractmethod
    def predict(self, params: ModelParams) -> Array:
        ...

    @abstractmethod
    def init_state(self, params: ModelParams) -> ModelParams:
        ...

    @abstractmethod
    def update(self, params: ModelParams) -> ModelParams:
        ...


class FixedPrior(PriorModel):
    def predict(self, params: ModelParams) -> Array:
        return params.pi

    def init_state(self, params: ModelParams) -> ModelParams:
        return params

    def update(self, params: ModelParams) -> ModelParams:
        return params


class AnnotationPriorModel(PriorModel):
    A: DataMatrix
    search: optx.AbstractMinimiser
    step = field(init=False)

    def __post_init__(self):
        self.step = eqx.filter_jit(eqx.Partial(self.search.step, _loss, options=None, tags=None))

    @property
    def shape(self):
        return self.A.shape

    def init_state(self, params: ModelParams) -> ModelParams:
        args = (self.A, params)
        f_struct, aux_struct = jax.eval_shape(_loss, params.theta, args)
        return params._replace(
            ann_state=self.search.init(
                _loss, params.theta, args, options=None, f_struct=f_struct, aux_struct=aux_struct, tags=None
            ),
        )

    def predict(self, params: ModelParams) -> Array:
        return _compute_pi(self.A, params.theta)

    def update(self, params: ModelParams) -> ModelParams:
        # take one step using optimistix optimizer
        theta, state, _ = self.step(params.theta, state=params.ann_state)
        return params._replace(
            theta=theta,
            pi=_compute_pi(self.A, theta),
            ann_state=state,
        )
