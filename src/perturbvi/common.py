from typing import Any, NamedTuple, Union

import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from .sparse import CenteredSparseMatrix, SparseMatrix


DataMatrix = Array | SparseMatrix | CenteredSparseMatrix
FloatOrArray = Union[float, ArrayLike]

class ELBOResults(NamedTuple):

    """Define the class of all components in ELBO.

    Attributes:
        elbo: the value of ELBO
        E_ll: Expectation of log-likelihood
        negKL_z: -KL divergence of Z
        negKL_w: -KL divergence of W
        negKL_gamma: -KL divergence of gamma

    """

    elbo: FloatOrArray
    E_ll: FloatOrArray
    negKL_z: FloatOrArray
    negKL_w: FloatOrArray
    negKL_gamma: FloatOrArray

    def __str__(self):
        return (
            f"ELBO = {self.elbo:.3f} | E[logl] = {self.E_ll:.3f} | "
            f"-KL[Z] = {self.negKL_z:.3f} | -KL[W] = {self.negKL_w:.3f} | "
            f"-KL[G] = {self.negKL_gamma:.3f}"
        )


class ModelParams(NamedTuple):
    """
    Define the class for variational parameters of all the variable we need
    to infer from the perturbVI.

    Attributes:
        mean_z: mean parameter for factor Z
        var_z: variance parameter for factor Z
        mean_w: conditional mean parameter for loadings W
        var_w: conditional variance parameter for loading W
        alpha: parameter for the gamma that follows multinomial
                distribution
        tau: inverse variance parameter of observed data X
        tau_0: inverse variance parameter of single effect w_kl
        theta: parameter for annotation prior
        pi: prior probability for gamma
        ann_state: internal state for learning theta
        mean_beta: mean parameters for perturbation effect matrix
        var_beta: variance parameters for perturbation effect matrix
        tau_beta: inverse variance parameters for perturbation effect matrix
        p: prior for Eta from spike-and-slab prior
        p_hat: variational parameters for Eta
    """

    # ssq
    x_ssq: Array

    # variational params for Z
    mean_z: Array
    var_z: Array

    # variational params for W given Gamma
    mean_w: Array
    var_w: Array

    @property
    def n_dim(self):
        return self.mean_z.shape[0]

    @property
    def z_dim(self):
        return self.mean_z.shape[1]

    @property
    def p_dim(self):
        return self.mean_w.shape[2]

    # variational params for Gamma
    alpha: Array

    # residual precision param
    tau: FloatOrArray
    tau_0: Array

    # prior probability for gamma
    theta: Array
    pi: Array
    # internal state for learning theta
    ann_state: Any

    # variational params of perturbation effects
    mean_beta: Array
    var_beta: Array

    # residual precision for beta
    tau_beta: Array

    # prior for Eta
    p: Array
    # variational params for Eta
    p_hat: Array

    @property
    def W(self) -> Array:
        return jnp.sum(self.mean_w * self.alpha, axis=0)
