from typing import NamedTuple, Union

import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike


FloatOrArray = Union[float, ArrayLike]


class ELBOResults_Design(NamedTuple):

    """Define the class of all components in ELBO.

    Attributes:
        elbo: the value of ELBO
        E_ll: Expectation of log-likelihood
        negKL_z: -KL divergence of Z
        negKL_w: -KL divergence of W
        negKL_gamma: -KL divergence of Gamma
        negKL_beta: -KL divergence of B
        negKL_eta: -KL divergence of Eta

    """

    elbo: FloatOrArray
    E_ll: FloatOrArray
    negKL_z: FloatOrArray
    negKL_w: FloatOrArray
    negKL_gamma: FloatOrArray
    negKL_beta: FloatOrArray
    negKL_eta: FloatOrArray

    def __str__(self):
        return (
            f"ELBO = {self.elbo:.3f} | E[logl] = {self.E_ll:.3f} | "
            f"-KL[Z] = {self.negKL_z:.3f} | -KL[W] = {self.negKL_w:.3f} | "
            f"-KL[G] = {self.negKL_gamma:.3f} |"
            f"-KL[Beta] = {self.negKL_beta:.3f}| -KL[Eta] = {self.negKL_eta:.3f}"
        )


class ModelParams_Design(NamedTuple):
    """
    Define the class for variational parameters of all the variable we need
    to infer from the SuSiE PCA.

    Attributes:
        mu_z: mean parameter for factor Z
        var_z: variance parameter for factor Z
        mu_w: conditional mean parameter for loadings W
        var_w: conditional variance parameter for loading W
        alpha: parameter for the gamma that follows multinomial
                distribution
        tau: inverse variance parameter of observed data X
        tau_0: inverse variance parameter of single effect w_kl
        pi: prior probability for gamma
        beta: parameters for perturbation matrix

    """

    # variational params for Z
    mu_z: Array
    var_z: Array

    # variational params for W given Gamma
    mu_w: Array
    var_w: Array

    # variational params for Gamma
    alpha: Array

    # residual precision param
    tau: FloatOrArray
    tau_0: Array

    # prior probability for gamma
    theta: Array
    pi: Array

    # variational params of perturbation effects
    mu_beta: Array
    var_beta: Array

    # residual precision for beta
    tau_beta: Array

    # prior for Eta
    p: FloatOrArray
    # variational params for Eta
    p_hat: Array

    @property
    def W(self) -> Array:
        return jnp.sum(self.mu_w * self.alpha, axis=0)
    @property
    def B(self) -> Array:
        return self.mu_beta * self.p_hat.T


class SuSiEPCAResults_Design(NamedTuple):
    """Define the results object returned by function :py:obj:`susie_pca`.

    Attributes:
        params: the dictionary contain all the infered parameters
        elbo: the value of ELBO
        pve: the ndarray of percent of variance explained
        pip: the ndarray of posterior inclusion probabilities
        W: the posterior mean parameter for loadings

    """

    params: ModelParams_Design
    elbo: ELBOResults_Design
    pve: Array
    pip: Array

    @property
    def W(self) -> Array:
        return self.params.W


def compute_pip(params: ModelParams_Design) -> Array:
    """Compute the posterior inclusion probabilities (PIPs).

    Args:
        params: instance of inferred parameters

    Returns:
        Array: Array of posterior inclusion probabilities (PIPs) for each of
        `K x P` factor, feature combinations
    """

    pip = -jnp.expm1(jnp.sum(jnp.log1p(-params.alpha), axis=0))

    return pip


def compute_pve(params: ModelParams_Design) -> Array:
    """Compute the percent of variance explained (PVE).

    Args:
        params: instance of inferred parameters

    Returns:
        Array: Array of length `K` that contains percent of variance
        explained by each factor (PVE)
    """

    n_dim, z_dim = params.mu_z.shape
    W = params.W

    z_dim, p_dim = W.shape

    sk = jnp.zeros(z_dim)
    for k in range(z_dim):
        sk = sk.at[k].set(jnp.sum((params.mu_z[:, k, jnp.newaxis] * W[k, :]) ** 2))

    s = jnp.sum(sk)
    pve = sk / (s + p_dim * n_dim * (1 / params.tau))

    return pve
