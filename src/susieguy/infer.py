from typing import get_args, Literal, NamedTuple, Optional, Tuple

from plum import dispatch

import equinox as eqx
import optax
import optimistix as optx

from jax import numpy as jnp, random
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike

from .annotation import AnnotationPriorModel, FixedPrior, PriorModel
from .common import (
    DataMatrix,
    ModelParams,
)
from .factorloadings import FactorModel, LoadingModel
from .guide import DenseGuideModel, GuideModel, SparseGuideModel
from .sparse import CenteredSparseMatrix, SparseMatrix
from .utils import prob_pca


_init_type = Literal["pca", "random"]


@dispatch
def _is_valid(X: ArrayLike):
    return jnp.all(jnp.isfinite(X))


@dispatch
def _is_valid(X: sparse.JAXSparse):
    return jnp.all(jnp.isfinite(X.data))


def _update_tau(X: DataMatrix, factor: FactorModel, loadings: LoadingModel, params: ModelParams) -> ModelParams:
    n_dim, z_dim = params.mean_z.shape
    l_dim, z_dim, p_dim = params.mean_w.shape

    # calculate moments of factors and loadings
    mean_z, mean_zz = factor.moments(params)
    mean_w, mean_ww = loadings.moments(params)

    # expectation of log likelihood
    # tr(A @ B) == sum(A * B)
    E_ss = jnp.sum(X * X) - 2 * jnp.trace(mean_w @ (X.T @ mean_z)) + jnp.sum(mean_zz * mean_ww)
    u_tau = (n_dim * p_dim) / E_ss

    return params._replace(tau=u_tau)


class ELBOResults(NamedTuple):

    """Define the class of all components in ELBO.

    Attributes:
        elbo: the value of ELBO
        expected_loglike: Expectation of log-likelihood
        kl_factors: -KL divergence of Z
        kl_loadings: -KL divergence of W
        negKL_gamma: -KL divergence of Gamma
        kl_guide: -KL divergence of B
        negKL_eta: -KL divergence of Eta

    """

    elbo: Array
    expected_loglike: Array
    kl_factors: Array
    kl_loadings: Array
    kl_guide: Array

    def __str__(self):
        return (
            f"ELBO = {self.elbo:.3f} | E[logl] = {self.expected_loglike:.3f} | "
            f"KL[Z] = {self.kl_factors:.3f} | E_Q[KL[W]] + KL[Gamma] = {self.kl_loadings:.3f} | "
            f"E_Q[KL[Beta]] + KL[Eta] = {self.kl_guide:.3f}|"
        )


@eqx.filter_jit
def compute_elbo(
    X: DataMatrix,
    guide: GuideModel,
    factors: FactorModel,
    loadings: LoadingModel,
    annotation: PriorModel,
    params: ModelParams,
) -> ELBOResults:
    """Create function to compute evidence lower bound (ELBO)

    Args:
        X: the observed data, an N by P ndarray
        G: secondary data.
        GtG_diag: diagonal element of GTG.
        params: the dictionary contains all the infered parameters

    Returns:
        ELBOResult_Design: the object contains all components in ELBO

    """
    n_dim, z_dim = params.mean_z.shape
    l_dim, z_dim, p_dim = params.mean_w.shape

    # calculate second moment of Z along k, (k x k) matrix
    # E[Z'Z] = V_k[Z] * tr(I_n) + E[Z]'E[Z] = V_k[Z] * n + E[Z]'E[Z]
    mean_z, mean_zz = factors.moments(params)

    # calculate moment of W
    mean_w, mean_ww = loadings.moments(params)

    # expectation of log likelihood
    # calculation tip: tr(A @ A.T) = tr(A.T @ A) = sum(A ** 2)
    # (X.T @ E[Z] @ E[W]) is p x p (big!); compute (E[W] @ X.T @ E[Z]) (k x k)
    exp_logl = (-0.5 * params.tau) * (
        jnp.sum(X**2)
        - 2 * jnp.einsum("kp,np,nk->", mean_w, X, mean_z)  # tr(E[W] @ X.T @ E[Z])
        + jnp.einsum("ij,ji->", mean_zz, mean_ww)  # tr(E[Z.T @ Z] @ E[W @ W.T])
    ) + 0.5 * n_dim * p_dim * jnp.log(params.tau)

    # neg-KL for Z
    kl_factors = factors.kl_divergence(guide, params)

    # neg-KL for w
    kl_loadings = loadings.kl_divergence(params)

    # neg-KL for beta
    kl_guide = guide.kl_divergence(params)

    elbo = exp_logl - (kl_factors + kl_loadings + kl_guide)

    result = ELBOResults(elbo, exp_logl, kl_factors, kl_loadings, kl_guide)

    return result


@eqx.filter_jit
def _inner_loop(
    X: DataMatrix,
    guide: GuideModel,
    factors: FactorModel,
    loadings: LoadingModel,
    annotation: PriorModel,
    params: ModelParams,
):
    # update annotation priors if any
    params = annotation.update(params)

    # update loadings prior precision via ~Empirical Bayes and update variational params
    params = loadings.update_hyperparam(params)
    params = loadings.update(X, factors, params)

    # update factor parameters
    params = factors.update(X, guide, loadings, params)

    # update beta and p_hat
    params = guide.update_hyperparam(params)
    params = guide.update(params)

    # update precision parameters via MLE
    params = _update_tau(X, factors, loadings, params)

    # compute elbo
    elbo_res = compute_elbo(X, guide, factors, loadings, annotation, params)

    return elbo_res, params


def _reorder_factors_by_pve(pve: Array, annotations: PriorModel, params: ModelParams) -> Tuple[Array, ModelParams]:
    sorted_indices = jnp.argsort(pve)[::-1]
    pve = pve[sorted_indices]

    sorted_mu_z = params.mean_z[:, sorted_indices]
    sorted_var_z = params.var_z[sorted_indices, sorted_indices]
    sorted_mu_beta = params.mean_beta[:, sorted_indices]
    sorted_var_beta = params.var_beta[:, sorted_indices]
    sorted_p_hat = params.p_hat[sorted_indices, :]
    sorted_mu_w = params.mean_w[:, sorted_indices, :]
    sorted_var_w = params.var_w[:, sorted_indices]
    sorted_tau_beta = params.tau_beta[sorted_indices]
    sorted_alpha = params.alpha[:, sorted_indices, :]
    sorted_tau_0 = params.tau_0[:, sorted_indices]
    if isinstance(annotations, AnnotationPriorModel):
        sorted_theta = params.theta[:, sorted_indices]
        sorted_pi = annotations.predict(ModelParams(theta=sorted_theta))  # type: ignore
    else:
        sorted_theta = None
        sorted_pi = params.pi

    params = ModelParams(
        sorted_mu_z,
        sorted_var_z,
        sorted_mu_w,
        sorted_var_w,
        sorted_alpha,
        params.tau,
        sorted_tau_0,
        sorted_theta,
        sorted_pi,
        None,
        sorted_mu_beta,
        sorted_var_beta,
        sorted_tau_beta,
        params.p,
        sorted_p_hat,
    )

    return pve, params


def _init_params(
    rng_key: random.PRNGKey,
    X: DataMatrix,
    guide: GuideModel,
    factors: FactorModel,
    loadings: LoadingModel,
    annotations: PriorModel,
    p_prior: float = 0.5,
    tau: float = 1.0,
    init: _init_type = "pca",
) -> ModelParams:
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
        ModelParams: initialized set of model parameters.

    Raises:
        ValueError: Invalid initialization scheme.
    """
    l_dim, z_dim, p_dim = loadings.shape
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
        print("Initialize factors using probabilistic PCA")
        init_mu_z, _ = prob_pca(svd_key, X, k=z_dim)
        print("Factor initialization finished.")
    elif init == "random":
        # random initialization
        init_mu_z = random.normal(mu_key, shape=(n_dim, z_dim))
        print("Factor initialization finished.")
    else:
        raise ValueError(f"Unknown initialization provided '{init}'; Choices: {type_options}")

    init_var_z = jnp.diag(random.normal(var_key, shape=(z_dim,)) ** 2)

    # each w_kl has a specific variance term
    init_mu_w = random.normal(muw_key, shape=(l_dim, z_dim, p_dim)) * 1e-3
    init_var_w = (1 / tau_0) * (random.normal(varw_key, shape=(l_dim, z_dim))) ** 2

    init_alpha = random.dirichlet(alpha_key, alpha=jnp.ones(p_dim), shape=(l_dim, z_dim))
    if isinstance(annotations, AnnotationPriorModel):
        p_dim, m = annotations.shape
        theta = random.normal(theta_key, shape=(m, z_dim))
        pi = annotations.predict(ModelParams(theta=theta))  # type: ignore
    else:
        theta = None
        pi = jnp.ones(shape=(z_dim, p_dim)) / p_dim

    # Initialization for perturbation effects
    n_dim, g_dim = guide.shape
    # Notice: shape maybe (z_dim,): same for all feature in each component
    tau_beta = jnp.ones((z_dim,))
    init_mu_beta = random.normal(beta_key, shape=(g_dim, z_dim)) * 1e-3
    init_var_beta = (1 / tau_beta) * random.normal(var_beta_key, shape=(g_dim, z_dim)) ** 2

    # uniform prior for eta
    if p_prior is not None:
        p_prior = p_prior * jnp.ones(g_dim)
    else:
        p_prior = None

    # Initialization of variational params for eta
    # p_hat is in shape of (z_dim,g_dim)
    p_hat = 0.5 * jnp.ones(shape=(z_dim, g_dim))
    print("Model paramterters initialization finished.")

    return ModelParams(
        init_mu_z,
        init_var_z,
        init_mu_w,
        init_var_w,
        init_alpha,
        tau,
        tau_0,
        theta=theta,
        pi=pi,
        ann_state=None,
        mean_beta=init_mu_beta,
        var_beta=init_var_beta,
        tau_beta=tau_beta,
        p=p_prior,
        p_hat=p_hat,
    )


def _check_args(
    X: ArrayLike | sparse.JAXSparse, A: Optional[ArrayLike | sparse.JAXSparse], z_dim: int, l_dim: int, init: _init_type
) -> Tuple[Array | sparse.JAXSparse, Array | sparse.JAXSparse]:
    # pull type options for init
    type_options = get_args(_init_type)

    if isinstance(X, ArrayLike):
        X = jnp.asarray(X)

    if X.ndim != 2:
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
        if isinstance(A, ArrayLike):
            A = jnp.asarray(A)
        if A.ndim != 2:
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

    return X, A  # type: ignore


class InferResults(NamedTuple):
    """Define the results object returned by function :py:obj:`susie_pca`.

    Attributes:
        params: the dictionary contain all the infered parameters
        elbo: the value of ELBO
        pve: the ndarray of percent of variance explained
        pip: the ndarray of posterior inclusion probabilities
        W: the posterior mean parameter for loadings

    """

    params: ModelParams
    elbo: ELBOResults
    pve: Array
    pip: Array

    @property
    def W(self) -> Array:
        return self.params.W


def infer(
    X: ArrayLike | sparse.JAXSparse,
    z_dim: int,
    l_dim: int,
    G: ArrayLike | sparse.JAXSparse,
    A: Optional[ArrayLike | sparse.JAXSparse] = None,
    p_prior: Optional[float] = 0.5,
    tau: float = 1.0,
    standardize: bool = False,
    init: _init_type = "pca",
    learning_rate: float = 1e-2,
    max_iter: int = 200,
    tol: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> InferResults:
    """The main inference function for SuSiE PCA.

    Args:
        X: Input data. Should be an array-like
        z_dim: Latent factor dimension (int; K)
        l_dim: Number of single-effects comprising each factor (int; L)
        G: Secondary information. Should be an array-like or sparse JAX matrix.
        A: Annotation matrix to use in parameterized-prior mode. If not `None`, leading dimension
            should match the feature dimension of X.
        p_prior: Prior probability for each perturbation to have a non-zero effect to predict latent factor.
            Set to `None` to use a dense, non-sparse model (i.e., OLS).
        tau: initial value of residual precision (default = 1)
        standardize: Whether to center and scale the input data with mean 0
            and variance 1 (default = False)
        init: How to initialize the variational mean parameters for latent factors.
            Either "pca" or "random" (default = "pca")
        learning_rate: Learning rate for prior annotation probability inference. Not used if `A` is `None`.
        max_iter: Maximum number of iterations for inference (int)
        tol: Numerical tolerance for ELBO convergence (float)
        seed: Seed for "random" initialization (int)
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
    X, A = _check_args(X, A, z_dim, l_dim, init)

    # cast to jax array
    if isinstance(X, Array):
        # option to center the data
        X -= jnp.mean(X, axis=0)
        if standardize:
            X /= jnp.std(X, axis=0)
    elif isinstance(X, sparse.JAXSparse):
        X = CenteredSparseMatrix(X, scale=standardize)  # type: ignore
    if isinstance(G, ArrayLike):
        G = jnp.asarray(G)
    elif isinstance(G, sparse.JAXSparse):
        G = SparseMatrix(G)

    if p_prior is None or jnp.isclose(p_prior, 0.0):
        guide = DenseGuideModel(G)
    else:
        guide = SparseGuideModel(G)

    if A is not None:
        adam = optax.adam(learning_rate)
        annotation = AnnotationPriorModel(A, optx.OptaxMinimiser(adam, rtol=1e-3, atol=1e-3))
    else:
        annotation = FixedPrior()

    n, p = X.shape  # type: ignore

    factors = FactorModel(n, z_dim)
    loadings = LoadingModel(p, z_dim, l_dim)
    # initialize PRNGkey and params
    rng_key = random.PRNGKey(seed)
    params = _init_params(rng_key, X, guide, factors, loadings, annotation, p_prior, tau, init)

    #  core loop for inference
    elbo = -5e25
    elbo_res = None
    for idx in range(1, max_iter + 1):
        elbo_res, params = _inner_loop(X, guide, factors, loadings, annotation, params)

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
    pve, params = _reorder_factors_by_pve(pve, annotation, params)

    # compute PIPs
    pip = compute_pip(params)

    return InferResults(params, elbo_res, pve, pip)


def compute_pip(params: ModelParams) -> Array:
    """Compute the posterior inclusion probabilities (PIPs).

    Args:
        params: instance of inferred parameters

    Returns:
        Array: Array of posterior inclusion probabilities (PIPs) for each of
        `K x P` factor, feature combinations
    """

    pip = -jnp.expm1(jnp.sum(jnp.log1p(-params.alpha), axis=0))

    return pip


def compute_pve(params: ModelParams) -> Array:
    """Compute the percent of variance explained (PVE).

    Args:
        params: instance of inferred parameters

    Returns:
        Array: Array of length `K` that contains percent of variance
        explained by each factor (PVE)
    """

    n_dim, z_dim = params.mean_z.shape
    W = params.W

    z_dim, p_dim = W.shape

    sk = jnp.zeros(z_dim)
    for k in range(z_dim):
        sk = sk.at[k].set(jnp.sum((params.mean_z[:, k, jnp.newaxis] * W[k, :]) ** 2))

    s = jnp.sum(sk)
    pve = sk / (s + p_dim * n_dim * (1 / params.tau))

    return pve
