from __future__ import annotations

# pattern: Functional Core
import logging
import math
import numbers

from typing import get_args, Literal, NamedTuple, Optional, Tuple

from plum import dispatch

import equinox as eqx
import optax
import optimistix as optx

from jax import nn, numpy as jnp, random
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike

from .annotation import AnnotationPriorModel, FixedPrior, PriorModel
from .common import (
    DataMatrix,
    ModelParams,
)
from .factorloadings import FactorModel, LoadingModel
from .guide import DenseGuideModel, GuideModel, SparseGuideModel
from .log import get_logger
from .sparse import CenteredSparseMatrix, sparse_column_variance, SparseMatrix
from .utils import prob_pca


log = get_logger("perturbvi")
log.setLevel(logging.INFO)

_init_type = Literal["pca", "random"]


def _normalize_sparse_matrix(matrix: sparse.JAXSparse, name: str) -> sparse.BCOO:
    """Normalize supported 2D JAX sparse layouts without materializing them."""
    if isinstance(matrix, sparse.BCOO):
        return matrix
    if isinstance(matrix, sparse.BCSR):
        row_counts = jnp.diff(matrix.indptr)
        rows = jnp.repeat(
            jnp.arange(matrix.shape[0], dtype=matrix.indices.dtype),
            row_counts,
            total_repeat_length=matrix.nse,
        )
        coordinates = jnp.stack((rows, matrix.indices), axis=1)
        return sparse.BCOO((matrix.data, coordinates), shape=matrix.shape)
    raise ValueError(f"{name} uses unsupported JAX sparse layout {type(matrix).__name__}; use BCOO or BCSR")


@dispatch
def _is_valid(X: ArrayLike):
    return jnp.all(jnp.isfinite(X))


@dispatch
def _is_valid(X: sparse.JAXSparse):
    return jnp.all(jnp.isfinite(X.data))


@dispatch
def _column_sumsq(X: ArrayLike):
    return jnp.sum(jnp.asarray(X) ** 2, axis=0)


@dispatch
def _column_sumsq(X: sparse.JAXSparse):
    return sparse.sparsify(jnp.sum)(X**2, axis=0).todense()  # type: ignore


@dispatch
def _column_variance(X: ArrayLike):
    return jnp.var(jnp.asarray(X), axis=0)


@dispatch
def _column_variance(X: sparse.JAXSparse):
    return sparse_column_variance(X, X.dtype)


def _update_tau(X: DataMatrix, factor: FactorModel, loadings: LoadingModel, params: ModelParams) -> ModelParams:
    n_dim, p_dim = X.shape

    # calculate moments of factors and loadings
    mean_z, mean_zz = factor.moments(params)
    mean_w, mean_ww = loadings.moments(params)

    # expectation of log likelihood
    # tr(A @ B) == sum(A * B)
    E_ss = params.x_ssq - 2 * jnp.trace(mean_w @ (X.T @ mean_z)) + jnp.sum(mean_zz * mean_ww)
    u_tau = (n_dim / E_ss) * p_dim

    return params._replace(tau=u_tau)


class ELBOResults(NamedTuple):
    """Define the class of all components in ELBO.

    **Arguments:**
        elbo: the value of ELBO
        expected_loglike: Expectation of log-likelihood
        kl_factors: -KL divergence of Z
        kl_loadings: -KL divergence of W
        kl_guide: -KL divergence of B

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


def compute_elbo(
    X: DataMatrix,
    guide: GuideModel,
    factors: FactorModel,
    loadings: LoadingModel,
    annotation: PriorModel,
    params: ModelParams,
) -> ELBOResults:
    """Create function to compute evidence lower bound (ELBO)

    **Arguments:**

    - `X` [`Array`]: The observed data, an N by P ndarray
    - `guide` [`GuideModel`]: The guide model
    - `factors` [`FactorModel`]: The factor model
    - `loadings` [`LoadingModel`]: The loading model
    - `annotation` [`PriorModel`]: The prior annotation model
    - `params` [`ModelParams`]: The dictionary contains all the inferred parameters

    **Returns:**
    - `ELBOResults` [`ELBOResults`]: The object contains all components in ELBO

    """
    n_dim, _ = params.mean_z.shape
    _, _, p_dim = params.mean_w.shape

    # calculate second moment of Z along k, (k x k) matrix
    # E[Z'Z] = V_k[Z] * tr(I_n) + E[Z]'E[Z] = V_k[Z] * n + E[Z]'E[Z]
    mean_z, mean_zz = factors.moments(params)

    # calculate moment of W
    mean_w, mean_ww = loadings.moments(params)

    # expectation of log likelihood
    # calculation tip: tr(A @ A.T) = tr(A.T @ A) = sum(A ** 2); but doubles mem
    # tr(A.T @ A) is inner product of A with itself = vdot(X, X)
    # (X.T @ E[Z] @ E[W]) is p x p (big!); compute (E[W] @ X.T @ E[Z]) (k x k)
    # jnp.vdot(X, X) is const wrt ELBO but costly to compute; could either compute once and store
    # or just ignore it
    exp_logl = (-0.5 * params.tau) * (
        params.x_ssq
        - 2 * jnp.sum(mean_w * (X.T @ mean_z).T)
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
    params = annotation.init_state(params)
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
        # sorted_pi = annotations.predict(ModelParams(theta=sorted_theta))  # type: ignore
        sorted_pi = annotations.predict(params._replace(theta=sorted_theta))
    else:
        sorted_theta = None
        sorted_pi = params.pi

    params = ModelParams(
        params.x_ssq,
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
    z_dim: int,
    l_dim: int,
    X: DataMatrix,
    guide: GuideModel,
    factors: FactorModel,
    loadings: LoadingModel,
    annotations: PriorModel,
    p_prior: float = 0.5,
    tau: float = 1.0,
    init: _init_type = "pca",
    verbose: bool = True,
) -> ModelParams:
    if verbose:
        log.info("Starting model parameter initialization...")

    # Base parameters
    n_dim, p_dim = X.shape
    tau_0 = jnp.ones((l_dim, z_dim))
    tau_0.block_until_ready()
    if verbose:
        log.info("✓ Base parameters initialized (5%)")

    # Random keys
    keys = random.split(rng_key, 10)
    keys[0].block_until_ready()
    rng_key, svd_key, mu_key, var_key, muw_key, varw_key, alpha_key, beta_key, var_beta_key, theta_key = keys
    if verbose:
        log.info("✓ Random keys setup (10%)")

    # Data statistics
    if isinstance(X, CenteredSparseMatrix):
        if X.squared_norm is None:
            raise ValueError("Sparse X squared norm is unavailable for this centering configuration")
        x_ssq = X.squared_norm
    else:
        x_ssq = jnp.sum(X * X)
    x_ssq.block_until_ready()
    if verbose:
        log.info("✓ Data statistics computed (15%)")

    # Factors
    if init == "pca":
        init_mu_z, _ = prob_pca(svd_key, X, k=z_dim)
    else:
        init_mu_z = random.normal(mu_key, shape=(n_dim, z_dim))
    init_mu_z.block_until_ready()
    if verbose:
        log.info("✓ Factors initialized (35%)")

    # Factor variance
    init_var_z = jnp.diag(random.normal(var_key, shape=(z_dim,)) ** 2)
    init_var_z.block_until_ready()
    if verbose:
        log.info("✓ Factor variance set (45%)")

    # Loadings
    init_mu_w = random.normal(muw_key, shape=(l_dim, z_dim, p_dim)) * 1e-3
    init_var_w = (1 / tau_0) * (random.normal(varw_key, shape=(l_dim, z_dim))) ** 2
    init_mu_w.block_until_ready()
    init_var_w.block_until_ready()
    if verbose:
        log.info("✓ Loadings initialized (60%)")

    # Alpha and pi
    # init_alpha = random.dirichlet(alpha_key, alpha=jnp.ones(p_dim), shape=(l_dim, z_dim))
    # init_alpha =jax.jit(random.dirichlet)(alpha_key, alpha=jnp.ones(p_dim), shape=(l_dim, z_dim))
    init_alpha = jnp.full((l_dim, z_dim, p_dim), 1.0 / p_dim)
    if verbose:
        log.info("Avoid dirichlet process")
    init_alpha.block_until_ready()
    if isinstance(annotations, AnnotationPriorModel):
        p_dim, m = annotations.shape
        theta = random.normal(theta_key, shape=(m, z_dim))
        pi = nn.softmax(annotations.A @ theta, axis=0).T
        theta.block_until_ready()
        pi.block_until_ready()
    else:
        theta = None
        pi = jnp.ones(shape=(z_dim, p_dim)) / p_dim
        pi.block_until_ready()
    if verbose:
        log.info("✓ Annotations setup complete (75%)")

    # Perturbation effects
    n_dim, g_dim = guide.shape
    tau_beta = jnp.ones((z_dim,))
    init_mu_beta = random.normal(beta_key, shape=(g_dim, z_dim)) * 1e-3
    init_var_beta = (1 / tau_beta) * random.normal(var_beta_key, shape=(g_dim, z_dim)) ** 2
    tau_beta.block_until_ready()
    init_mu_beta.block_until_ready()
    init_var_beta.block_until_ready()
    if verbose:
        log.info("✓ Perturbation effects initialized (90%)")

    # Priors
    if p_prior is not None:
        p_prior = p_prior * jnp.ones(g_dim)
        p_prior.block_until_ready()
    p_hat = 0.5 * jnp.ones(shape=(z_dim, g_dim))
    p_hat.block_until_ready()
    if verbose:
        log.info("✓ Priors setup complete (100%)")

    if verbose:
        log.info("✓ Model parameter initialization completed successfully")

    return ModelParams(
        x_ssq,
        init_mu_z,
        init_var_z,
        init_mu_w,
        init_var_w,
        init_alpha,
        jnp.array(tau, dtype=float),
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
    X: ArrayLike | sparse.JAXSparse,
    G: ArrayLike | sparse.JAXSparse,
    A: Optional[ArrayLike | sparse.JAXSparse],
    z_dim: int,
    l_dim: int,
    init: _init_type,
) -> Tuple[Array | sparse.JAXSparse, Array | sparse.JAXSparse, Array | sparse.JAXSparse | None]:
    # pull type options for init
    type_options = get_args(_init_type)

    if isinstance(z_dim, bool) or not isinstance(z_dim, numbers.Integral):
        raise ValueError(f"z_dim must be a positive integer; received {z_dim!r}")
    if isinstance(l_dim, bool) or not isinstance(l_dim, numbers.Integral):
        raise ValueError(f"l_dim must be a positive integer; received {l_dim!r}")
    if not isinstance(init, str) or init not in type_options:
        raise ValueError(f"Unknown initialization provided {init!r}; Choices: {type_options}")

    if isinstance(X, sparse.JAXSparse):
        X = _normalize_sparse_matrix(X, "X")
    else:
        try:
            X = jnp.asarray(X)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a dense array-like or JAX sparse matrix") from exc

    if X.ndim != 2:
        raise ValueError(f"Shape of X = {X.shape}; Expected 2-dim matrix")

    # should we check for n < p?
    n_dim, p_dim = X.shape
    if n_dim <= 0 or p_dim <= 0:
        raise ValueError(f"X should contain at least one row and column: received shape {X.shape}")

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

    if isinstance(G, sparse.JAXSparse):
        G = _normalize_sparse_matrix(G, "G")
    else:
        try:
            G = jnp.asarray(G)
        except (TypeError, ValueError) as exc:
            raise ValueError("G must be a dense array-like or JAX sparse matrix") from exc
    if G.ndim != 2:
        raise ValueError(f"Dimension of guide matrix G should be 2: received {len(G.shape)}")
    g_n_dim, g_dim = G.shape
    if g_n_dim != n_dim:
        raise ValueError(
            f"Leading dimension of guide matrix G should match sample dimension {n_dim}: received {g_n_dim}"
        )
    if g_dim <= 0:
        raise ValueError("Guide matrix G should contain at least one perturbation column")
    if not _is_valid(G):
        raise ValueError("G contains 'nan/inf'. Please check input data for correctness or missingness")
    zero_cols = jnp.where(_column_sumsq(G) == 0)[0]
    if zero_cols.size > 0:
        raise ValueError(f"G contains all-zero perturbation columns at indices {zero_cols.tolist()}")

    if A is not None:
        if isinstance(A, sparse.JAXSparse):
            A = _normalize_sparse_matrix(A, "A")
        else:
            try:
                A = jnp.asarray(A)
            except (TypeError, ValueError) as exc:
                raise ValueError("A must be a dense array-like or JAX sparse matrix") from exc
        if A.ndim != 2:
            raise ValueError(f"Dimension of annotation matrix A should be 2: received {len(A.shape)}")
        a_p_dim, _ = A.shape
        if a_p_dim != p_dim:
            raise ValueError(
                f"Leading dimension of annotation matrix A should match feature dimension {p_dim}: received {a_p_dim}"
            )
        if not _is_valid(A):
            raise ValueError("A contains 'nan/inf'. Please check input data for correctness or missingness")
        if A.shape[1] <= 0:
            raise ValueError("Annotation matrix A should contain at least one annotation column")

    return X, G, A  # type: ignore


def _check_control_args(
    *,
    p_prior: Optional[float],
    tau: float,
    standardize: bool,
    learning_rate: float,
    max_iter: int,
    tol: float,
    seed: int,
    verbose: bool,
) -> None:
    def finite_float(value, name: str) -> float:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a finite numeric scalar; received {value!r}")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite numeric scalar; received {value!r}") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite; received {value}")
        return result

    if p_prior is not None:
        p_prior_value = finite_float(p_prior, "p_prior")
        if not 0 <= p_prior_value <= 1:
            raise ValueError(f"p_prior must be between 0 and 1; received {p_prior}")
    tau_value = finite_float(tau, "tau")
    if tau_value <= 0:
        raise ValueError(f"tau must be finite and positive; received {tau}")
    if not isinstance(standardize, (bool, jnp.bool_)):
        raise ValueError(f"standardize must be boolean; received {standardize!r}")
    learning_rate_value = finite_float(learning_rate, "learning_rate")
    if learning_rate_value <= 0:
        raise ValueError(f"learning_rate must be finite and positive; received {learning_rate}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, numbers.Integral) or max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer; received {max_iter}")
    tol_value = finite_float(tol, "tol")
    if tol_value <= 0:
        raise ValueError(f"tol must be finite and positive; received {tol}")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f"seed must be an integer; received {seed!r}")
    if not isinstance(verbose, (bool, jnp.bool_)):
        raise ValueError(f"verbose must be boolean; received {verbose!r}")


class InferResults(NamedTuple):
    """Results returned by :func:`infer` and :func:`perturbvi.fit_screen`.

    Attributes:
        params: Inferred model parameters.
        elbo: Final ELBO components, or ``None`` for results reopened from disk.
        pve: Percent of variance explained for each latent factor.
        pip: Posterior inclusion probabilities, with shape factors by genes.
        W: Posterior mean loadings, exposed as a property of ``params``.
    """

    params: ModelParams
    elbo: Optional[ELBOResults]
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
    max_iter: int = 400,
    tol: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> InferResults:
    """Fit PerturbVI from preprocessed array inputs.

    Args:
        X: Cell-by-gene expression matrix. Dense array-like and JAX sparse
            matrices are supported.
        z_dim: Number of latent factors.
        l_dim: Number of single effects per factor.
        G: Cell-by-perturbation guide design matrix. The positional argument
            remains part of the stable low-level contract.
        A: Optional gene-by-annotation matrix for parameterized loading priors.
        p_prior: Prior inclusion probability for perturbation effects. ``None``
            or zero selects deterministic dense guide regression.
        tau: Positive initial residual precision.
        standardize: Center every expression column and scale it to unit
            population variance. Constant columns are rejected.
        init: Latent-factor initialization, either ``"pca"`` or ``"random"``.
        learning_rate: Positive optimizer learning rate used only with ``A``.
        max_iter: Positive maximum number of variational iterations.
        tol: Positive absolute ELBO convergence tolerance.
        seed: Integer JAX random seed.
        verbose: Log initialization, iteration, and convergence progress.

    Returns:
        Inferred parameters, ELBO, PVE, and PIP values.

    Raises:
        ValueError: If dimensions, values, guide columns, or scalar controls are
            invalid. Validation occurs before the iterative fit begins.
    """

    # Validate cheap scalar controls before copying or transferring matrices.
    _check_control_args(
        p_prior=p_prior,
        tau=tau,
        standardize=standardize,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        verbose=verbose,
    )
    X, G, A = _check_args(X, G, A, z_dim, l_dim, init)

    if standardize:
        variances = _column_variance(X)
        invalid_scale = jnp.where((~jnp.isfinite(variances)) | (variances <= 0))[0]
        if invalid_scale.size:
            raise ValueError(
                "Cannot standardize X because columns have zero or invalid variance at indices "
                f"{invalid_scale.tolist()}"
            )

    # cast to jax array
    if isinstance(X, Array):
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

    # initialize PRNGkey and params
    rng_key = random.PRNGKey(seed)
    factors = FactorModel()
    loadings = LoadingModel()
    params = _init_params(
        rng_key,
        z_dim,
        l_dim,
        X,
        guide,
        factors,
        loadings,
        annotation,
        p_prior,
        tau,
        init,
        verbose,
    )

    #  core loop for inference
    elbo = -5e25
    elbo_res = None
    for idx in range(1, max_iter + 1):
        elbo_res, params = _inner_loop(X, guide, factors, loadings, annotation, params)

        if verbose:
            log.info(f"Iter [{idx}] | {elbo_res}")

        diff = elbo_res.elbo - elbo
        if diff < 0 and verbose:
            log.info(f"Alert! Diff between elbo[{idx - 1}] and elbo[{idx}] = {diff}")
        if jnp.fabs(diff) < tol:
            if verbose:
                log.info(f"Elbo diff tolerance reached at iteration {idx}")
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

    **Arguments:**

    -`params` [`ModelParams`]: Instance of inferred parameters

    **Returns:**

    -`PIP` [`Array`]: Array of posterior inclusion probabilities (PIPs) for each of `K x P` factor,
              feature combinations

    """

    pip = -jnp.expm1(jnp.sum(jnp.log1p(-params.alpha), axis=0))

    return pip


def compute_pve(params: ModelParams) -> Array:
    """Compute the percent of variance explained (PVE).

    **Arguments:**

    -`params` [`ModelParams`]: Instance of inferred parameters

    **Returns:**

    -`PVE` [`Array`]: Array of length `K` that contains percent of variance
              explained by each factor (PVE)
    """

    n_dim = params.n_dim
    W = params.W
    z_dim, p_dim = W.shape

    sk = jnp.zeros(z_dim)
    for k in range(z_dim):
        sk = sk.at[k].set(jnp.sum((params.mean_z[:, k, jnp.newaxis] * W[k, :]) ** 2))

    s = jnp.sum(sk)
    pve = sk / (s + p_dim * n_dim * (1 / params.tau))

    return pve
