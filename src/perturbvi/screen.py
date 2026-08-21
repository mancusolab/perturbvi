from __future__ import annotations

from collections import namedtuple
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from scipy import sparse as scipy_sparse

from jax.experimental import sparse as jax_sparse
from jaxtyping import ArrayLike


if TYPE_CHECKING:
    from .infer import InferResults


_ScreenDataTuple = namedtuple(
    "_ScreenDataTuple",
    [
        "X",
        "G",
        "gene_names",
        "perturbation_names",
        "cell_names",
        "source",
        "covariates",
        "covariate_names",
    ],
)


class ScreenData(_ScreenDataTuple):
    """Validated in-memory inputs for a PerturbVI fit.

    The container uses canonical ``gene_names`` and ``perturbation_names``
    fields, never retains a raw AnnData object, and keeps tuple behavior such
    as ``_replace``.
    """

    X: ArrayLike
    G: ArrayLike
    gene_names: Optional[Sequence[str]]
    perturbation_names: Optional[Sequence[str]]
    cell_names: Optional[Sequence[str]]
    source: Mapping[str, Any]
    covariates: Optional[ArrayLike]
    covariate_names: Optional[Sequence[str]]
    __slots__ = ()

    def __new__(
        cls,
        X: ArrayLike,
        G: ArrayLike,
        gene_names: Optional[Sequence[str]] = None,
        perturbation_names: Optional[Sequence[str]] = None,
        cell_names: Optional[Sequence[str]] = None,
        source: Optional[Mapping[str, Any]] = None,
        covariates: Optional[ArrayLike] = None,
        covariate_names: Optional[Sequence[str]] = None,
    ) -> "ScreenData":
        if source is None:
            source = {}
        return super().__new__(
            cls,
            X,
            G,
            gene_names,
            perturbation_names,
            cell_names,
            source,
            covariates,
            covariate_names,
        )


def _shape(value, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has an invalid shape: {shape}") from exc


def _stored_values(value) -> np.ndarray:
    if isinstance(value, jax_sparse.JAXSparse) or scipy_sparse.issparse(value):
        return np.asarray(value.data)
    return np.asarray(value)


def _column_sumsq(value, n_columns: int) -> np.ndarray:
    if isinstance(value, jax_sparse.JAXSparse):
        indices = np.asarray(value.indices)
        data = np.asarray(value.data).reshape(-1)
        columns = indices.reshape(-1) if indices.ndim == 1 else indices[..., 1].reshape(-1)
        sums = np.zeros(n_columns, dtype=np.float64)
        valid = (columns >= 0) & (columns < n_columns)
        np.add.at(sums, columns[valid], np.abs(data[valid]) ** 2)
        return sums
    if scipy_sparse.issparse(value):
        return np.asarray(value.multiply(value.conjugate()).sum(axis=0)).ravel()
    array = np.asarray(value)
    return np.sum(np.abs(array) ** 2, axis=0)


def _validate_names(names: Optional[Sequence[str]], expected: int, label: str) -> None:
    if names is None:
        return
    values = list(names)
    if len(values) != expected:
        raise ValueError(f"{label} has {len(values)} entries but the corresponding matrix dimension is {expected}")
    if np.asarray(pd.isna(values), dtype=bool).any():
        raise ValueError(f"{label} contains missing values")
    as_strings = [str(value) for value in values]
    if len(set(as_strings)) != len(as_strings):
        raise ValueError(f"{label} contains duplicate names")


def matrix_to_numpy(value, *, dtype=None) -> np.ndarray:
    """Materialize a supported matrix, used only by explicitly dense operations."""
    if isinstance(value, jax_sparse.JAXSparse):
        value = value.todense()
    elif scipy_sparse.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=dtype)


def validate_screen(screen: ScreenData) -> None:
    """Validate a screen before any expensive inference or preprocessing."""
    if not isinstance(screen.source, Mapping):
        raise ValueError("source must be a mapping")
    x_shape = _shape(screen.X, "X")
    g_shape = _shape(screen.G, "G")

    if len(x_shape) != 2:
        raise ValueError(f"X must be 2D; got shape {x_shape}")
    if len(g_shape) != 2:
        raise ValueError(f"G must be 2D; got shape {g_shape}")
    if x_shape[0] == 0 or x_shape[1] == 0:
        raise ValueError(f"X must contain at least one cell and one gene; got shape {x_shape}")
    if g_shape[1] == 0:
        raise ValueError("G must contain at least one perturbation column")
    if x_shape[0] != g_shape[0]:
        raise ValueError(
            f"X and G must have the same number of rows (cells); got X: {x_shape[0]}, G: {g_shape[0]}"
        )

    if not np.all(np.isfinite(_stored_values(screen.X))):
        raise ValueError("X contains non-finite values (nan or inf)")
    if not np.all(np.isfinite(_stored_values(screen.G))):
        raise ValueError("G contains non-finite values (nan or inf)")

    empty_cols = np.flatnonzero(_column_sumsq(screen.G, g_shape[1]) == 0)
    if empty_cols.size:
        raise ValueError(f"G has all-zero columns at indices {empty_cols.tolist()}")

    _validate_names(screen.gene_names, x_shape[1], "gene_names")
    _validate_names(screen.perturbation_names, g_shape[1], "perturbation_names")
    _validate_names(screen.cell_names, x_shape[0], "cell_names")

    if screen.covariates is None:
        if screen.covariate_names is not None:
            raise ValueError("covariate_names requires covariates")
        return

    covariates = np.asarray(screen.covariates)
    if covariates.ndim != 2:
        raise ValueError(f"covariates must be 2D; got shape {covariates.shape}")
    if covariates.shape[0] != x_shape[0]:
        raise ValueError(f"covariates must have {x_shape[0]} rows (cells); got {covariates.shape[0]}")
    if covariates.shape[1] == 0:
        raise ValueError("covariates must contain at least one column")
    if np.asarray(pd.isna(covariates), dtype=bool).any():
        raise ValueError("covariates contains missing values")
    if np.issubdtype(covariates.dtype, np.number) and not np.all(np.isfinite(covariates)):
        raise ValueError("covariates contains non-finite values")
    _validate_names(screen.covariate_names, covariates.shape[1], "covariate_names")


def fit_screen(
    screen: ScreenData,
    *,
    z_dim: int,
    l_dim: int,
    tau: float,
    p_prior: float = 0.1,
    standardize: bool = True,
    init: str = "random",
    tol: float = 1e-2,
    max_iter: int = 500,
    seed: int = 0,
    A: Optional[Union[ArrayLike, jax_sparse.JAXSparse]] = None,
    learning_rate: float = 1e-2,
    verbose: bool = False,
) -> "InferResults":
    """Fit a validated screen and return :class:`InferResults`.

    Optional residualization must be performed before calling this function.
    Expression features are always centered before fitting. When
    ``standardize=True``, centered features are also divided by their
    population standard deviation.

    Args:
        screen: Validated expression, perturbation design, and metadata.
        z_dim: Number of latent factors.
        l_dim: Number of single effects per factor.
        tau: Positive initial residual precision.
        p_prior: Prior inclusion probability for perturbation effects.
        standardize: Scale centered expression features to unit population
            variance. Centering is always applied.
        init: Latent-factor initialization, ``"random"`` or ``"pca"``.
        tol: Positive absolute convergence tolerance.
        max_iter: Positive maximum number of inference iterations.
        seed: Random seed.
        A: Optional gene-by-annotation matrix for annotation-informed loading
            priors. Its rows must follow the gene order in ``screen.X``.
        learning_rate: Positive optimizer learning rate used only when ``A``
            is provided.
        verbose: Report inference progress.

    Returns:
        Fitted parameters, ELBO, PVE, and PIP values.
    """
    from .infer import infer

    validate_screen(screen)
    return infer(
        screen.X,
        z_dim=z_dim,
        l_dim=l_dim,
        G=screen.G,
        A=A,
        tau=tau,
        p_prior=p_prior,
        standardize=standardize,
        init=init,
        learning_rate=learning_rate,
        tol=tol,
        max_iter=max_iter,
        seed=seed,
        verbose=verbose,
    )
