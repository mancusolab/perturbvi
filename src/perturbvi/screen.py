from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from scipy import sparse as scipy_sparse

from jax.experimental import sparse as jax_sparse
from jaxtyping import ArrayLike

from ._defaults import (
    DEFAULT_INIT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_ITER,
    DEFAULT_P_PRIOR,
    DEFAULT_SEED,
    DEFAULT_STANDARDIZE,
    DEFAULT_TAU,
    DEFAULT_TOL,
    DEFAULT_VERBOSE,
)


if TYPE_CHECKING:
    from .infer import ELBOResults, InferResults, ModelParams


@dataclass(frozen=True)
class PerturbData:
    """Expression, perturbation design, labels, and optional covariates."""

    X: ArrayLike
    G: ArrayLike
    gene_names: Optional[Sequence[str]] = None
    perturbation_names: Optional[Sequence[str]] = None
    covariates: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        """Infer DataFrame labels and check row order when indexes are available."""
        row_index = None
        if isinstance(self.X, pd.DataFrame):
            row_index = self.X.index
            if self.gene_names is None:
                object.__setattr__(self, "gene_names", tuple(str(name) for name in self.X.columns))
        if isinstance(self.G, pd.DataFrame):
            if row_index is not None and not row_index.equals(self.G.index):
                raise ValueError("X and G DataFrames must have identical row indexes in the same order")
            row_index = self.G.index if row_index is None else row_index
            if self.perturbation_names is None:
                object.__setattr__(
                    self,
                    "perturbation_names",
                    tuple(str(name) for name in self.G.columns),
                )
        if (
            isinstance(self.covariates, pd.DataFrame)
            and row_index is not None
            and not row_index.equals(self.covariates.index)
        ):
            raise ValueError("covariates must have the same row index and order as X and G")
        if self.gene_names is not None:
            object.__setattr__(self, "gene_names", tuple(str(name) for name in self.gene_names))
        if self.perturbation_names is not None:
            object.__setattr__(
                self,
                "perturbation_names",
                tuple(str(name) for name in self.perturbation_names),
            )


@dataclass(frozen=True)
class FitResults:
    """Labeled result returned by :func:`fit_screen`."""

    inference: "InferResults"
    gene_names: tuple[str, ...]
    perturbation_names: tuple[str, ...]

    @property
    def params(self) -> "ModelParams":
        return self.inference.params

    @property
    def elbo(self) -> Optional["ELBOResults"]:
        return self.inference.elbo

    @property
    def pve(self):
        return self.inference.pve

    @property
    def pip(self):
        return self.inference.pip

    @property
    def W(self):
        return self.inference.W


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
        raise ValueError(f"{label} is required for the high-level screen API")
    values = list(names)
    if len(values) != expected:
        raise ValueError(f"{label} has {len(values)} entries but the corresponding matrix dimension is {expected}")
    if np.asarray(pd.isna(values), dtype=bool).any():
        raise ValueError(f"{label} contains missing values")
    labels = [str(value) for value in values]
    if any(not value.strip() for value in labels):
        raise ValueError(f"{label} contains empty names")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{label} contains duplicate names")


def validate_screen(screen: PerturbData) -> None:
    """Validate an in-memory high-level screen before preprocessing or fitting."""
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
        raise ValueError(f"X and G must have the same number of rows (cells); got X: {x_shape[0]}, G: {g_shape[0]}")

    if not np.all(np.isfinite(_stored_values(screen.X))):
        raise ValueError("X contains non-finite values (nan or inf)")
    guide_values = _stored_values(screen.G)
    if not np.all(np.isfinite(guide_values)):
        raise ValueError("G contains non-finite values (nan or inf)")
    if not np.isin(guide_values, [0.0, 1.0]).all():
        raise ValueError("G must contain only binary perturbation assignments")

    empty_cols = np.flatnonzero(_column_sumsq(screen.G, g_shape[1]) == 0)
    if empty_cols.size:
        raise ValueError(f"G has all-zero columns at indices {empty_cols.tolist()}")

    _validate_names(screen.gene_names, x_shape[1], "gene_names")
    _validate_names(screen.perturbation_names, g_shape[1], "perturbation_names")

    if screen.covariates is None:
        return
    if not isinstance(screen.covariates, pd.DataFrame):
        raise ValueError("covariates must be a pandas DataFrame so names and dtypes remain aligned")
    if screen.covariates.shape[0] != x_shape[0]:
        raise ValueError(f"covariates must have {x_shape[0]} rows (cells); got {screen.covariates.shape[0]}")
    if screen.covariates.shape[1] == 0:
        raise ValueError("covariates must contain at least one column")
    _validate_names(list(screen.covariates.columns), screen.covariates.shape[1], "covariate names")
    for name in screen.covariates:
        series = screen.covariates[name]
        if series.isna().any():
            raise ValueError(f"Covariate '{name}' contains missing values")
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
            if pd.api.types.is_complex_dtype(series.dtype):
                raise ValueError(f"Covariate '{name}' must not be complex-valued")
            values = series.to_numpy(dtype=np.float64)
        else:
            continue
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Covariate '{name}' contains non-finite values")


def fit_screen(
    screen: PerturbData,
    *,
    z_dim: int,
    l_dim: int,
    tau: float = DEFAULT_TAU,
    p_prior: float = DEFAULT_P_PRIOR,
    standardize: bool = DEFAULT_STANDARDIZE,
    init: Literal["random", "pca"] = DEFAULT_INIT,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = DEFAULT_SEED,
    A: Optional[Union[ArrayLike, jax_sparse.JAXSparse]] = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    verbose: bool = DEFAULT_VERBOSE,
) -> FitResults:
    """Fit a validated screen, applying any selected covariates once."""
    from .infer import infer

    screen_for_fit = screen
    if screen_for_fit.covariates is not None:
        from .preprocess import residualize_screen

        screen_for_fit = residualize_screen(screen_for_fit)
    else:
        validate_screen(screen_for_fit)

    inference = infer(
        screen_for_fit.X,
        screen_for_fit.G,
        z_dim=z_dim,
        l_dim=l_dim,
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

    return FitResults(
        inference=inference,
        gene_names=tuple(
            str(name) for name in (screen_for_fit.gene_names if screen_for_fit.gene_names is not None else ())
        ),
        perturbation_names=tuple(
            str(name)
            for name in (screen_for_fit.perturbation_names if screen_for_fit.perturbation_names is not None else ())
        ),
    )


__all__ = ["FitResults", "PerturbData", "fit_screen", "validate_screen"]
