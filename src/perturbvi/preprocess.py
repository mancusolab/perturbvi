from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scipy import sparse as scipy_sparse
from scipy.linalg import qr as scipy_qr

from jax.experimental import sparse as jax_sparse

from ._defaults import DEFAULT_BLOCK_SIZE
from .log import get_logger
from .screen import PerturbData, validate_screen


log = get_logger(__name__)


def _build_design_matrix_with_names(covariates: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build an intercept plus centered numeric and categorical columns."""
    if not isinstance(covariates, pd.DataFrame):
        raise ValueError("covariates must be a pandas DataFrame")
    if covariates.empty:
        raise ValueError("covariates must contain rows and columns")

    columns = [np.ones((len(covariates), 1), dtype=np.float64)]
    names = ["intercept"]
    for name in covariates:
        series = covariates[name]
        if series.isna().any():
            raise ValueError(f"Covariate '{name}' contains missing values")
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            levels = list(pd.unique(series))
            for level in levels[1:]:
                columns.append((series.to_numpy() == level).astype(np.float64).reshape(-1, 1))
                names.append(f"{name}={level}")
            continue

        numeric = series.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"Covariate '{name}' contains non-finite values")
        centered = numeric - numeric.mean()
        if np.any(centered):
            columns.append(centered.reshape(-1, 1))
            names.append(str(name))

    design = np.hstack(columns)
    if not np.all(np.isfinite(design)):
        raise ValueError("Design matrix contains non-finite values after encoding")
    return design, names


def _projection_basis(C: np.ndarray, names: list[str]) -> np.ndarray:
    """Return a float32 orthonormal basis using a float64 rank-aware SVD."""
    scaled = C.astype(np.float64, copy=True)
    if scaled.shape[1] > 1:
        scales = np.std(scaled[:, 1:], axis=0)
        scales[scales == 0] = 1.0
        scaled[:, 1:] /= scales

    U, singular_values, _ = np.linalg.svd(scaled, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(scaled.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < len(names):
        _, _, pivot = scipy_qr(scaled, mode="economic", pivoting=True)
        log.warning(
            "Covariate design is rank-deficient; dropped design directions: %s",
            ", ".join(names[index] for index in pivot[rank:]),
        )
    return U[:, :rank].astype(np.float32)


def _as_csc(value):
    """Convert sparse input once so column-block reads are efficient."""
    if scipy_sparse.issparse(value):
        return value.astype(np.float32, copy=False).tocsc()
    if isinstance(value, jax_sparse.BCSR):
        indptr = np.asarray(value.indptr)
        indices = np.column_stack(
            [
                np.repeat(np.arange(value.shape[0]), np.diff(indptr)),
                np.asarray(value.indices).reshape(-1),
            ]
        )
        data = np.asarray(value.data).reshape(-1).astype(np.float32, copy=False)
    elif isinstance(value, jax_sparse.BCOO):
        indices = np.asarray(value.indices)
        data = np.asarray(value.data).reshape(-1).astype(np.float32, copy=False)
    else:
        return None
    return scipy_sparse.csc_matrix(
        (data, (indices[:, 0], indices[:, 1])),
        shape=value.shape,
    )


def residualize_screen(
    screen: PerturbData,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    output: Optional[Path] = None,
) -> PerturbData:
    """Regress selected covariates out of expression in gene blocks.

    ``block_size`` bounds peak memory (genes per projection chunk) and defaults
    to :data:`perturbvi._defaults.DEFAULT_BLOCK_SIZE`. Output is float32.
    """
    validate_screen(screen)
    if screen.covariates is None:
        raise ValueError("screen.covariates is None. Load the screen with covariates=[...] first.")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    design, design_names = _build_design_matrix_with_names(screen.covariates)
    basis = _projection_basis(design, design_names)
    n_cells, n_genes = screen.X.shape
    sparse_x = _as_csc(screen.X)
    dense_x = None if sparse_x is not None else np.asarray(screen.X, dtype=np.float32)
    if output is None:
        X_resid = np.empty((n_cells, n_genes), dtype=np.float32, order="C")
    else:
        X_resid = np.memmap(
            Path(output),
            mode="w+",
            dtype=np.float32,
            shape=(n_cells, n_genes),
            order="F",
        )

    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        if sparse_x is not None:
            block = sparse_x[:, start:stop].toarray()
        else:
            block = dense_x[:, start:stop]
        block = np.asarray(block, dtype=np.float32)
        X_resid[:, start:stop] = block - basis @ (basis.T @ block)
    if isinstance(X_resid, np.memmap):
        X_resid.flush()
    return replace(screen, X=X_resid, covariates=None)


__all__ = ["residualize_screen"]
