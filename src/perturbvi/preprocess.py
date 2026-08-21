from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .screen import matrix_to_numpy, ScreenData, validate_screen


def _build_design_matrix(
    covariates: np.ndarray,
    covariate_names: Sequence[str],
    categorical_covariates: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Build an intercept plus centered numeric and reference-coded categorical columns."""
    covariates = np.asarray(covariates)
    names = list(covariate_names)
    cat_names = list(categorical_covariates or [])
    cat_set = set(cat_names)

    if covariates.ndim != 2:
        raise ValueError(f"covariates must be 2D; got shape {covariates.shape}")
    if covariates.shape[1] != len(names):
        raise ValueError(
            f"covariate_names has {len(names)} entries but covariates has {covariates.shape[1]} columns"
        )
    unknown = [name for name in cat_names if name not in names]
    if unknown:
        raise ValueError(f"categorical_covariates {unknown} not found in covariate_names: {names}")
    if np.asarray(pd.isna(covariates), dtype=bool).any():
        raise ValueError("Covariates contain missing values")

    columns = [np.ones((covariates.shape[0], 1), dtype=np.float64)]
    for index, name in enumerate(names):
        column = covariates[:, index]
        if name in cat_set:
            levels = list(pd.unique(column))
            for level in levels[1:]:
                columns.append((column == level).astype(np.float64).reshape(-1, 1))
            continue

        try:
            numeric = column.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Covariate '{name}' is not numeric; include it in categorical_covariates"
            ) from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"Covariate '{name}' contains non-finite values")
        columns.append((numeric - numeric.mean()).reshape(-1, 1))

    design = np.hstack(columns)
    if not np.all(np.isfinite(design)):
        raise ValueError("Design matrix contains non-finite values after encoding")
    return design


def _least_squares_residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residualize with an SVD-backed, rank-aware least-squares solve."""
    coefficients, _, _, _ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ coefficients


def residualize_screen(
    screen: ScreenData,
    *,
    covariates: Optional[Sequence[str]] = None,
    categorical_covariates: Optional[Sequence[str]] = None,
) -> ScreenData:
    """Regress selected covariates out of every expression feature.

    Numeric covariates are centered, categorical covariates use reference-level
    one-hot coding, and an intercept is included. An SVD-backed least-squares
    solve handles rank-deficient designs predictably. This explicitly dense
    operation preserves cell order, records its choices in ``screen.source``,
    and removes only the consumed columns from the returned covariate matrix.

    Args:
        screen: A validated screen loaded with covariates.
        covariates: Covariate names to use. All loaded covariates are used when
            omitted.
        categorical_covariates: Selected covariates to encode as categorical.
    Returns:
        A new screen containing dense residualized expression and any
        covariates that were loaded but not selected.
    """
    validate_screen(screen)
    if screen.covariates is None:
        raise ValueError(
            "screen.covariates is None. Load the screen with covariates=['col1', ...] first."
        )
    if screen.covariate_names is None:
        raise ValueError("screen.covariate_names is required for residualization")

    all_names = list(screen.covariate_names)
    selected = list(covariates) if covariates is not None else all_names
    if not selected:
        raise ValueError("At least one covariate must be selected for residualization")
    unknown = [name for name in selected if name not in all_names]
    if unknown:
        raise ValueError(f"covariates {unknown} not found in covariate_names: {all_names}")

    cat_names = list(categorical_covariates or [])
    unknown_cat = [name for name in cat_names if name not in all_names]
    if unknown_cat:
        raise ValueError(
            f"categorical_covariates {unknown_cat} not found in covariate_names: {all_names}"
        )
    unselected_cat = [name for name in cat_names if name not in selected]
    if unselected_cat:
        raise ValueError(
            f"categorical_covariates {unselected_cat} are not selected covariates: {selected}"
        )

    selected_idx = [all_names.index(name) for name in selected]
    selected_values = np.asarray(screen.covariates)[:, selected_idx]
    design = _build_design_matrix(
        selected_values,
        selected,
        categorical_covariates=cat_names,
    )
    if np.linalg.matrix_rank(design) <= 1:
        raise ValueError("Covariate design must contain at least one non-intercept direction")

    X_resid = _least_squares_residualize(
        matrix_to_numpy(screen.X, dtype=np.float64),
        design,
    )
    source = dict(screen.source)
    source.update(
        {
            "residualized": True,
            "residualized_covariate_names": selected,
            "residualized_categorical_covariates": cat_names,
            "residualization_method": "rank-aware least squares",
        }
    )
    keep_idx = [index for index, name in enumerate(all_names) if name not in selected]
    kept_covars = np.asarray(screen.covariates)[:, keep_idx] if keep_idx else None
    kept_names = [all_names[index] for index in keep_idx] or None
    return screen._replace(
        X=X_resid,
        covariates=kept_covars,
        covariate_names=kept_names,
        source=source,
    )
