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
    categorical_names = list(categorical_covariates or [])
    categorical_set = set(categorical_names)

    if covariates.ndim != 2:
        raise ValueError(f"covariates must be 2D; got shape {covariates.shape}")
    if covariates.shape[1] != len(names):
        raise ValueError(
            f"covariate_names has {len(names)} entries but covariates has {covariates.shape[1]} columns"
        )
    unknown = [name for name in categorical_names if name not in names]
    if unknown:
        raise ValueError(f"categorical_covariates {unknown} not found in covariate_names: {names}")
    if np.asarray(pd.isna(covariates), dtype=bool).any():
        raise ValueError("Covariates contain missing values")

    columns = [np.ones((covariates.shape[0], 1), dtype=np.float64)]
    for index, name in enumerate(names):
        column = covariates[:, index]
        if name in categorical_set:
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

    available_names = list(screen.covariate_names)
    selected_names = list(covariates) if covariates is not None else available_names
    if not selected_names:
        raise ValueError("At least one covariate must be selected for residualization")
    unknown = [name for name in selected_names if name not in available_names]
    if unknown:
        raise ValueError(f"covariates {unknown} not found in covariate_names: {available_names}")

    categorical_names = list(categorical_covariates or [])
    unknown_categorical = [name for name in categorical_names if name not in available_names]
    if unknown_categorical:
        raise ValueError(
            f"categorical_covariates {unknown_categorical} not found in covariate_names: {available_names}"
        )
    outside_selection = [name for name in categorical_names if name not in selected_names]
    if outside_selection:
        raise ValueError(
            f"categorical_covariates {outside_selection} are not selected covariates: {selected_names}"
        )

    selected_indices = [available_names.index(name) for name in selected_names]
    selected_values = np.asarray(screen.covariates)[:, selected_indices]
    design = _build_design_matrix(
        selected_values,
        selected_names,
        categorical_covariates=categorical_names,
    )
    if np.linalg.matrix_rank(design) <= 1:
        raise ValueError("Covariate design must contain at least one non-intercept direction")

    X_residualized = _least_squares_residualize(
        matrix_to_numpy(screen.X, dtype=np.float64),
        design,
    )
    source = dict(screen.source)
    source.update(
        {
            "residualized": True,
            "residualized_covariate_names": selected_names,
            "residualized_categorical_covariates": categorical_names,
            "residualization_method": "rank-aware least squares",
        }
    )
    remaining_indices = [index for index, name in enumerate(available_names) if name not in selected_names]
    remaining_covariates = (
        np.asarray(screen.covariates)[:, remaining_indices] if remaining_indices else None
    )
    remaining_names = [available_names[index] for index in remaining_indices] or None
    return screen._replace(
        X=X_residualized,
        covariates=remaining_covariates,
        covariate_names=remaining_names,
        source=source,
    )
