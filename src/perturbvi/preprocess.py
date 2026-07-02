from typing import Optional, Sequence

import numpy as np

from .screen import ScreenData


def _build_design_matrix(
    covariates: np.ndarray,
    covariate_names: Sequence[str],
    categoricals: Optional[Sequence[str]],
) -> np.ndarray:
    cat_set = set(categoricals or [])
    cols = [np.ones((covariates.shape[0], 1))]

    for i, name in enumerate(covariate_names):
        col = covariates[:, i]
        if name in cat_set:
            unique_vals = np.unique(col)
            for val in unique_vals[1:]:  # drop first category (reference level)
                cols.append((col == val).astype(np.float64).reshape(-1, 1))
        else:
            centered = col.astype(np.float64) - col.mean()
            cols.append(centered.reshape(-1, 1))

    return np.hstack(cols)


def _qr_residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(C, mode="reduced")
    return X - Q @ (Q.T @ X)


def residualize_screen(
    screen: ScreenData,
    *,
    categoricals: Optional[Sequence[str]] = None,
) -> ScreenData:
    """Regress covariates out of X using QR-based least squares.

    Args:
        screen: Must have screen.covariates populated via load_screen(covariates=...).
        categoricals: Subset of screen.covariate_names to one-hot encode.
                      All other covariates are treated as numeric (centered).

    Returns:
        New ScreenData with residualized X and covariates cleared.
    """
    if screen.covariates is None:
        raise ValueError(
            "screen.covariates is None. "
            "Load the screen with covariates=['col1', ...] to extract covariate columns."
        )

    cov_names = list(screen.covariate_names or [])

    if categoricals:
        unknown = [c for c in categoricals if c not in cov_names]
        if unknown:
            raise ValueError(
                f"categoricals {unknown} not found in covariate_names: {cov_names}"
            )

    C = _build_design_matrix(np.asarray(screen.covariates), cov_names, categoricals)

    if not np.all(np.isfinite(C)):
        raise ValueError("Design matrix contains non-finite values after encoding.")

    X_resid = _qr_residualize(np.asarray(screen.X, dtype=np.float64), C)

    source = dict(screen.source)
    source["residualized"] = True
    source["cov_names"] = cov_names
    source["categoricals"] = list(categoricals or [])

    return screen._replace(X=X_resid, covariates=None, covariate_names=None, source=source)
