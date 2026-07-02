from typing import Any, Mapping, NamedTuple, Optional, Sequence, TYPE_CHECKING

import numpy as np

from jaxtyping import ArrayLike


if TYPE_CHECKING:
    from .infer import InferResults


class ScreenData(NamedTuple):
    X: ArrayLike                          # cells × genes
    G: ArrayLike                          # cells × perturbations
    gene_names: Optional[Sequence[str]]
    perturbation_names: Optional[Sequence[str]]
    cell_names: Optional[Sequence[str]]
    source: Mapping[str, Any]             # loader metadata: path, format, layer, etc.
    covariates: Optional[ArrayLike] = None         # cells × covariates (raw values)
    covariate_names: Optional[Sequence[str]] = None


def validate_screen(screen: ScreenData) -> None:
    X = np.asarray(screen.X)
    G = np.asarray(screen.G)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}")
    if G.ndim != 2:
        raise ValueError(f"G must be 2D; got shape {G.shape}")
    if X.shape[0] != G.shape[0]:
        raise ValueError(
            f"X and G must have the same number of rows (cells); "
            f"got X: {X.shape[0]}, G: {G.shape[0]}"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (nan or inf)")

    col_sums = np.sum(np.abs(G), axis=0)
    empty_cols = np.where(col_sums == 0)[0]
    if empty_cols.size > 0:
        raise ValueError(f"G has all-zero columns at indices {empty_cols.tolist()}")

    if screen.gene_names is not None and len(screen.gene_names) != X.shape[1]:
        raise ValueError(
            f"gene_names has {len(screen.gene_names)} entries but X has {X.shape[1]} columns"
        )
    if screen.perturbation_names is not None and len(screen.perturbation_names) != G.shape[1]:
        raise ValueError(
            f"perturbation_names has {len(screen.perturbation_names)} entries but G has {G.shape[1]} columns"
        )
    if screen.cell_names is not None and len(screen.cell_names) != X.shape[0]:
        raise ValueError(
            f"cell_names has {len(screen.cell_names)} entries but X has {X.shape[0]} rows"
        )

    if screen.covariates is not None:
        C = np.asarray(screen.covariates)
        if C.ndim != 2:
            raise ValueError(f"covariates must be 2D; got shape {C.shape}")
        if C.shape[0] != X.shape[0]:
            raise ValueError(
                f"covariates must have {X.shape[0]} rows (cells); got {C.shape[0]}"
            )
        if np.issubdtype(C.dtype, np.number) and not np.all(np.isfinite(C)):
            raise ValueError("covariates contains non-finite values")
        if screen.covariate_names is not None and len(screen.covariate_names) != C.shape[1]:
            raise ValueError(
                f"covariate_names has {len(screen.covariate_names)} entries "
                f"but covariates has {C.shape[1]} columns"
            )


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
) -> "InferResults":
    from .infer import infer

    validate_screen(screen)
    return infer(
        screen.X,
        z_dim=z_dim,
        l_dim=l_dim,
        G=screen.G,
        tau=tau,
        p_prior=p_prior,
        standardize=standardize,
        init=init,
        tol=tol,
        max_iter=max_iter,
        seed=seed,
    )
