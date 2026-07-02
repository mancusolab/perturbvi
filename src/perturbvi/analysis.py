import warnings

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def analyze(
    results,
    *,
    genes: Optional[Sequence[str]] = None,
    perturbations: Optional[Sequence[str]] = None,
    compute_lfsr: bool = False,
    lfsr_iters: int = 2000,
    path: Optional[str] = None,
) -> dict:
    """Build analysis tables from an InferResults object.

    Args:
        results: InferResults returned by infer() or fit_screen(), or loaded by load_results().
        genes: Gene names for row labels. If None, integer indices are used.
        perturbations: Perturbation names for column/row labels. If None, integer indices are used.
        compute_lfsr: If True, compute LFSR (expensive Monte Carlo step).
        lfsr_iters: Number of Monte Carlo iterations for LFSR (used only if compute_lfsr=True).
        path: Directory containing (or to receive) lfsr.csv. If compute_lfsr=True, LFSR is
            written to f"{path}/lfsr.csv". If compute_lfsr=False, an existing
            f"{path}/lfsr.csv" is loaded instead of recomputing. Ignored if path is None.

    Returns:
        dict with keys: pip_df, pve_df, beta_df, p_hat_df, overall_effect_df, lfsr.
        lfsr is a DataFrame if computed or found on disk, otherwise None.
    """
    params = results.params
    pip = np.array(results.pip)

    z_dim, p_dim = pip.shape
    g_dim = params.mean_beta.shape[0]  # noqa: F841

    col_names_w = [f"w{i}" for i in range(z_dim)]
    col_names_b = [f"b{i}" for i in range(z_dim)]

    pip_df = pd.DataFrame(pip.T, columns=col_names_w, index=genes)

    pve_df = pd.DataFrame({"factor": col_names_w, "pve": np.array(results.pve)})

    beta_sparse = np.array(params.mean_beta * params.p_hat.T)
    beta_df = pd.DataFrame(beta_sparse, columns=col_names_b, index=perturbations)

    p_hat_df = pd.DataFrame(np.array(params.p_hat.T), columns=col_names_b, index=perturbations)

    W = np.array(params.W)
    overall_effect = beta_sparse @ W
    overall_effect_df = pd.DataFrame(overall_effect.T, columns=perturbations, index=genes)

    out = {
        "pip_df": pip_df,
        "pve_df": pve_df,
        "beta_df": beta_df,
        "p_hat_df": p_hat_df,
        "overall_effect_df": overall_effect_df,
    }

    if compute_lfsr:
        lfsr_df = _compute_lfsr(results, genes=genes, perturbations=perturbations, lfsr_iters=lfsr_iters)
        if path is not None:
            lfsr_df.to_csv(Path(path) / "lfsr.csv")
        out["lfsr"] = lfsr_df
    elif path is not None and (Path(path) / "lfsr.csv").exists():
        out["lfsr"] = pd.read_csv(Path(path) / "lfsr.csv", index_col=0)
    else:
        if path is not None:
            warnings.warn(
                f"Could not locate lfsr.csv in {path} directory. lfsr is None. "
                "Pass compute_lfsr=True to compute it.",
                stacklevel=2,
            )
        out["lfsr"] = None

    return out


def _compute_lfsr(
    results,
    *,
    genes: Optional[Sequence[str]] = None,
    perturbations: Optional[Sequence[str]] = None,
    lfsr_iters: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute the local false sign rate (LFSR) for each gene x perturbation pair.

    Private for now. Expensive: runs an `lfsr_iters`-draw Monte Carlo computation.
    """
    from jax import random as rdm

    from .utils import compute_lfsr as _compute_lfsr_raw

    lfsr = _compute_lfsr_raw(rdm.PRNGKey(seed), results.params, iters=lfsr_iters)
    return pd.DataFrame(np.array(lfsr).T, index=genes, columns=perturbations)
