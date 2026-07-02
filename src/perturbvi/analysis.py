from typing import Optional, Sequence

import numpy as np
import pandas as pd


def analyze(
    results,
    *,
    gene_names: Optional[Sequence[str]] = None,
    perturbation_names: Optional[Sequence[str]] = None,
    compute_lfsr: bool = False,
    lfsr_iters: int = 2000,
    seed: int = 0,
) -> dict:
    """Build analysis tables from an InferResults object.

    Args:
        results: InferResults returned by infer() or fit_screen(), or loaded by load_results().
        gene_names: Gene names for row labels. If None, integer indices are used.
        perturbation_names: Perturbation names for column/row labels. If None, integer indices are used.
        compute_lfsr: If True, compute LFSR. Expensive — skip unless needed.
        lfsr_iters: Number of Monte Carlo iterations for LFSR (used only if compute_lfsr=True).
        seed: Random seed for LFSR sampling (used only if compute_lfsr=True).

    Returns:
        dict with keys: pip_df, pve_df, beta_df, p_hat_df, overall_effect_df,
        and lfsr_df if compute_lfsr=True.
    """
    params = results.params
    pip = np.array(results.pip)

    z_dim, p_dim = pip.shape
    g_dim = params.mean_beta.shape[0]  # noqa: F841

    col_names_w = [f"w{i}" for i in range(z_dim)]
    col_names_b = [f"b{i}" for i in range(z_dim)]

    pip_df = pd.DataFrame(pip.T, columns=col_names_w, index=gene_names)

    pve_df = pd.DataFrame({"factor": col_names_w, "pve": np.array(results.pve)})

    beta_sparse = np.array(params.mean_beta * params.p_hat.T)
    beta_df = pd.DataFrame(beta_sparse, columns=col_names_b, index=perturbation_names)

    p_hat_df = pd.DataFrame(np.array(params.p_hat.T), columns=col_names_b, index=perturbation_names)

    W = np.array(params.W)
    overall_effect = beta_sparse @ W
    overall_effect_df = pd.DataFrame(overall_effect.T, columns=perturbation_names, index=gene_names)

    out = {
        "pip_df": pip_df,
        "pve_df": pve_df,
        "beta_df": beta_df,
        "p_hat_df": p_hat_df,
        "overall_effect_df": overall_effect_df,
    }

    if compute_lfsr:
        from jax import random as rdm

        from .utils import compute_lfsr as _compute_lfsr

        lfsr = _compute_lfsr(rdm.PRNGKey(seed), params, iters=lfsr_iters)
        lfsr_df = pd.DataFrame(np.array(lfsr).T, index=gene_names, columns=perturbation_names)
        out["lfsr_df"] = lfsr_df

    return out
