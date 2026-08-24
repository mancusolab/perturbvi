from __future__ import annotations

import numbers

from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from .screen import FitResults


if TYPE_CHECKING:
    from .infer import InferResults


def _labels(names: Optional[Sequence[str]], size: int, label: str) -> list[str] | list[int]:
    if names is None:
        return list(range(size))
    values = [str(name) for name in names]
    if len(values) != size:
        raise ValueError(f"{label} has {len(values)} entries; expected {size}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate names")
    return values


def _compute_lfsr(
    results,
    *,
    gene_names: Sequence,
    perturbation_names: Sequence,
    lfsr_iters: int,
    seed: int,
) -> pd.DataFrame:
    from jax import random as rdm

    from .utils import compute_lfsr

    values = compute_lfsr(rdm.PRNGKey(seed), results.params, iters=lfsr_iters)
    return pd.DataFrame(np.asarray(values).T, index=gene_names, columns=perturbation_names)


def _analyze(
    results,
    *,
    gene_names: Optional[Sequence[str]],
    perturbation_names: Optional[Sequence[str]],
    compute_lfsr: bool,
    lfsr_iters: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    pip = np.asarray(results.pip)
    pve = np.asarray(results.pve).reshape(-1)
    W = np.asarray(results.params.W)
    mean_beta = np.asarray(results.params.mean_beta)
    p_hat = np.asarray(results.params.p_hat)

    genes = _labels(gene_names, pip.shape[1], "gene_names")
    perturbations = _labels(perturbation_names, mean_beta.shape[0], "perturbation_names")
    factors = [f"factor_{index}" for index in range(pip.shape[0])]
    perturbation_effect = mean_beta * p_hat.T

    tables = {
        "pip": pd.DataFrame(pip.T, index=genes, columns=factors),
        "pve": pd.DataFrame({"pve": pve}, index=pd.Index(factors, name="factor")),
        "perturbation_effect": pd.DataFrame(perturbation_effect, index=perturbations, columns=factors),
        "perturbation_pip": pd.DataFrame(p_hat.T, index=perturbations, columns=factors),
        "gene_effect": pd.DataFrame((perturbation_effect @ W).T, index=genes, columns=perturbations),
    }
    if compute_lfsr:
        if isinstance(lfsr_iters, bool) or not isinstance(lfsr_iters, numbers.Integral) or lfsr_iters <= 0:
            raise ValueError("lfsr_iters must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError("seed must be an integer")
        tables["lfsr"] = _compute_lfsr(
            results,
            gene_names=genes,
            perturbation_names=perturbations,
            lfsr_iters=lfsr_iters,
            seed=seed,
        )
    return tables


def analyze_output(
    results_or_path: InferResults | FitResults | str | Path,
    *,
    gene_names: Optional[Sequence[str]] = None,
    perturbation_names: Optional[Sequence[str]] = None,
    compute_lfsr: bool = False,
    lfsr_iters: int = 2000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Create labeled result tables, optionally computing LFSR.

    Paths are loaded with :func:`load_results`. This function does not write
    files and never reads or computes LFSR unless ``compute_lfsr=True``.
    """
    if isinstance(results_or_path, (str, Path)):
        from .io import load_results

        results_or_path = load_results(results_or_path)
    if isinstance(results_or_path, FitResults):
        if gene_names is None:
            gene_names = results_or_path.gene_names
        if perturbation_names is None:
            perturbation_names = results_or_path.perturbation_names
    return _analyze(
        results_or_path,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        compute_lfsr=compute_lfsr,
        lfsr_iters=lfsr_iters,
        seed=seed,
    )
