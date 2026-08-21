from __future__ import annotations

import numbers

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _validate_threshold(value: float, name: str) -> None:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a numeric value between 0 and 1; received {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric value between 0 and 1; received {value!r}") from exc
    if not np.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be between 0 and 1; received {value}")


def _labels(names: Optional[Sequence[str]], size: int, label: str) -> list:
    if names is None:
        return list(range(size))
    names = list(names)
    if len(names) != size:
        raise ValueError(f"{label} has {len(names)} entries but expected {size}")
    if np.asarray(pd.isna(names), dtype=bool).any():
        raise ValueError(f"{label} contains missing values")
    as_strings = [str(name) for name in names]
    if len(set(as_strings)) != len(as_strings):
        raise ValueError(f"{label} contains duplicate names")
    return names


def _validate_results(results) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = results.params
    pip = np.asarray(results.pip)
    pve = np.asarray(results.pve).reshape(-1)
    W = np.asarray(params.W)
    mean_beta = np.asarray(params.mean_beta)
    p_hat = np.asarray(params.p_hat)
    if pip.ndim != 2:
        raise ValueError(f"results.pip must be 2D; got shape {pip.shape}")
    z_dim, gene_count = pip.shape
    if W.shape != (z_dim, gene_count):
        raise ValueError(f"params.W has shape {W.shape}; expected {(z_dim, gene_count)}")
    if pve.shape != (z_dim,):
        raise ValueError(f"results.pve has shape {pve.shape}; expected {(z_dim,)}")
    if mean_beta.ndim != 2 or mean_beta.shape[1] != z_dim:
        raise ValueError(f"params.mean_beta has incompatible shape {mean_beta.shape}")
    if p_hat.shape != (z_dim, mean_beta.shape[0]):
        raise ValueError(f"params.p_hat has shape {p_hat.shape}; expected {(z_dim, mean_beta.shape[0])}")
    for name, value in {"pip": pip, "pve": pve, "W": W, "mean_beta": mean_beta, "p_hat": p_hat}.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"results contain non-finite values in {name}")
    return pip, pve, W, mean_beta, p_hat


def _pip_significant_table(pip_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for factor in pip_df.columns:
        selected = pip_df.index[pip_df[factor] >= threshold]
        rows.extend({"factor": factor, "gene": gene, "pip": pip_df.at[gene, factor]} for gene in selected)
    return pd.DataFrame(rows, columns=["factor", "gene", "pip"])


def _attach_lfsr(
    tables: dict,
    lfsr_df: pd.DataFrame,
    *,
    gene_names: Sequence,
    perturbation_names: Sequence,
    lfsr_threshold: float,
) -> None:
    expected = (len(gene_names), len(perturbation_names))
    if lfsr_df.shape != expected:
        raise ValueError(f"LFSR table has shape {lfsr_df.shape}; expected {expected}")
    lfsr_df = pd.DataFrame(lfsr_df.to_numpy(), index=gene_names, columns=perturbation_names)
    rows = []
    for perturbation in lfsr_df.columns:
        selected = lfsr_df.index[lfsr_df[perturbation] <= lfsr_threshold]
        rows.extend(
            {
                "perturbation": perturbation,
                "gene": gene,
                "lfsr": lfsr_df.at[gene, perturbation],
            }
            for gene in selected
        )
    tables["lfsr"] = lfsr_df
    tables["lfsr_significant"] = pd.DataFrame(rows, columns=["perturbation", "gene", "lfsr"])


def _analyze_result(
    results,
    *,
    gene_names: Optional[Sequence[str]] = None,
    perturbation_names: Optional[Sequence[str]] = None,
    pip_threshold: float = 0.9,
    lfsr_threshold: float = 0.05,
    compute_lfsr: bool = False,
    lfsr_iters: int = 2000,
    seed: int = 0,
) -> dict:
    """Build explicit tables from an in-memory fit without implicit I/O.

    Args:
        results: An :class:`InferResults` returned by ``infer`` or ``fit_screen``.
        gene_names: Optional labels for expression features.
        perturbation_names: Optional labels for guide features.
        pip_threshold: Inclusive threshold used by ``pip_significant``.
        lfsr_threshold: Inclusive threshold used by ``lfsr_significant``.
        compute_lfsr: Explicitly run the expensive Monte Carlo LFSR step.
        lfsr_iters: Positive number of Monte Carlo iterations.
        seed: Integer random seed passed to LFSR computation.

    Returns:
        A dictionary of PIP, PVE, perturbation-effect, inclusion-probability,
        overall-effect, and threshold-summary DataFrames. LFSR entries are
        ``None`` unless explicitly computed.
    """
    _validate_threshold(pip_threshold, "pip_threshold")
    _validate_threshold(lfsr_threshold, "lfsr_threshold")
    if isinstance(lfsr_iters, bool) or not isinstance(lfsr_iters, numbers.Integral) or lfsr_iters <= 0:
        raise ValueError(f"lfsr_iters must be a positive integer; received {lfsr_iters}")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f"seed must be an integer; received {seed!r}")

    pip, pve, W, mean_beta, p_hat = _validate_results(results)
    z_dim, gene_count = pip.shape
    perturbation_count = mean_beta.shape[0]
    genes = _labels(gene_names, gene_count, "gene_names")
    perturbations = _labels(perturbation_names, perturbation_count, "perturbation_names")
    factor_names = [f"w{index}" for index in range(z_dim)]
    beta_names = [f"b{index}" for index in range(z_dim)]

    pip_df = pd.DataFrame(pip.T, columns=factor_names, index=genes)
    beta_sparse = mean_beta * p_hat.T
    beta_df = pd.DataFrame(beta_sparse, columns=beta_names, index=perturbations)
    p_hat_df = pd.DataFrame(p_hat.T, columns=beta_names, index=perturbations)
    overall_effect_df = pd.DataFrame((beta_sparse @ W).T, columns=perturbations, index=genes)
    pip_summary_df = pd.DataFrame(
        {
            "factor": factor_names,
            "pve": pve,
            "n_pip_significant": (pip >= pip_threshold).sum(axis=1),
        }
    )

    tables = {
        "pip": pip_df,
        "pve": pd.DataFrame({"factor": factor_names, "pve": pve}),
        "beta": beta_df,
        "p_hat": p_hat_df,
        "overall_effect": overall_effect_df,
        "pip_significant": _pip_significant_table(pip_df, pip_threshold),
        "pip_summary": pip_summary_df,
        "lfsr": None,
        "lfsr_significant": None,
    }
    if compute_lfsr:
        lfsr_df = _compute_lfsr(
            results,
            gene_names=genes,
            perturbation_names=perturbations,
            lfsr_iters=lfsr_iters,
            seed=seed,
        )
        _attach_lfsr(
            tables,
            lfsr_df,
            gene_names=genes,
            perturbation_names=perturbations,
            lfsr_threshold=lfsr_threshold,
        )
    return tables


def _read_saved_names(path: Path, filename: str) -> Optional[list[str]]:
    names_path = path / filename
    if not names_path.is_file():
        return None
    return [line.rstrip("\n\r") for line in names_path.read_text(encoding="utf-8").splitlines()]


def _analyze_saved_result(
    path: str,
    *,
    gene_names: Optional[Sequence[str]] = None,
    perturbation_names: Optional[Sequence[str]] = None,
    **kwargs,
) -> dict:
    """Load saved results and delegate to the in-memory analysis core.

    Saved gene and perturbation names are used when explicit names are omitted.
    A pre-existing root ``lfsr.csv`` is reused without recomputation; a new LFSR
    is calculated only when ``compute_lfsr=True`` is forwarded in ``kwargs``.
    """
    from .io import load_results

    result_path = Path(path)
    if gene_names is None:
        gene_names = _read_saved_names(result_path, "gene_names.txt")
    if perturbation_names is None:
        perturbation_names = _read_saved_names(result_path, "perturbation_names.txt")
    tables = _analyze_result(
        load_results(str(result_path)),
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        **kwargs,
    )

    if not kwargs.get("compute_lfsr", False) and (result_path / "lfsr.csv").is_file():
        cached = pd.read_csv(result_path / "lfsr.csv", index_col=0)
        genes = list(tables["pip"].index)
        perturbations = list(tables["beta"].index)
        _attach_lfsr(
            tables,
            cached,
            gene_names=genes,
            perturbation_names=perturbations,
            lfsr_threshold=kwargs.get("lfsr_threshold", 0.05),
        )
    return tables


def analyze(
    results_or_path,
    *,
    gene_names: Optional[Sequence[str]] = None,
    perturbation_names: Optional[Sequence[str]] = None,
    pip_threshold: float = 0.9,
    lfsr_threshold: float = 0.05,
    compute_lfsr: bool = False,
    lfsr_iters: int = 2000,
    seed: int = 0,
) -> dict:
    """Analyze an in-memory fit or a saved result directory.

    A path automatically loads saved names and an existing root ``lfsr.csv``.
    An :class:`InferResults` is analyzed without I/O.
    """
    if isinstance(results_or_path, (str, Path)):
        return _analyze_saved_result(
            str(results_or_path),
            gene_names=gene_names,
            perturbation_names=perturbation_names,
            pip_threshold=pip_threshold,
            lfsr_threshold=lfsr_threshold,
            compute_lfsr=compute_lfsr,
            lfsr_iters=lfsr_iters,
            seed=seed,
        )

    return _analyze_result(
        results_or_path,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        pip_threshold=pip_threshold,
        lfsr_threshold=lfsr_threshold,
        compute_lfsr=compute_lfsr,
        lfsr_iters=lfsr_iters,
        seed=seed,
    )


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
