"""Posterior sign uncertainty for overall perturbation effects."""

from __future__ import annotations

import numbers

from pathlib import Path

import numpy as np
import pandas as pd

from jax import random

from ._results import _posterior
from .infer import InferResults
from .screen import FitResults


def estimate_lfsr(
    results: FitResults | InferResults | str | Path,
    *,
    draws: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate overall-effect LFSR from an in-memory or saved fit.

    Returns a labeled DataFrame with perturbations on rows and genes on
    columns, matching ``BW.csv``. Nothing is written; use ``to_csv()`` to
    save it. A directory is loaded internally from ``model.pkl`` (or the
    older ``params_file.pkl``). No other summary tables are calculated.

    Each draw samples perturbation coefficients and gene loadings from the
    fitted variational posterior and multiplies them across all factors.
    LFSR is the smaller of the fractions of nonnegative and nonpositive
    overall effects; exact zeros count in both fractions. ``draws`` controls
    Monte Carlo precision, and ``seed`` controls reproducibility.

    Saved fits retain their labels. Low-level or older fits without labels
    use positional identifiers. This computes fresh samples, rather than
    reading an existing ``LFSR_BW.csv``.
    """
    from .utils import compute_lfsr

    if isinstance(draws, bool) or not isinstance(draws, numbers.Integral) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")

    params, genes, perturbations = _posterior(results)
    values = compute_lfsr(random.PRNGKey(seed), params, iters=draws)
    return pd.DataFrame(
        np.asarray(values),
        index=pd.Index(perturbations, name="perturbation_id"),
        columns=genes,
    )
