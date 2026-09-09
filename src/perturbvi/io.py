from __future__ import annotations

import pickle

from pathlib import Path

import numpy as np
from jax.experimental import enable_x64

from .infer import InferResults
from .log import get_logger
from .screen import FitResults


log = get_logger(__name__)


def save_results(
    results: InferResults | FitResults | str | Path,
    path: str | Path | None = None,
) -> None:
    """Save ``model.pkl`` and six labeled CSV summaries.

    Writes ``W.csv``, ``PIP_W.csv``, ``B.csv``, ``PIP_B.csv``, ``BW.csv``, and
    ``PVE.csv``. Matrices follow the model algebra: W is factors by genes,
    B is perturbations by factors, and BW is perturbations by genes.
    LFSR requires ``estimate_lfsr()`` or the ``perturbvi lfsr`` CLI; no
    sampling occurs here.

    Pass a fitted result and destination, e.g. ``save_results(fit, "results")``.
    To regenerate summaries without refitting, pass a saved directory alone:
    ``save_results("results")``. This refreshes the six CSVs without rewriting
    its existing ``model.pkl``. A second directory copies the posterior and
    writes summaries there. Existing LFSR files are not modified or copied.
    """
    from ._results import _posterior
    from .infer import compute_pip, compute_pve

    source = Path(results) if isinstance(results, (str, Path)) else None
    if path is None:
        if source is None:
            raise ValueError("path is required when saving an in-memory fit")
        path = source
    output = Path(path)
    if not isinstance(results, FitResults):
        params, genes, perturbations = _posterior(results)
        if source is not None:
            with enable_x64(np.asarray(params.mean_w).dtype == np.dtype("float64")):
                results = InferResults(
                    params=params, elbo=None, pip=compute_pip(params), pve=compute_pve(params),
                )
        results = FitResults(results, genes, perturbations)

    output.mkdir(parents=True, exist_ok=True)
    W, B = results.W, results.B
    for name, table in (
        ("W", W), ("PIP_W", results.PIP_W), ("B", B),
        ("PIP_B", results.PIP_B), ("BW", B @ W), ("PVE", results.PVE),
    ):
        table.to_csv(output / f"{name}.csv")

    if source is None or source.resolve() != output.resolve() or not (output / "model.pkl").exists():
        payload = {
            "params": results.params,
            "gene_names": results.gene_names,
            "perturbation_names": results.perturbation_names,
        }
        with (output / "model.pkl").open("wb") as stream:
            pickle.dump(payload, stream)
    log.info(f"Results saved to {output}")


__all__ = ["save_results"]
