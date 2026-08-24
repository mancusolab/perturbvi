from __future__ import annotations

import pickle

from pathlib import Path
from typing import Union

import numpy as np

from .infer import compute_pip, compute_pve, InferResults
from .log import get_logger
from .screen import FitResults


log = get_logger(__name__)


def save_results(results: Union[InferResults, FitResults], path: str) -> None:
    """Save a fitted model and its stable text exports."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    inference = results.inference if isinstance(results, FitResults) else results

    np.savetxt(output / "W.txt", np.asarray(inference.W))
    np.savetxt(output / "pip.txt", np.asarray(inference.pip))
    np.savetxt(output / "pve.txt", np.asarray(inference.pve))

    payload = inference.params
    if isinstance(results, FitResults):
        payload = {
            "params": inference.params,
            "gene_names": results.gene_names,
            "perturbation_names": results.perturbation_names,
        }
    with (output / "params_file.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    log.info(f"Results saved to {output}")


def load_results(path: str) -> Union[InferResults, FitResults]:
    """Reload a fit from ``params_file.pkl``.

    The text files written by :func:`save_results` are exports, not load-time
    dependencies.
    """
    with (Path(path) / "params_file.pkl").open("rb") as stream:
        payload = pickle.load(stream)

    if isinstance(payload, dict) and "params" in payload:
        params = payload["params"]
        gene_names = payload.get("gene_names")
        perturbation_names = payload.get("perturbation_names")
    else:
        params = payload
        gene_names = None
        perturbation_names = None

    inference = InferResults(params=params, elbo=None, pve=compute_pve(params), pip=compute_pip(params))
    if gene_names is None or perturbation_names is None:
        return inference
    return FitResults(
        inference=inference,
        gene_names=tuple(str(name) for name in gene_names),
        perturbation_names=tuple(str(name) for name in perturbation_names),
    )


__all__ = ["load_results", "save_results"]
