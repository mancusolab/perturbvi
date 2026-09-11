"""Internal access to saved posterior parameters and their labels."""

from __future__ import annotations

import pickle

from pathlib import Path

from .screen import FitResults


def _posterior(results):
    """Return parameters and labels without computing summary matrices."""
    if isinstance(results, (str, Path)):
        directory = Path(results)
        model_file = directory / "model.pkl"
        if not model_file.exists():
            model_file = directory / "params_file.pkl"
        with model_file.open("rb") as stream:
            payload = pickle.load(stream)
        if isinstance(payload, dict) and "params" in payload:
            params = payload["params"]
            genes = payload.get("gene_names")
            perturbations = payload.get("perturbation_names")
        else:
            params, genes, perturbations = payload, None, None
    else:
        params = results.params
        genes = results.gene_names if isinstance(results, FitResults) else None
        perturbations = results.perturbation_names if isinstance(results, FitResults) else None

    if genes is None:
        genes = tuple(str(i) for i in range(params.mean_w.shape[-1]))
    if perturbations is None:
        perturbations = tuple(str(i) for i in range(params.mean_beta.shape[0]))
    return params, tuple(genes), tuple(perturbations)
