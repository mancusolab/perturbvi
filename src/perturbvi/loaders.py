from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from scipy import sparse as scipy_sparse

from jax.experimental import sparse as jax_sparse

from .screen import PerturbData, validate_screen


def _read_anndata(source: Any):
    import anndata as ad

    if isinstance(source, ad.AnnData):
        return source
    if not isinstance(source, (str, Path)):
        raise TypeError("source must be an AnnData object or an H5AD/Zarr path")

    path = Path(source)
    if path.suffix.lower() == ".h5ad":
        return ad.read_h5ad(path)
    if path.suffix.lower() == ".zarr" or path.is_dir():
        return ad.read_zarr(path)
    raise ValueError(f"Unsupported input '{path}'. Expected an AnnData object, .h5ad file, or .zarr store")


def _columns(values: Optional[Sequence[str]], name: str) -> Optional[list[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    columns = [str(value) for value in values]
    if not columns:
        raise ValueError(f"{name} must contain at least one column")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} contains duplicate columns")
    return columns


def _matrix(value):
    if isinstance(value, jax_sparse.JAXSparse):
        return value.astype(np.float64)
    if scipy_sparse.issparse(value):
        return jax_sparse.BCOO.from_scipy_sparse(value.astype(np.float64).tocoo())
    return np.asarray(value, dtype=np.float64)


def load_screen(
    source: Any,
    *,
    x_key: Optional[str] = None,
    covariates: Optional[Sequence[str]] = None,
    g_key: str = "G",
    control: Optional[str] = None,
) -> PerturbData:
    """Build :class:`PerturbData` from AnnData.

    Expression comes from ``adata.X`` unless ``x_key`` names a layer. The
    binary perturbation matrix is read from ``adata.obsm[g_key]`` (default
    ``"G"``) and must be a named pandas DataFrame with one row per cell.
    Column order is preserved exactly: ``perturbation_names[i]`` labels
    ``G[:, i]``.
    ``control`` names a column of that frame to drop as the reference; pass
    ``None`` when the stored G is already baseline-free.
    """
    adata = _read_anndata(source)
    if x_key is None:
        expression = adata.X
        if expression is None:
            raise ValueError("adata.X is empty; pass x_key=... to select expression")
    else:
        if x_key not in adata.layers:
            raise KeyError(f"AnnData layer '{x_key}' does not exist")
        expression = adata.layers[x_key]

    if g_key not in adata.obsm:
        raise ValueError(
            f"no G provided and adata.obsm[{g_key!r}] does not exist; "
            f"store your binary perturbation matrix at obsm[{g_key!r}] "
            "(or pass g_key= pointing at it)"
        )
    G_frame = adata.obsm[g_key]
    if not isinstance(G_frame, pd.DataFrame) or G_frame.columns.empty:
        raise ValueError(
            f"adata.obsm[{g_key!r}] must be a named pandas DataFrame of binary columns"
        )
    if G_frame.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm[{g_key!r}] has {G_frame.shape[0]} rows but the expression "
            f"matrix has {adata.n_obs} cells"
        )
    if control is not None:
        control = str(control)
        if control not in G_frame.columns:
            raise ValueError(
                f"control column {control!r} is not present in adata.obsm[{g_key!r}]; "
                "pass control=None when G is already baseline-free"
            )
        G_frame = G_frame.drop(columns=control)
    perturbation_names = tuple(str(name) for name in G_frame.columns)

    covariate_names = _columns(covariates, "covariates")
    covariate_frame = None
    if covariate_names is not None:
        missing = [name for name in covariate_names if name not in adata.obs]
        if missing:
            raise KeyError(f"AnnData covariate columns do not exist: {missing}")
        covariate_frame = adata.obs[covariate_names].copy()

    data = PerturbData(
        X=_matrix(expression),
        G=_matrix(G_frame),
        gene_names=tuple(str(name) for name in adata.var_names),
        perturbation_names=perturbation_names,
        covariates=covariate_frame,
    )
    validate_screen(data)
    return data


__all__ = ["load_screen"]
