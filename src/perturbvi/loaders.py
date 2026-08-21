from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from scipy import sparse as scipy_sparse

from jax.experimental import sparse as jax_sparse

from .screen import ScreenData, validate_screen


_FORMATS = {
    "auto",
    "anndata",
    "h5ad",
    "10x-h5",
    "10x-mex",
    "csv",
    "tsv",
}
_MISSING_GUIDE_POLICIES = {"error", "unassigned"}


@dataclass(frozen=True)
class _LoadRequest:
    """Validated canonical loader options."""

    format: str
    layer: str
    guide_key: Optional[str]
    guide_obsm: Optional[str]
    control_label: Optional[str]
    missing_guide: str
    expression_feature_type: str
    covariates: Optional[Sequence[str]]
    guide_path: Optional[str]
    metadata_path: Optional[str]
    header: Optional[int]
    index_col: Optional[int]

def _to_internal_matrix(matrix, *, dtype=np.float64):
    """Convert external matrices without densifying sparse inputs."""
    if isinstance(matrix, jax_sparse.JAXSparse):
        return matrix.astype(dtype)
    if scipy_sparse.issparse(matrix):
        return jax_sparse.BCOO.from_scipy_sparse(matrix.astype(dtype).tocoo())
    return np.asarray(matrix, dtype=dtype)


def _stored_values(matrix) -> np.ndarray:
    if isinstance(matrix, jax_sparse.JAXSparse) or scipy_sparse.issparse(matrix):
        return np.asarray(matrix.data)
    return np.asarray(matrix)


def _separator(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _read_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, index_col=0, sep=_separator(path))
    if not frame.index.is_unique:
        raise ValueError(f"Metadata index contains duplicate cell/barcode names: {path}")
    return frame


def _align_metadata(frame: pd.DataFrame, cell_names: Sequence[str], *, label: str) -> pd.DataFrame:
    cells = [str(cell) for cell in cell_names]
    frame = frame.copy()
    frame.index = frame.index.astype(str)
    missing = [cell for cell in cells if cell not in frame.index]
    if missing:
        if label == "covariate file":
            raise ValueError(
                f"{len(missing)} barcodes not found in covariate file. First missing: {missing[:5]}"
            )
        raise ValueError(f"{len(missing)} cells are missing from {label}. First missing: {missing[:5]}")
    return frame.loc[cells]


def _extract_covariates(frame: pd.DataFrame, keys: Sequence[str], *, label: str) -> np.ndarray:
    keys = list(keys)
    if not keys:
        raise ValueError("covariates must contain at least one column name")
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        if label == "adata.obs":
            raise ValueError(
                f"covariates not found in adata.obs: {missing}. Available: {list(frame.columns)}"
            )
        raise ValueError(
            f"Covariate columns not found in {label}: {missing}. Available: {list(frame.columns)}"
        )
    subset = frame[keys]
    if subset.isna().to_numpy().any():
        raise ValueError(f"Covariates in {label} contain missing values")
    numeric = subset.select_dtypes(include="number")
    if not numeric.empty and not np.all(np.isfinite(numeric.to_numpy())):
        raise ValueError(f"Covariates in {label} contain non-finite values")
    return subset.to_numpy()


def _select_external_metadata(
    path: str | Path,
    cell_names: Sequence[str],
) -> pd.DataFrame:
    """Align external metadata and allow it to define a 10x barcode subset."""
    frame = _read_metadata(path)
    frame.index = frame.index.astype(str)
    available = [str(cell) for cell in cell_names]
    available_set = set(available)
    unknown = [cell for cell in frame.index if cell not in available_set]
    if unknown:
        raise ValueError(
            f"{len(unknown)} metadata barcodes are not present in the 10x matrix. "
            f"First unknown: {unknown[:5]}"
        )
    selected = [cell for cell in available if cell in frame.index]
    if not selected:
        raise ValueError("Metadata has no barcodes in common with the 10x matrix")
    return frame.loc[selected]


def _guides_from_labels(
    labels: pd.Series,
    *,
    control_label: Optional[str],
    missing_guide: str,
) -> tuple[np.ndarray, list[str]]:
    if missing_guide not in _MISSING_GUIDE_POLICIES:
        raise ValueError(f"Unknown missing_guide policy '{missing_guide}'")
    if control_label is None:
        raise ValueError("control_label is required when constructing guides from categorical perturbation labels")
    missing = labels.isna()
    if missing.any() and missing_guide == "error":
        raise ValueError(
            f"Perturbation labels are missing for {int(missing.sum())} cells. "
            "Set missing_guide='unassigned' to keep them as all-zero guide rows."
        )

    guide_frame = pd.get_dummies(labels.astype("string"), dummy_na=False, dtype=np.float64)
    guide_frame.columns = guide_frame.columns.astype(str)
    control_label = str(control_label)
    if control_label not in guide_frame.columns:
        raise ValueError(f"control_label '{control_label}' was not present in the perturbation labels")
    guide_frame = guide_frame.drop(columns=[control_label])
    if guide_frame.shape[1] == 0:
        raise ValueError("Guide construction produced no perturbation columns")
    return guide_frame.to_numpy(dtype=np.float64), guide_frame.columns.tolist()


def _screen_from_anndata(
    adata,
    *,
    source_path: Optional[str],
    source_format: str,
    layer: str,
    guide_key: Optional[str],
    guide_obsm: Optional[str],
    control_label: Optional[str],
    covariates: Optional[Sequence[str]],
    missing_guide: str,
    obs: Optional[pd.DataFrame] = None,
) -> ScreenData:
    obs = adata.obs if obs is None else obs
    if layer == "X":
        expr = adata.X
    elif layer in adata.layers:
        expr = adata.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in AnnData. Available layers: {list(adata.layers.keys())}")
    X = _to_internal_matrix(expr)

    if (guide_key is None) == (guide_obsm is None):
        raise ValueError("Provide exactly one of guide_key or guide_obsm for AnnData input")

    if guide_key is not None:
        if guide_key not in obs.columns:
            raise ValueError(
                f"guide_key '{guide_key}' not found in observation metadata. "
                f"Available columns: {list(obs.columns)}"
            )
        guide_values, perturbations = _guides_from_labels(
            obs[guide_key],
            control_label=control_label,
            missing_guide=missing_guide,
        )
        G = _to_internal_matrix(guide_values)
    else:
        if guide_obsm not in adata.obsm:
            raise ValueError(
                f"guide_obsm '{guide_obsm}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
            )
        obsm_value = adata.obsm[guide_obsm]
        perturbations = list(obsm_value.columns.astype(str)) if hasattr(obsm_value, "columns") else None
        if control_label is not None:
            if perturbations is None:
                raise ValueError("control_label requires named guide_obsm columns")
            if str(control_label) not in perturbations:
                raise ValueError(f"control_label '{control_label}' was not present in guide_obsm columns")
            keep = [index for index, name in enumerate(perturbations) if name != str(control_label)]
            obsm_value = obsm_value.iloc[:, keep] if hasattr(obsm_value, "iloc") else obsm_value[:, keep]
            perturbations = [perturbations[index] for index in keep]
        G = _to_internal_matrix(obsm_value)

    covar_data = None
    if covariates is not None:
        covar_data = _extract_covariates(obs, covariates, label="adata.obs")

    screen = ScreenData(
        X=X,
        G=G,
        gene_names=list(adata.var_names.astype(str)),
        perturbation_names=perturbations,
        cell_names=list(adata.obs_names.astype(str)),
        source={
            "path": source_path,
            "format": source_format,
            "layer": layer,
            "guide_key": guide_key,
            "guide_obsm": guide_obsm,
            "control_label": control_label,
            "missing_guide": missing_guide,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
        },
        covariates=covar_data,
        covariate_names=list(covariates) if covariates is not None else None,
    )
    validate_screen(screen)
    return screen


def _load_h5ad(path: Path, **kwargs) -> ScreenData:
    import scanpy as sc

    return _screen_from_anndata(
        sc.read_h5ad(path),
        source_path=str(path),
        source_format="h5ad",
        **kwargs,
    )


def _metadata_10x_assignment(
    expr,
    *,
    metadata_path: str,
    guide_key: str,
    control_label: Optional[str],
    missing_guide: str,
    covariates: Optional[Sequence[str]],
):
    metadata = _select_external_metadata(metadata_path, expr.obs_names.astype(str))
    if guide_key not in metadata.columns:
        raise ValueError(f"guide_key '{guide_key}' not found in metadata columns: {list(metadata.columns)}")
    expr = expr[metadata.index]
    guides, perturbations = _guides_from_labels(
        metadata[guide_key],
        control_label=control_label,
        missing_guide=missing_guide,
    )
    covar_data = (
        _extract_covariates(metadata, covariates, label="metadata file") if covariates is not None else None
    )
    return expr, guides, perturbations, covar_data


def _load_10x(
    adata,
    *,
    expression_feature_type: str,
    source_format: str,
    source_path: str,
    metadata_path: str,
    guide_key: str,
    control_label: Optional[str] = None,
    missing_guide: str = "error",
    covariates: Optional[Sequence[str]] = None,
) -> ScreenData:
    """Load current 10x expression with aligned, confident metadata labels."""
    if "feature_types" not in adata.var.columns:
        raise ValueError(
            "10x input requires a Cell Ranger feature-barcode matrix with a 'feature_types' column "
            "(the three-column features.tsv format for MEX). Convert older genes.tsv matrices to "
            "H5AD before loading them with PerturbVI."
        )
    feature_types = adata.var["feature_types"]
    expr_mask = feature_types == expression_feature_type
    available_types = feature_types.unique().tolist()
    if not expr_mask.any():
        raise ValueError(
            f"No features with feature_type='{expression_feature_type}' found. Available types: {available_types}"
        )
    expr = adata[:, expr_mask]
    n_input = int(adata.n_obs)
    expr, guides, perturbations, covar_data = _metadata_10x_assignment(
        expr,
        metadata_path=metadata_path,
        guide_key=guide_key,
        control_label=control_label,
        missing_guide=missing_guide,
        covariates=covariates,
    )

    screen = ScreenData(
        X=_to_internal_matrix(expr.X),
        G=_to_internal_matrix(guides),
        gene_names=list(expr.var_names.astype(str)),
        perturbation_names=perturbations,
        cell_names=list(expr.obs_names.astype(str)),
        source={
            "path": source_path,
            "format": source_format,
            "expression_feature_type": expression_feature_type,
            "feature_type_source": "column",
            "guide_assignment": "metadata",
            "metadata_path": metadata_path,
            "guide_key": guide_key,
            "control_label": control_label,
            "missing_guide": missing_guide,
            "n_input_obs": n_input,
            "n_obs": int(expr.n_obs),
            "n_exp_vars": int(expr_mask.sum()),
            "n_total_vars": int(adata.n_vars),
        },
        covariates=covar_data,
        covariate_names=list(covariates) if covariates is not None else None,
    )
    validate_screen(screen)
    return screen


def _load_10x_h5(
    path: Path,
    *,
    expression_feature_type: str,
    metadata_path: str,
    guide_key: str,
    control_label: Optional[str],
    missing_guide: str,
    covariates: Optional[Sequence[str]],
) -> ScreenData:
    import scanpy as sc

    adata = sc.read_10x_h5(path, gex_only=False)
    adata.var_names_make_unique()
    return _load_10x(
        adata,
        expression_feature_type=expression_feature_type,
        source_format="10x-h5",
        source_path=str(path),
        metadata_path=metadata_path,
        guide_key=guide_key,
        control_label=control_label,
        missing_guide=missing_guide,
        covariates=covariates,
    )


def _load_10x_mex(
    path: Path,
    *,
    expression_feature_type: str,
    metadata_path: str,
    guide_key: str,
    control_label: Optional[str],
    missing_guide: str,
    covariates: Optional[Sequence[str]],
) -> ScreenData:
    import scanpy as sc

    adata = sc.read_10x_mtx(str(path), var_names="gene_symbols", make_unique=True, gex_only=False)
    return _load_10x(
        adata,
        expression_feature_type=expression_feature_type,
        source_format="10x-mex",
        source_path=str(path),
        metadata_path=metadata_path,
        guide_key=guide_key,
        control_label=control_label,
        missing_guide=missing_guide,
        covariates=covariates,
    )


def _read_matrix_table(path: Path, *, header: Optional[int], index_col: Optional[int]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=_separator(path), header=header, index_col=index_col)
    if not frame.index.is_unique:
        raise ValueError(f"Matrix row names are duplicated in {path}")
    if not frame.columns.is_unique:
        raise ValueError(f"Matrix column names are duplicated in {path}")
    return frame


def _load_delimited(
    path: Path,
    *,
    guide_path: Optional[str],
    metadata_path: Optional[str],
    guide_key: Optional[str],
    control_label: Optional[str],
    covariates: Optional[Sequence[str]],
    missing_guide: str,
    header: Optional[int],
    index_col: Optional[int],
    source_format: str,
) -> ScreenData:
    expression = _read_matrix_table(path, header=header, index_col=index_col)
    try:
        X = expression.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expression CSV/TSV must contain only numeric values") from exc
    cell_names = expression.index.astype(str).tolist()

    if guide_path is not None and guide_key is not None:
        raise ValueError("Provide a guide matrix or perturbation labels, not both")
    if guide_path is None and guide_key is None:
        raise ValueError("Delimited input requires guide_path or guide_key with metadata_path")

    metadata = None
    if metadata_path is not None:
        metadata = _align_metadata(_read_metadata(metadata_path), cell_names, label="metadata file")

    if guide_path is not None:
        guide_frame = _read_matrix_table(Path(guide_path), header=header, index_col=index_col)
        guide_frame.index = guide_frame.index.astype(str)
        if set(guide_frame.index) != set(cell_names):
            raise ValueError("Expression and guide matrix row names do not match")
        guide_frame = guide_frame.loc[cell_names]
        if control_label is not None:
            if str(control_label) not in guide_frame.columns.astype(str):
                raise ValueError(f"control_label '{control_label}' was not present in guide matrix columns")
            guide_frame.columns = guide_frame.columns.astype(str)
            guide_frame = guide_frame.drop(columns=[str(control_label)])
        try:
            G = guide_frame.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Guide CSV/TSV must contain only numeric values") from exc
        perturbations = guide_frame.columns.astype(str).tolist()
    else:
        if metadata is None:
            raise ValueError("metadata_path is required when guide_key is used for delimited input")
        if guide_key not in metadata.columns:
            raise ValueError(f"guide_key '{guide_key}' not found in metadata columns: {list(metadata.columns)}")
        G, perturbations = _guides_from_labels(
            metadata[guide_key],
            control_label=control_label,
            missing_guide=missing_guide,
        )

    covar_data = None
    if covariates is not None:
        if metadata is None:
            raise ValueError("metadata_path is required when covariates are requested for delimited input")
        covar_data = _extract_covariates(metadata, covariates, label="metadata file")

    screen = ScreenData(
        X=X,
        G=G,
        gene_names=expression.columns.astype(str).tolist(),
        perturbation_names=perturbations,
        cell_names=cell_names,
        source={
            "path": str(path),
            "format": source_format,
            "guide_path": guide_path,
            "metadata_path": metadata_path,
            "guide_key": guide_key,
            "control_label": control_label,
            "missing_guide": missing_guide,
            "header": header,
            "index_col": index_col,
        },
        covariates=covar_data,
        covariate_names=list(covariates) if covariates is not None else None,
    )
    validate_screen(screen)
    return screen


def _canonical_request(
    *,
    format: str,
    layer: str,
    guide_key: Optional[str],
    guide_obsm: Optional[str],
    control_label: Optional[str],
    missing_guide: str,
    expression_feature_type: str,
    covariates: Optional[Sequence[str]],
    guide_path: Optional[str],
    metadata_path: Optional[str],
    header: Optional[int],
    index_col: Optional[int],
) -> _LoadRequest:
    if format not in _FORMATS:
        raise ValueError(f"Unknown format '{format}'. Choose from: {sorted(_FORMATS)}")
    if missing_guide not in _MISSING_GUIDE_POLICIES:
        raise ValueError(f"Unknown missing_guide policy '{missing_guide}'")
    return _LoadRequest(
        format=format,
        layer=layer,
        guide_key=guide_key,
        guide_obsm=guide_obsm,
        control_label=control_label,
        missing_guide=missing_guide,
        expression_feature_type=expression_feature_type,
        covariates=covariates,
        guide_path=guide_path,
        metadata_path=metadata_path,
        header=header,
        index_col=index_col,
    )


def _detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    suffix_formats = {
        ".h5ad": "h5ad",
        ".h5": "10x-h5",
        ".hdf5": "10x-h5",
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "tsv",
    }
    if suffix in suffix_formats:
        return suffix_formats[suffix]
    if path.is_dir():
        return "10x-mex"
    raise ValueError(
        f"Cannot auto-detect format from '{path}'. Specify h5ad, 10x-h5, 10x-mex, csv, or tsv."
    )


def _load_h5ad_request(path: Path, request: _LoadRequest) -> ScreenData:
    if request.metadata_path is not None or request.guide_path is not None:
        raise ValueError("metadata_path is not supported for h5ad; metadata must come from adata.obs/obsm")
    if (request.guide_key is None) == (request.guide_obsm is None):
        raise ValueError("Provide exactly one of guide_key or guide_obsm for .h5ad input")
    if not path.is_file():
        raise FileNotFoundError(f"Input h5ad file does not exist: {path}")
    return _load_h5ad(
        path,
        layer=request.layer,
        guide_key=request.guide_key,
        guide_obsm=request.guide_obsm,
        control_label=request.control_label,
        covariates=request.covariates,
        missing_guide=request.missing_guide,
    )


def _load_10x_request(path: Path, request: _LoadRequest) -> ScreenData:
    if request.guide_obsm is not None or request.guide_path is not None:
        raise ValueError("guide_obsm and guide_path are not applicable to 10x input")
    if request.layer != "X":
        raise ValueError("layer= is only applicable to h5ad input")
    if request.metadata_path is None:
        raise ValueError("10x input requires barcode-indexed metadata with one perturbation label per cell")
    if request.guide_key is None:
        raise ValueError("guide_key is required with metadata_path for 10x input")

    loader = _load_10x_h5 if request.format == "10x-h5" else _load_10x_mex
    exists = path.is_file() if request.format == "10x-h5" else path.is_dir()
    if not exists:
        label = "H5 file" if request.format == "10x-h5" else "MEX directory"
        raise FileNotFoundError(f"Input 10x {label} does not exist: {path}")
    return loader(
        path,
        expression_feature_type=request.expression_feature_type,
        metadata_path=request.metadata_path,
        guide_key=request.guide_key,
        control_label=request.control_label,
        missing_guide=request.missing_guide,
        covariates=request.covariates,
    )


def _load_delimited_request(path: Path, request: _LoadRequest) -> ScreenData:
    if request.guide_path is not None and request.guide_key is not None:
        raise ValueError("Provide a guide matrix or perturbation labels, not both")
    if request.guide_path is None and request.guide_key is None:
        raise ValueError("Delimited input requires guide_path or guide_key with metadata_path")
    if request.guide_key is not None and request.metadata_path is None:
        raise ValueError("metadata_path is required when guide_key is used for delimited input")
    if request.covariates is not None and request.metadata_path is None:
        raise ValueError("metadata_path is required when covariates are requested for delimited input")
    if not path.is_file():
        raise FileNotFoundError(f"Input expression table does not exist: {path}")
    return _load_delimited(
        path,
        guide_path=request.guide_path,
        metadata_path=request.metadata_path,
        guide_key=request.guide_key,
        control_label=request.control_label,
        covariates=request.covariates,
        missing_guide=request.missing_guide,
        header=request.header,
        index_col=request.index_col,
        source_format=request.format,
    )


def load_screen(
    path: Any,
    *,
    format: str = "auto",
    layer: str = "X",
    guide_key: Optional[str] = None,
    guide_obsm: Optional[str] = None,
    control_label: Optional[str] = None,
    missing_guide: str = "error",
    expression_feature_type: str = "Gene Expression",
    covariates: Optional[Sequence[str]] = None,
    guide_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    header: Optional[int] = 0,
    index_col: Optional[int] = 0,
) -> ScreenData:
    """Load and validate a perturbation screen from a supported file format.

    AnnData accepts exactly one of ``guide_key`` or ``guide_obsm``. 10x H5/MEX
    requires one barcode-indexed ``metadata_path`` with one perturbation label
    per cell in ``guide_key`` and any requested covariates. The metadata rows may
    define an analyzed barcode subset. Small CSV/TSV expression tables accept
    either ``guide_path`` or a ``guide_key`` in ``metadata_path``.
    Sparse AnnData and 10x expression matrices remain JAX sparse.

    Args:
        path: Input file, 10x MEX directory, or AnnData object.
        format: ``auto``, ``anndata``, ``h5ad``, ``10x-h5``, ``10x-mex``,
            ``csv``, or ``tsv``.
        layer: AnnData expression source; ``X`` selects ``adata.X``.
        guide_key: Per-cell perturbation-label column in AnnData obs or metadata.
        guide_obsm: AnnData obsm key containing an existing guide matrix.
        control_label: Control category to exclude. Required with ``guide_key``;
            optional for a prepared guide matrix, where controls may already be
            represented by all-zero rows.
        missing_guide: Reject missing labels with ``error`` (default), or keep
            them as all-zero rows with ``unassigned``.
        expression_feature_type: 10x expression feature type.
        covariates: Metadata columns to retain for later residualization.
        guide_path: Cell-by-guide CSV/TSV used with delimited expression.
        metadata_path: Cell-indexed CSV/TSV containing perturbation labels and/or
            covariates. For 10x input, its rows may select a barcode subset.
        header: Header row passed to delimited matrix readers.
        index_col: Cell-name column passed to delimited matrix readers.

    Returns:
        A validated :class:`ScreenData` with aligned names and matrices.
    """
    request = _canonical_request(
        format=format,
        layer=layer,
        guide_key=guide_key,
        guide_obsm=guide_obsm,
        control_label=control_label,
        missing_guide=missing_guide,
        expression_feature_type=expression_feature_type,
        covariates=covariates,
        guide_path=guide_path,
        metadata_path=metadata_path,
        header=header,
        index_col=index_col,
    )

    try:
        import anndata as ad
    except ImportError:  # pragma: no cover - scanpy requires anndata
        ad = None
    if ad is not None and isinstance(path, ad.AnnData):
        if request.format not in {"auto", "anndata", "h5ad"}:
            raise ValueError(f"AnnData object is incompatible with format='{request.format}'")
        if request.metadata_path is not None or request.guide_path is not None:
            raise ValueError("metadata_path is not supported for AnnData; metadata must come from adata.obs/obsm")
        if (request.guide_key is None) == (request.guide_obsm is None):
            raise ValueError("Provide exactly one of guide_key or guide_obsm for AnnData input")
        return _screen_from_anndata(
            path,
            source_path=None,
            source_format="anndata",
            layer=request.layer,
            guide_key=request.guide_key,
            guide_obsm=request.guide_obsm,
            control_label=request.control_label,
            covariates=request.covariates,
            missing_guide=request.missing_guide,
        )

    if request.format == "anndata":
        raise TypeError(f"format='{request.format}' requires an in-memory {request.format.title()} object")

    input_path = Path(path)
    request = replace(request, format=_detect_format(input_path, request.format))
    if request.format == "h5ad":
        return _load_h5ad_request(input_path, request)
    if request.format in {"10x-h5", "10x-mex"}:
        return _load_10x_request(input_path, request)
    return _load_delimited_request(input_path, request)
