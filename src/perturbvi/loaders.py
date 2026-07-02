import warnings

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .screen import ScreenData, validate_screen


def _to_dense(mat) -> np.ndarray:
    if hasattr(mat, "toarray"):
        return mat.toarray()
    return np.asarray(mat)


def _load_covariate_file(
    path: str,
    barcodes: Sequence[str],
    covariate_keys: Sequence[str],
) -> np.ndarray:
    """Load covariates from a barcode-indexed CSV/TSV and align to barcodes."""
    p = Path(path)
    sep = "\t" if p.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(p, index_col=0, sep=sep)

    missing_cols = [k for k in covariate_keys if k not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Covariate columns not found in file: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )

    missing_barcodes = [b for b in barcodes if b not in df.index]
    if missing_barcodes:
        raise ValueError(
            f"{len(missing_barcodes)} barcodes not found in covariate file. "
            f"First missing: {missing_barcodes[:5]}"
        )

    subset = df.loc[list(barcodes), list(covariate_keys)]
    numeric_cols = subset.select_dtypes(include="number")
    if not numeric_cols.empty and not np.all(np.isfinite(numeric_cols.values)):
        raise ValueError("Covariate file contains non-finite values.")
    return subset.values


def _load_h5ad(
    path: Path,
    *,
    layer: str,
    guide_key: Optional[str],
    guide_obsm: Optional[str],
    control_label: Optional[str],
    covariates: Optional[Sequence[str]],
) -> ScreenData:
    import scanpy as sc

    adata = sc.read_h5ad(path)

    if layer == "X":
        X = _to_dense(adata.X).astype(np.float64)
    else:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found in AnnData. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        X = _to_dense(adata.layers[layer]).astype(np.float64)

    if guide_key is not None and guide_obsm is not None:
        raise ValueError("Provide exactly one of guide_key or guide_obsm, not both.")
    if guide_key is None and guide_obsm is None:
        raise ValueError("One of guide_key or guide_obsm is required for .h5ad input.")

    if guide_key is not None:
        if guide_key not in adata.obs.columns:
            raise ValueError(
                f"guide_key '{guide_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        G_df = pd.get_dummies(adata.obs[guide_key].astype(str))
        if control_label is not None:
            G_df = G_df.drop(columns=[str(control_label)], errors="ignore")
        G = G_df.values.astype(np.float64)
        perturbation_names = list(G_df.columns)
    else:
        if guide_obsm not in adata.obsm:
            raise ValueError(
                f"guide_obsm '{guide_obsm}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        obsm_val = adata.obsm[guide_obsm]
        G = _to_dense(obsm_val).astype(np.float64)
        perturbation_names = list(obsm_val.columns) if hasattr(obsm_val, "columns") else None

    cov_matrix = None
    if covariates is not None:
        missing = [k for k in covariates if k not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"covariates not found in adata.obs: {missing}. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        cov_df = adata.obs[list(covariates)]
        numeric_cols = cov_df.select_dtypes(include="number")
        if not numeric_cols.empty and not np.all(np.isfinite(numeric_cols.values)):
            raise ValueError("Covariate columns contain non-finite values.")
        cov_matrix = cov_df.values

    source = {
        "path": str(path),
        "format": "h5ad",
        "layer": layer,
        "guide_key": guide_key,
        "guide_obsm": guide_obsm,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }

    screen = ScreenData(
        X=X,
        G=G,
        gene_names=list(adata.var_names),
        perturbation_names=perturbation_names,
        cell_names=list(adata.obs_names),
        source=source,
        covariates=cov_matrix,
        covariate_names=list(covariates) if covariates is not None else None,
    )
    validate_screen(screen)
    return screen


def _load_10x(
    adata,
    *,
    expression_feature_type: str,
    guide_feature_type: str,
    guide_threshold: int,
    multi_guide: str,
    source_format: str,
    source_path: str,
    covariate_file: Optional[str],
    covariate_keys: Optional[Sequence[str]],
) -> ScreenData:
    """Shared logic for 10x H5 and MEX: split features, binarize guides, validate."""
    if "feature_types" not in adata.var.columns:
        raise ValueError(
            "adata.var does not contain a 'feature_types' column. "
            "Make sure the file contains CRISPR feature-barcode data from Cell Ranger 3+."
        )

    feature_types = adata.var["feature_types"]
    exp_mask = feature_types == expression_feature_type
    guide_mask = feature_types == guide_feature_type

    if not exp_mask.any():
        raise ValueError(
            f"No features with feature_type='{expression_feature_type}' found. "
            f"Available types: {feature_types.unique().tolist()}"
        )
    if not guide_mask.any():
        raise ValueError(
            f"No features with feature_type='{guide_feature_type}' found. "
            f"Available types: {feature_types.unique().tolist()}"
        )

    adata_exp = adata[:, exp_mask]
    adata_guide = adata[:, guide_mask]

    X = _to_dense(adata_exp.X).astype(np.float64)
    guide_counts = _to_dense(adata_guide.X)
    G = (guide_counts >= guide_threshold).astype(np.float64)

    n_multi = int((G.sum(axis=1) > 1).sum())
    if n_multi > 0:
        if multi_guide == "error":
            raise ValueError(
                f"{n_multi} cells have more than one guide assignment above threshold {guide_threshold}. "
                "Set multi_guide='warn' or 'allow' to proceed."
            )
        if multi_guide == "warn":
            warnings.warn(
                f"{n_multi} cells have more than one guide assignment above threshold {guide_threshold}.",
                stacklevel=3,
            )

    cov_matrix = None
    if covariate_file is not None:
        if covariate_keys is None:
            raise ValueError("covariates= must be provided alongside covariate_file=.")
        cov_matrix = _load_covariate_file(covariate_file, list(adata.obs_names), covariate_keys)

    source = {
        "path": source_path,
        "format": source_format,
        "expression_feature_type": expression_feature_type,
        "guide_feature_type": guide_feature_type,
        "guide_threshold": guide_threshold,
        "n_obs": int(adata.n_obs),
        "n_exp_vars": int(exp_mask.sum()),
        "n_guide_vars": int(guide_mask.sum()),
    }

    screen = ScreenData(
        X=X,
        G=G,
        gene_names=list(adata_exp.var_names),
        perturbation_names=list(adata_guide.var_names),
        cell_names=list(adata.obs_names),
        source=source,
        covariates=cov_matrix,
        covariate_names=list(covariate_keys) if covariate_keys is not None else None,
    )
    validate_screen(screen)
    return screen


def _load_10x_h5(
    path: Path,
    *,
    expression_feature_type: str,
    guide_feature_type: str,
    guide_threshold: int,
    multi_guide: str,
    covariate_file: Optional[str],
    covariate_keys: Optional[Sequence[str]],
) -> ScreenData:
    import scanpy as sc

    adata = sc.read_10x_h5(path)
    return _load_10x(
        adata,
        expression_feature_type=expression_feature_type,
        guide_feature_type=guide_feature_type,
        guide_threshold=guide_threshold,
        multi_guide=multi_guide,
        source_format="10x-h5",
        source_path=str(path),
        covariate_file=covariate_file,
        covariate_keys=covariate_keys,
    )


def _load_10x_mex(
    path: Path,
    *,
    expression_feature_type: str,
    guide_feature_type: str,
    guide_threshold: int,
    multi_guide: str,
    covariate_file: Optional[str],
    covariate_keys: Optional[Sequence[str]],
) -> ScreenData:
    import scanpy as sc

    adata = sc.read_10x_mtx(str(path), var_names="gene_ids", make_unique=True)
    return _load_10x(
        adata,
        expression_feature_type=expression_feature_type,
        guide_feature_type=guide_feature_type,
        guide_threshold=guide_threshold,
        multi_guide=multi_guide,
        source_format="10x-mex",
        source_path=str(path),
        covariate_file=covariate_file,
        covariate_keys=covariate_keys,
    )


def load_screen(
    path: str,
    *,
    format: str = "auto",
    layer: str = "X",
    guide_key: Optional[str] = None,
    guide_obsm: Optional[str] = None,
    control_label: Optional[str] = None,
    guide_threshold: int = 1,
    expression_feature_type: str = "Gene Expression",
    guide_feature_type: str = "CRISPR Guide Capture",
    covariates: Optional[Sequence[str]] = None,
    covariate_file: Optional[str] = None,
    multi_guide: str = "warn",
) -> ScreenData:
    """Load a perturbation screen from disk into a validated ScreenData object.

    Args:
        path: Path to input file or directory.
        format: One of 'auto', 'h5ad', '10x-h5', '10x-mex'. 'auto' detects by extension.
        layer: Expression layer in .h5ad (default 'X' uses adata.X).
        guide_key: adata.obs column to build one-hot G from (h5ad only).
        guide_obsm: adata.obsm key for an existing guide matrix (h5ad only).
        control_label: Label to drop from perturbation columns when using guide_key.
        guide_threshold: UMI count threshold to binarize guide counts (10x only).
        expression_feature_type: Feature type string for expression (10x only).
        guide_feature_type: Feature type string for guide capture (10x only).
        covariates: For h5ad: adata.obs column names to extract as covariates.
                    For 10x: column names to read from covariate_file.
        covariate_file: Path to a barcode-indexed CSV/TSV of covariates (10x only).
        multi_guide: Policy for cells with >1 guide: 'warn' (default), 'allow', or 'error'.

    Returns:
        Validated ScreenData.
    """
    path = Path(path)

    if format == "auto":
        if path.suffix == ".h5ad":
            format = "h5ad"
        elif path.suffix == ".h5":
            format = "10x-h5"
        elif path.is_dir():
            format = "10x-mex"
        else:
            raise ValueError(
                f"Cannot auto-detect format from '{path}'. "
                "Specify format explicitly: h5ad, 10x-h5, or 10x-mex."
            )

    if format == "h5ad":
        if covariate_file is not None:
            raise ValueError(
                "covariate_file= is not supported for h5ad. "
                "Use covariates= to extract columns from adata.obs."
            )
        return _load_h5ad(
            path,
            layer=layer,
            guide_key=guide_key,
            guide_obsm=guide_obsm,
            control_label=control_label,
            covariates=covariates,
        )

    if covariates is not None and covariate_file is None:
        raise ValueError(
            "For 10x formats, provide covariates via covariate_file= (a barcode-indexed CSV/TSV) "
            "alongside covariates= (column names). covariates= alone is not supported for 10x."
        )

    if format == "10x-h5":
        return _load_10x_h5(
            path,
            expression_feature_type=expression_feature_type,
            guide_feature_type=guide_feature_type,
            guide_threshold=guide_threshold,
            multi_guide=multi_guide,
            covariate_file=covariate_file,
            covariate_keys=covariates,
        )

    if format == "10x-mex":
        return _load_10x_mex(
            path,
            expression_feature_type=expression_feature_type,
            guide_feature_type=guide_feature_type,
            guide_threshold=guide_threshold,
            multi_guide=multi_guide,
            covariate_file=covariate_file,
            covariate_keys=covariates,
        )

    raise ValueError(f"Unknown format '{format}'. Choose from: auto, h5ad, 10x-h5, 10x-mex.")
