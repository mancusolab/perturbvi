from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

from .log import get_logger


_FIT_EPILOG = """Input recipes:
  H5AD with labels:
    perturbvi fit screen.h5ad --perturbation-column perturbation --control-label control ...

  H5AD with a guide matrix in obsm:
    perturbvi fit screen.h5ad --guide-matrix-key guide_matrix ...

  10x H5 or MEX with cell metadata:
    perturbvi fit matrix.h5 --cell-metadata cells.csv --perturbation-column perturbation --control-label control ...

  CSV/TSV expression with a separate guide matrix:
    perturbvi fit expression.csv --guide-matrix guides.csv ...

Only use the input options relevant to the selected file type.
"""


def _optional_integer(value: str):
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer or 'none'") from exc


def _setup_jax(device: str) -> str:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    try:
        devices = jax.devices(device)
    except RuntimeError as exc:
        raise RuntimeError(f"Requested JAX device '{device}' is not available") from exc
    if not devices:
        raise RuntimeError(f"Requested JAX device '{device}' is not available")
    jax.config.update("jax_default_device", devices[0])
    return str(devices[0])


def _read_names(path: str) -> list[str]:
    import pandas as pd

    names_path = Path(path)
    if not names_path.is_file():
        raise FileNotFoundError(f"Name file does not exist: {names_path}")
    if names_path.suffix.lower() == ".csv":
        names = pd.read_csv(names_path, header=None).iloc[:, 0].astype(str).tolist()
    else:
        names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(names)) != len(names):
        raise ValueError(f"Name file contains duplicates: {names_path}")
    return names


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    required = parser.add_argument_group("required fit arguments")
    required.add_argument(
        "input",
        help="Path to .h5ad, 10x H5/MEX, or a small CSV/TSV expression matrix",
    )
    required.add_argument("--output", required=True, help="Results output directory")
    required.add_argument("--z-dim", type=int, required=True, help="Number of latent gene programs")
    required.add_argument("--l-dim", type=int, required=True, help="Number of single effects per program")
    required.add_argument("--tau", type=float, required=True, help="Positive initial residual precision")

    assignments = parser.add_argument_group("perturbation assignments")
    assignments.add_argument(
        "--perturbation-column",
        default=None,
        help="Column containing one perturbation label per cell in AnnData obs or cell metadata",
    )
    assignments.add_argument(
        "--guide-matrix-key",
        default=None,
        help="AnnData obsm key containing an existing cell-by-perturbation matrix",
    )
    assignments.add_argument(
        "--guide-matrix",
        dest="guide_matrix_path",
        default=None,
        help="Cell-by-perturbation CSV/TSV paired with CSV/TSV expression",
    )
    assignments.add_argument(
        "--cell-metadata",
        dest="cell_metadata_path",
        default=None,
        help="Cell/barcode-indexed CSV/TSV used with 10x or delimited expression",
    )
    assignments.add_argument(
        "--control-label",
        default=None,
        help="Control category to exclude; required with --perturbation-column",
    )

    preprocessing = parser.add_argument_group("preprocessing")
    preprocessing.add_argument(
        "--covariates",
        nargs="+",
        default=None,
        help="AnnData obs or cell-metadata columns to residualize before fitting",
    )
    preprocessing.add_argument(
        "--categorical-covariates",
        dest="categorical_covariates",
        nargs="+",
        default=None,
        help="Subset of covariates treated as categorical",
    )
    preprocessing.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale centered expression to unit variance (default: on)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--p-prior", type=float, default=0.1, help="Prior inclusion probability (default: 0.1)")
    model.add_argument("--init", default="random", choices=["random", "pca"], help="Factor initialization")
    model.add_argument("--max-iter", type=int, default=500, help="Maximum inference iterations (default: 500)")
    model.add_argument("--tol", type=float, default=1e-2, help="ELBO convergence tolerance (default: 0.01)")

    advanced = parser.add_argument_group("advanced input")
    advanced.add_argument(
        "--format",
        default="auto",
        choices=["auto", "h5ad", "10x-h5", "10x-mex", "csv", "tsv"],
        help="Input format (default: auto-detect)",
    )
    advanced.add_argument(
        "--expression-layer",
        default="X",
        help="AnnData expression source: X or a key in adata.layers (default: X)",
    )
    advanced.add_argument(
        "--expression-feature-type",
        default="Gene Expression",
        help="10x feature type used as expression (default: Gene Expression)",
    )
    advanced.add_argument(
        "--header",
        type=_optional_integer,
        default=0,
        help="CSV/TSV header-row index, or 'none' (default: 0)",
    )
    advanced.add_argument(
        "--index-col",
        type=_optional_integer,
        default=0,
        help="CSV/TSV cell-ID column index, or 'none' (default: 0)",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    runtime.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="JAX device (default: cpu)")
    runtime.add_argument("--verbose", action="store_true", default=False, help="Show inference progress")


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
    required = parser.add_argument_group("required analysis argument")
    required.add_argument("results_dir", help="Directory produced by perturbvi fit or save_results()")

    labels = parser.add_argument_group("optional label overrides")
    labels.add_argument(
        "--gene-names",
        dest="gene_names",
        default=None,
        help="Optional one-name-per-row file overriding saved gene labels",
    )
    labels.add_argument(
        "--perturbation-names",
        dest="perturbation_names",
        default=None,
        help="Optional one-name-per-row file overriding saved perturbation labels",
    )
    thresholds = parser.add_argument_group("significance thresholds")
    thresholds.add_argument(
        "--pip-threshold",
        type=float,
        default=0.9,
        help="PIP cutoff; significant values are at least this value (default: 0.9)",
    )
    thresholds.add_argument(
        "--lfsr-threshold",
        type=float,
        default=0.05,
        help="LFSR cutoff; significant values are at most this value (default: 0.05)",
    )
    lfsr = parser.add_argument_group("LFSR computation")
    lfsr.add_argument(
        "--compute-lfsr",
        action="store_true",
        default=False,
        help="Recompute LFSR and replace lfsr.csv in the fit directory",
    )
    lfsr.add_argument("--lfsr-iters", type=int, default=2000, help="Monte Carlo iterations (default: 2000)")
    lfsr.add_argument("--seed", type=int, default=0, help="LFSR random seed (default: 0)")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_fit_metadata(args, screen, output: Path, selected_device: str) -> None:
    preprocessing_steps = []
    if args.covariates:
        preprocessing_steps.append("residualize")
    preprocessing_steps.append("center")
    if args.standardize:
        preprocessing_steps.append("scale")
    run_config = {
        "input": args.input,
        "format": screen.source.get("format", args.format),
        "output": str(output),
        "z_dim": args.z_dim,
        "l_dim": args.l_dim,
        "tau": args.tau,
        "expression_layer": args.expression_layer,
        "perturbation_column": args.perturbation_column,
        "guide_matrix_key": args.guide_matrix_key,
        "guide_matrix": args.guide_matrix_path,
        "cell_metadata": args.cell_metadata_path,
        "control_label": args.control_label,
        "expression_feature_type": args.expression_feature_type,
        "covariates": args.covariates,
        "categorical_covariates": args.categorical_covariates,
        "header": args.header,
        "index_col": args.index_col,
        "p_prior": args.p_prior,
        "standardize": args.standardize,
        "preprocessing_steps": preprocessing_steps,
        "preprocessing_order": "_then_".join(preprocessing_steps) if preprocessing_steps else "none",
        "init": args.init,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "device": args.device,
        "selected_device": selected_device,
        "verbose": args.verbose,
    }
    input_summary = {
        "source": dict(screen.source),
        "X_shape": list(screen.X.shape),
        "G_shape": list(screen.G.shape),
        "n_genes": len(screen.gene_names) if screen.gene_names is not None else None,
        "n_perturbations": len(screen.perturbation_names) if screen.perturbation_names is not None else None,
        "n_cells": len(screen.cell_names) if screen.cell_names is not None else None,
        "gene_names": [str(name) for name in screen.gene_names] if screen.gene_names is not None else None,
        "perturbation_names": (
            [str(name) for name in screen.perturbation_names]
            if screen.perturbation_names is not None
            else None
        ),
    }
    for filename, payload in {"run_config.json": run_config, "input_summary.json": input_summary}.items():
        (output / filename).write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )


def _cmd_fit(args, log, selected_device: str) -> None:
    from perturbvi import fit_screen, load_screen, residualize_screen, save_results

    log.info(f"Loading screen from: {args.input}")
    screen = load_screen(
        args.input,
        format=args.format,
        layer=args.expression_layer,
        guide_key=args.perturbation_column,
        guide_obsm=args.guide_matrix_key,
        guide_path=args.guide_matrix_path,
        metadata_path=args.cell_metadata_path,
        control_label=args.control_label,
        expression_feature_type=args.expression_feature_type,
        covariates=args.covariates,
        header=args.header,
        index_col=args.index_col,
    )
    log.info(f"Loaded: X={screen.X.shape}, G={screen.G.shape}")

    if args.covariates:
        log.info(f"Residualizing covariates: {args.covariates}")
        screen = residualize_screen(
            screen,
            covariates=args.covariates,
            categorical_covariates=args.categorical_covariates,
        )

    log.info("Starting inference")
    results = fit_screen(
        screen,
        z_dim=args.z_dim,
        l_dim=args.l_dim,
        tau=args.tau,
        p_prior=args.p_prior,
        standardize=args.standardize,
        init=args.init,
        tol=args.tol,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=args.verbose,
    )

    output = Path(args.output)
    save_results(
        results,
        path=str(output),
    )
    _write_fit_metadata(args, screen, output, selected_device)
    log.info(f"Results and reproducibility metadata saved to {output}")


def _cmd_analyze(args, log) -> None:
    from perturbvi import analyze

    results_dir = Path(args.results_dir)
    gene_names = _read_names(args.gene_names) if args.gene_names else None
    perturbation_names = _read_names(args.perturbation_names) if args.perturbation_names else None
    tables = analyze(
        str(results_dir),
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        pip_threshold=args.pip_threshold,
        lfsr_threshold=args.lfsr_threshold,
        compute_lfsr=args.compute_lfsr,
        lfsr_iters=args.lfsr_iters,
        seed=args.seed,
    )
    for name, table in tables.items():
        if table is None:
            continue
        destination = results_dir / f"{name}.csv"
        table.to_csv(destination)
        log.info(f"Saved {destination}")
    log.info(f"Analysis complete: {results_dir}")


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="perturbvi",
        description="PerturbVI: variational inference for single-cell Perturb-seq data",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit PerturbVI from a screen file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_FIT_EPILOG,
        allow_abbrev=False,
    )
    _add_fit_args(fit_parser)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze saved PerturbVI results",
        allow_abbrev=False,
    )
    _add_analyze_args(analyze_parser)
    parsed = parser.parse_args(args)
    if parsed.command == "fit" and parsed.categorical_covariates and not parsed.covariates:
        parser.error("--categorical-covariates requires --covariates")

    log = get_logger(__name__)
    log.setLevel(logging.DEBUG if getattr(parsed, "verbose", False) else logging.INFO)
    selected_device = _setup_jax(getattr(parsed, "device", "cpu"))
    if parsed.command == "fit":
        _cmd_fit(parsed, log, selected_device)
    else:
        _cmd_analyze(parsed, log)


def run_cli():
    main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
