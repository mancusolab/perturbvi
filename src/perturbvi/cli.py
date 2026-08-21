from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

from .log import get_logger


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
    parser.add_argument(
        "input",
        help="Path to .h5ad, 10x H5/MEX, or a small CSV/TSV expression matrix",
    )
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "h5ad", "10x-h5", "10x-mex", "csv", "tsv"],
        help="Input format (default: auto-detect)",
    )
    parser.add_argument("--output", required=True, help="Results output directory")
    parser.add_argument("--z-dim", type=int, required=True, help="Number of latent factors")
    parser.add_argument("--l-dim", type=int, required=True, help="Number of single effects per factor")
    parser.add_argument("--tau", type=float, required=True, help="Positive residual precision initial value")
    parser.add_argument("--layer", default="X", help="Expression layer for .h5ad input")
    parser.add_argument("--guide-key", default=None, help="Observation/metadata column used to construct guides")
    parser.add_argument("--guide-obsm", default=None, help="AnnData obsm key containing an existing guide matrix")
    parser.add_argument("--guide-matrix", dest="guide_path", default=None, help="Guide CSV/TSV for delimited input")
    parser.add_argument(
        "--metadata",
        dest="metadata_path",
        default=None,
        help="Cell/barcode-indexed labels and covariates for delimited or 10x input",
    )
    parser.add_argument("--control-label", default=None, help="Control label/column to exclude from G")
    parser.add_argument(
        "--missing-guide",
        choices=["error", "unassigned"],
        default="error",
        help="Policy for missing guide labels (default: error)",
    )
    parser.add_argument("--expression-feature-type", default="Gene Expression")
    parser.add_argument(
        "--covariates",
        nargs="+",
        default=None,
        help="Covariates from obs/metadata, residualized before standardization",
    )
    parser.add_argument(
        "--categorical-covariates",
        dest="categorical_covariates",
        nargs="+",
        default=None,
        help="Subset of covariates treated as categorical",
    )
    parser.add_argument(
        "--header",
        type=_optional_integer,
        default=0,
        help="Header row for delimited input, or 'none' (default: 0)",
    )
    parser.add_argument(
        "--index-col",
        type=_optional_integer,
        default=0,
        help="Cell-name column for delimited input, or 'none' (default: 0)",
    )
    parser.add_argument("--p-prior", type=float, default=0.1)
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize expression after optional residualization (default: on)",
    )
    parser.add_argument("--init", default="random", choices=["random", "pca"], help="Factor initialization")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--verbose", action="store_true", default=False)


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("results_dir", help="Directory produced by perturbvi fit or save_results()")
    parser.add_argument("--gene-names", dest="gene_names", default=None)
    parser.add_argument(
        "--perturbation-names",
        dest="perturbation_names",
        default=None,
    )
    parser.add_argument("--output", default=None, help="Output directory (default: <results_dir>/analysis)")
    parser.add_argument("--pip-threshold", type=float, default=0.9)
    parser.add_argument("--lfsr-threshold", type=float, default=0.05)
    parser.add_argument("--rho-prime", type=float, default=0.1)
    parser.add_argument("--compute-lfsr", action="store_true", default=False)
    parser.add_argument("--lfsr-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", default=False)


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
    if args.standardize:
        preprocessing_steps.append("standardize")
    run_config = {
        "input": args.input,
        "format": screen.source.get("format", args.format),
        "output": str(output),
        "z_dim": args.z_dim,
        "l_dim": args.l_dim,
        "tau": args.tau,
        "layer": args.layer,
        "guide_key": args.guide_key,
        "guide_obsm": args.guide_obsm,
        "guide_matrix": args.guide_path,
        "metadata": args.metadata_path,
        "control_label": args.control_label,
        "missing_guide": args.missing_guide,
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
        "gene_names_file": "gene_names.txt" if screen.gene_names is not None else None,
        "perturbation_names_file": "perturbation_names.txt" if screen.perturbation_names is not None else None,
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
        layer=args.layer,
        guide_key=args.guide_key,
        guide_obsm=args.guide_obsm,
        guide_path=args.guide_path,
        metadata_path=args.metadata_path,
        control_label=args.control_label,
        missing_guide=args.missing_guide,
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
        gene_names=screen.gene_names,
        perturbation_names=screen.perturbation_names,
    )
    _write_fit_metadata(args, screen, output, selected_device)
    log.info(f"Results and reproducibility metadata saved to {output}")


def _cmd_analyze(args, log) -> None:
    from perturbvi import analyze

    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "analysis"
    gene_names = _read_names(args.gene_names) if args.gene_names else None
    perturbation_names = _read_names(args.perturbation_names) if args.perturbation_names else None
    tables = analyze(
        str(results_dir),
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        pip_threshold=args.pip_threshold,
        lfsr_threshold=args.lfsr_threshold,
        rho_prime=args.rho_prime,
        compute_lfsr=args.compute_lfsr,
        lfsr_iters=args.lfsr_iters,
        seed=args.seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table is None:
            continue
        destination = output / f"{name}.csv"
        table.to_csv(destination)
        log.info(f"Saved {destination}")
    log.info(f"Analysis complete: {output}")


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="perturbvi",
        description="PerturbVI: variational inference for single-cell Perturb-seq data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_fit_args(subparsers.add_parser("fit", help="Fit PerturbVI from a screen file"))
    _add_analyze_args(subparsers.add_parser("analyze", help="Analyze saved PerturbVI results"))
    parsed = parser.parse_args(args)
    if parsed.command == "fit" and parsed.categorical_covariates and not parsed.covariates:
        parser.error("--categorical-covariates requires --covariates")

    log = get_logger(__name__)
    log.setLevel(logging.DEBUG if parsed.verbose else logging.INFO)
    selected_device = _setup_jax(getattr(parsed, "device", "cpu"))
    if parsed.command == "fit":
        _cmd_fit(parsed, log, selected_device)
    else:
        _cmd_analyze(parsed, log)


def run_cli():
    main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
