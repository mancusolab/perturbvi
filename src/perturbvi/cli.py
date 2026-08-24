from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

import pandas as pd

from ._defaults import (
    DEFAULT_INIT,
    DEFAULT_MAX_ITER,
    DEFAULT_P_PRIOR,
    DEFAULT_SEED,
    DEFAULT_STANDARDIZE,
    DEFAULT_TAU,
    DEFAULT_TOL,
    DEFAULT_VERBOSE,
)
from .log import DATE_FORMAT, get_logger, LOG_FORMAT, set_verbose


_FIT_EPILOG = """Binary perturbation matrix stored in obsm["G"]:
  perturbvi fit screen.h5ad --output results --z-dim 12 --l-dim 400 --tau 50

Reference column kept inside G:
  perturbvi fit screen.h5ad --control control --output results --z-dim 12 --l-dim 400 --tau 50
"""


def _setup_jax(device: str) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices(device)
    if not devices:
        raise RuntimeError(f"Requested JAX device '{device}' is not available")
    jax.config.update("jax_default_device", devices[0])


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    required = parser.add_argument_group("required fit arguments")
    required.add_argument("input", help="Prepared .h5ad file or AnnData Zarr store")
    required.add_argument("--output", required=True, help="Results output directory")
    required.add_argument("--z-dim", type=int, required=True, help="Number of latent gene programs")
    required.add_argument("--l-dim", type=int, required=True, help="Number of single effects per program")

    parser.add_argument(
        "--control",
        help="Column of G (obsm[g_key]) to drop as the reference (default: none)",
    )
    parser.add_argument("--covariates", nargs="+", help="AnnData obs columns to residualize")
    parser.add_argument("--x-key", help="AnnData layer containing expression (default: X)")
    parser.add_argument("--g-key", default="G", help="obsm key holding the binary perturbation matrix (default: G)")

    preprocessing = parser.add_argument_group("model preprocessing")
    preprocessing.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STANDARDIZE,
        help="Scale centered expression to unit variance (default: off)",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--p-prior",
        type=float,
        default=DEFAULT_P_PRIOR,
        help=f"Prior inclusion probability (default: {DEFAULT_P_PRIOR})",
    )
    model.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help=f"Positive initial residual precision (default: {DEFAULT_TAU})",
    )
    model.add_argument("--init", default=DEFAULT_INIT, choices=["random", "pca"], help="Factor initialization")
    model.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Maximum inference iterations (default: {DEFAULT_MAX_ITER})",
    )
    model.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"ELBO convergence tolerance (default: {DEFAULT_TOL})",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    runtime.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="JAX device (default: cpu)")
    runtime.add_argument(
        "--verbose",
        action="store_true",
        default=DEFAULT_VERBOSE,
        help="Show inference progress (default: off)",
    )


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("results_dir", help="Directory produced by perturbvi fit or save_results()")
    parser.add_argument("--compute-lfsr", action="store_true", help="Compute and save LFSR")
    parser.add_argument("--lfsr-iters", type=int, default=2000, help="Monte Carlo iterations (default: 2000)")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="LFSR random seed, used only with --compute-lfsr (default: 0)",
    )


def _write_run_metadata(output: Path, data, args) -> None:
    """Write reproducible run metadata next to the saved fit."""
    output.mkdir(parents=True, exist_ok=True)

    categoricals = None
    if args.covariates and data.covariates is not None:
        categoricals = [
            name
            for name in args.covariates
            if not (
                pd.api.types.is_numeric_dtype(data.covariates[name].dtype)
                and not pd.api.types.is_bool_dtype(data.covariates[name].dtype)
            )
        ]

    run_config = {
        "input": str(args.input),
        "z_dim": args.z_dim,
        "l_dim": args.l_dim,
        "tau": args.tau,
        "p_prior": args.p_prior,
        "standardize": args.standardize,
        "init": args.init,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "verbose": args.verbose,
        "x_key": args.x_key,
        "g_key": args.g_key,
        "control": args.control,
        "covariates": args.covariates,
        "categoricals": categoricals,
        "device": args.device,
    }
    input_summary = {
        "X_shape": list(data.X.shape),
        "G_shape": list(data.G.shape),
        "n_cells": int(data.X.shape[0]),
        "n_genes": int(data.X.shape[1]),
        "n_perturbations": int(data.G.shape[1]),
        "gene_names": list(data.gene_names),
        "perturbation_names": list(data.perturbation_names),
        "residualized": bool(args.covariates),
    }
    with (output / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2)
    with (output / "input_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(input_summary, stream, indent=2)


def _cmd_fit(args, log) -> None:
    from perturbvi import fit_screen, load_screen, save_results

    log.info(f"Loading screen from: {args.input}")
    data = load_screen(
        args.input,
        x_key=args.x_key,
        covariates=args.covariates,
        g_key=args.g_key,
        control=args.control,
    )
    log.info(f"Loaded: X={data.X.shape}, G={data.G.shape}")
    _write_run_metadata(Path(args.output), data, args)
    fit = fit_screen(
        data,
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
    save_results(fit, args.output)
    log.info(f"Results saved to {args.output}")


def _cmd_analyze(args, log) -> None:
    from perturbvi.utils import analyze

    results_dir = Path(args.results_dir)
    tables = analyze(
        results_dir,
        compute_lfsr=args.compute_lfsr,
        lfsr_iters=args.lfsr_iters,
        seed=args.seed,
    )
    for name, table in tables.items():
        destination = results_dir / f"{name}.csv"
        table.to_csv(destination)
        log.info(f"Saved {destination}")


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="perturbvi",
        description="PerturbVI: variational inference for single-cell perturbation screens",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit PerturbVI from an AnnData object, H5AD file, or Zarr store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_FIT_EPILOG,
        allow_abbrev=False,
    )
    _add_fit_args(fit_parser)
    analyze_parser = subparsers.add_parser("analyze", help="Write labeled tables from a saved fit")
    _add_analyze_args(analyze_parser)
    parsed = parser.parse_args(args)

    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)
    log = get_logger("perturbvi")
    # Same verbosity contract as fit_screen()/infer(): --verbose shows
    # timestamped progress, otherwise warnings and errors only.
    set_verbose(bool(getattr(parsed, "verbose", False)))
    if parsed.command == "fit":
        _setup_jax(parsed.device)
        _cmd_fit(parsed, log)
    else:
        _cmd_analyze(parsed, log)


def run_cli():
    main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
