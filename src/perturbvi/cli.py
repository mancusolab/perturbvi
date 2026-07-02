import argparse
import json
import logging
import sys

from pathlib import Path

import numpy as np

from .log import get_logger


def _setup_jax(device: str) -> None:
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if device == "cpu":
        jax.config.update("jax_default_device", jax.devices("cpu")[0])


def _read_names(path: str) -> list:
    """Read one name per line from a .csv or .txt file (no header assumed for csv)."""
    import pandas as pd
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p, header=None)[0].tolist()
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _add_fit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="Path to .h5ad file, 10x H5 file, or 10x MEX directory")
    p.add_argument(
        "--format", dest="format", default="auto",
        choices=["auto", "h5ad", "10x-h5", "10x-mex"],
        help="Input format (default: auto-detect from extension)",
    )
    p.add_argument("--output", required=True, help="Results output directory")
    p.add_argument("--z-dim", type=int, required=True, help="Number of latent factors")
    p.add_argument("--l-dim", type=int, required=True, help="Number of single effects per factor")
    p.add_argument("--tau", type=float, required=True, help="Residual precision initial value")
    p.add_argument("--layer", default="X", help="Expression layer in .h5ad (default: X uses adata.X)")
    p.add_argument("--guide-key", default=None, help="adata.obs column to build one-hot G from (h5ad)")
    p.add_argument("--guide-obsm", default=None, help="adata.obsm key for existing guide matrix (h5ad)")
    p.add_argument("--control-label", default=None, help="Perturbation label to drop from G columns")
    p.add_argument(
        "--guide-threshold", type=int, default=1,
        help="UMI count threshold to binarize guide counts (10x only, default: 1)",
    )
    p.add_argument(
        "--expression-feature-type", default="Gene Expression",
        help="Feature type string for expression matrix (10x only)",
    )
    p.add_argument(
        "--guide-feature-type", default="CRISPR Guide Capture",
        help="Feature type string for guide capture (10x only)",
    )
    p.add_argument(
        "--covariates", nargs="+", default=None,
        help="Covariate column names (h5ad: from adata.obs; 10x: from --covariate-file)",
    )
    p.add_argument(
        "--covariate-file", default=None,
        help="Barcode-indexed CSV/TSV of covariates (10x only; use with --covariates)",
    )
    p.add_argument(
        "--categoricals", nargs="+", default=None,
        help="Subset of --covariates to treat as categorical (one-hot encoded)",
    )
    p.add_argument(
        "--multi-guide", default="warn", choices=["allow", "warn", "error"],
        help="Policy for cells with >1 guide assignment (10x only, default: warn)",
    )
    p.add_argument("--p-prior", type=float, default=0.1, help="Prior perturbation inclusion probability (default: 0.1)")
    p.add_argument(
        "--standardize", action=argparse.BooleanOptionalAction, default=True,
        help="Standardize expression to unit variance before fitting (default: on)",
    )
    p.add_argument("--init", default="pca", choices=["random", "pca"], help="Factor init (default: random)")
    p.add_argument("--max-iter", type=int, default=500, help="Maximum ELBO iterations (default: 500)")
    p.add_argument("--tol", type=float, default=1e-2, help="ELBO convergence tolerance (default: 1e-2)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="JAX device (default: cpu)")
    p.add_argument("--verbose", action="store_true", default=False)


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("results_dir", help="Directory produced by perturbvi fit or save_results()")
    p.add_argument("--gene-names", default=None, help=".csv or .txt file with gene names, one per line")
    p.add_argument(
        "--perturbation-names", default=None,
        help=".csv or .txt file with perturbation names, one per line",
    )
    p.add_argument(
        "--output", default=None,
        help="Analysis output directory (default: <results_dir>/analysis)",
    )
    p.add_argument(
        "--compute-lfsr", action="store_true", default=False,
        help="Compute LFSR (expensive Monte Carlo step; off by default)",
    )
    p.add_argument("--lfsr-iters", type=int, default=2000, help="Monte Carlo iterations for LFSR (default: 2000)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--verbose", action="store_true", default=False)


def _cmd_fit(args, log) -> None:
    from perturbvi import fit_screen, load_screen, save_results

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading screen from: {args.input}")
    screen = load_screen(
        args.input,
        format=args.format,
        layer=args.layer,
        guide_key=args.guide_key,
        guide_obsm=args.guide_obsm,
        control_label=args.control_label,
        guide_threshold=args.guide_threshold,
        expression_feature_type=args.expression_feature_type,
        guide_feature_type=args.guide_feature_type,
        covariates=args.covariates,
        covariate_file=args.covariate_file,
        multi_guide=args.multi_guide,
    )
    log.info(f"Loaded: X={np.array(screen.X).shape}, G={np.array(screen.G).shape}")

    if args.covariates:
        from perturbvi import residualize_screen
        log.info(f"Residualizing covariates: {args.covariates}")
        screen = residualize_screen(screen, categoricals=args.categoricals)

    log.info("Starting inference...")
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
    )
    log.info(f"Done. PVE per factor: {np.array(results.pve).round(4).tolist()}")

    save_results(results, path=str(out))
    log.info(f"Results saved to {out}/")

    run_config = {
        "input": args.input,
        "format": args.format,
        "output": str(out),
        "z_dim": args.z_dim,
        "l_dim": args.l_dim,
        "tau": args.tau,
        "layer": args.layer,
        "guide_key": args.guide_key,
        "guide_obsm": args.guide_obsm,
        "control_label": args.control_label,
        "guide_threshold": args.guide_threshold,
        "covariates": args.covariates,
        "covariate_file": args.covariate_file,
        "categoricals": args.categoricals,
        "multi_guide": args.multi_guide,
        "p_prior": args.p_prior,
        "standardize": args.standardize,
        "init": args.init,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "device": args.device,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    X_arr = np.array(screen.X)
    G_arr = np.array(screen.G)
    input_summary = {
        "source": screen.source,
        "X_shape": list(X_arr.shape),
        "G_shape": list(G_arr.shape),
        "n_gene_names": len(screen.gene_names) if screen.gene_names else None,
        "n_perturbation_names": len(screen.perturbation_names) if screen.perturbation_names else None,
        "n_cell_names": len(screen.cell_names) if screen.cell_names else None,
    }
    with open(out / "input_summary.json", "w") as f:
        json.dump(input_summary, f, indent=2)

    log.info(f"Metadata written to {out}/run_config.json and {out}/input_summary.json")


def _cmd_analyze(args, log) -> None:
    from perturbvi import analyze, load_results

    results_dir = Path(args.results_dir)
    out = Path(args.output) if args.output else results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    gene_names = _read_names(args.gene_names) if args.gene_names else None
    perturbation_names = _read_names(args.perturbation_names) if args.perturbation_names else None

    log.info(f"Loading results from: {results_dir}")
    fitted = load_results(str(results_dir))

    log.info("Running analysis...")
    tables = analyze(
        fitted,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        compute_lfsr=args.compute_lfsr,
        lfsr_iters=args.lfsr_iters,
        seed=args.seed,
    )

    for name, df in tables.items():
        csv_path = out / f"{name}.csv"
        df.to_csv(csv_path)
        log.info(f"Saved {csv_path}")

    log.info(f"Analysis complete. Output: {out}/")


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="perturbvi",
        description="perturbVI: variational inference for single-cell Perturb-seq data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_fit_args(subparsers.add_parser("fit", help="Fit perturbVI from a screen file"))
    _add_analyze_args(subparsers.add_parser("analyze", help="Analyze saved perturbVI results"))

    args = parser.parse_args(args)

    log = get_logger(__name__)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    _setup_jax(getattr(args, "device", "cpu"))

    if args.command == "fit":
        _cmd_fit(args, log)
    elif args.command == "analyze":
        _cmd_analyze(args, log)


def run_cli():
    main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
