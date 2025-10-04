from pathlib import Path
import argparse as ap
import logging
import sys
import os

import scanpy as sc
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from perturbvi.io import save_results
from perturbvi.log import get_logger
from perturbvi import infer

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

def main(args):
    argp = ap.ArgumentParser(description="Run perturbVI inference")
    argp.add_argument("matrix", type=str, help="perturbation matrix (csv) file")
    argp.add_argument("guide", type=str, help="guide csv file")
    argp.add_argument("z_dim", type=int, help="Number of latent factors")
    argp.add_argument("l_dim", type=int, help="Number of single effects")
    argp.add_argument("tau", type=int, help="residual precision")
    argp.add_argument("-o", "--output", type=str, help="Output directory")
    argp.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu", help="JAX device to use"
    )
    argp.add_argument(
        "--verbose", action="store_true", default=False, help="verbose logging",
    )

    args = argp.parse_args(args)

    out = args.output.rstrip('/')
    os.makedirs(out, exist_ok=True)
    log = get_logger(__name__, out)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    matrix_path = Path(args.matrix)
    guide_path = Path(args.guide)
    ext = matrix_path.suffix.lower()

    if matrix_path.exists() and guide_path.exists():
        log.info("files OK!")
    else:
        log.error("files not found!")
        sys.exit(1)

    if ext == ".h5ad":
        dt = sc.read_h5ad(matrix_path)
        data = jnp.asarray(dt.X)
    elif ext == ".csv":
        data = jnp.asarray(pd.read_csv(matrix_path, index_col=0)).astype(jnp.float64)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df_G = pd.read_csv(guide_path, index_col=0)
    df_G = df_G.drop(
        ["cell_barcode", "non-targeting", "Nontargeting"], axis=1, errors="ignore"
    )
    G = jnp.asarray(df_G).astype(jnp.float64)
    g_sp = sparse.bcoo_fromdense(G)
    del G, df_G

    log.info("starting inference...")

    results = infer(
        data,
        z_dim=args.z_dim,
        l_dim=args.l_dim,
        G=g_sp,
        A=None,
        p_prior=0.1,
        standardize=True,
        init="random",
        tau=args.tau,
        tol=1e-2,
        max_iter=500,
    )

    log.info("finished inference!")

    log.info(
        f"PVE across {args.z_dim} factors are {results.pve}; total PVE is {np.sum(results.pve)}"
    )

    save_results(results, path=out)
    log.info("saved results!")


def run_cli():
    return main(sys.argv[1:])

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))