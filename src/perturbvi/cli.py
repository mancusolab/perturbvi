import argparse as ap
import logging
from pathlib import Path
import sys
import os

import susiepca as sp
import scanpy as sc
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax.experimental import sparse

import perturbvi

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main(args):
    argp = ap.ArgumentParser(description="Run perturbVI inference")
    argp.add_argument("matrix", type=str, help="residual h5ad or csv file")
    argp.add_argument("guide", type=str, help="guide csv file")
    argp.add_argument("z_dim", type=int, help="Number of latent factors")
    argp.add_argument("l_dim", type=int, help="Number of single effects")
    argp.add_argument("tau", type=int, help="residual precision")
    argp.add_argument("-o", "--output", type=str, help="Output directory")
    argp.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu", help="JAX device to use"
    )

    args = argp.parse_args(args)
    os.makedirs(args.output, exist_ok=True)

    matrix_path = Path(args.matrix)
    guide_path = Path(args.guide)
    ext = matrix_path.suffix.lower()

    if matrix_path.exists() and guide_path.exists():
        logging.info("files OK!")
    else:
        logging.error("files not found!")
        sys.exit(1)

    if ext == ".h5ad":
        dt = sc.read_h5ad(matrix_path)
        data = dt.X
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

    logging.info("starting inference")

    results = perturbvi.infer(
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

    logging.info("finished inference!")

    logging.info(
        f"PVE across {args.z_dim} factors are {results.pve}; total PVE is {np.sum(results.pve)}"
    )

    sp.io.save_results(results, path=args.output)
    logging.info("saved results!")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))