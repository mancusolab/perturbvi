# #!/usr/bin/env python3
import argparse as ap

import pandas as pd
import numpy as np

import logging
import pickle
import csv
import sys
import os

from jax import config
import jax.numpy as jnp
from jax.experimental import sparse

from perturbvi.log import get_logger
from perturbvi import infer


def _parse_args(args):
    argp = ap.ArgumentParser(description="PerturbVI: infer regulatory modules from CRISPR perturbation data")
    subp = argp.add_subparsers(
        dest="command",
        help="Subcommands: infer to perform regulatory module inference using SuShiE PCA",
        required=True,
    )

    arg_infer = subp.add_parser("infer", help="Perform inference using SuSiE PCA")
    arg_infer.add_argument("X", type=str, help="Experiment CSV file")
    arg_infer.add_argument("G", type=str, help="Guide CSV file")
    # arg_infer.add_argument("S", type=str, help="Gene symbol CSV file")

    arg_infer.add_argument(
        "--platform",
        "-p",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="Platform: cpu, gpu or tpu",
    )
    arg_infer.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    arg_infer.add_argument(
        "--out",
        "-o",
        required=True,
        type=str,
        help="Output file prefix",
    )

    return argp.parse_args(args)


def _main(args):
    args = _parse_args(args)

    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", args.platform)

    out = args.out.rstrip('/')

    if not os.path.exists(out):
        os.makedirs(out)

    log = get_logger(__name__, out)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.command == "infer":

        log.info(f"Starting PerturbVI infer")

        non_targeting_column = "Nontargeting"
        params_file = f"{out}/params_file.pkl"

        log.info(f"Loading experiment data from {args.X}")
        exp_data = pd.read_csv(args.X, index_col=0)
        x = jnp.asarray(exp_data)
        log.info(f"Loaded experiment data shape: {x.shape}")

        log.info(f"Loading guide data from {args.G}")
        guide_data = pd.read_csv(args.G, index_col=0)
        log.info(f"Loaded guide data shape: {guide_data.shape}")

        reduced_guide_data = guide_data.drop(columns=[non_targeting_column])
        g_reduce_sp = sparse.bcoo_fromdense(jnp.array(reduced_guide_data.values))

        # g_sp = sparse.bcoo_fromdense(jnp.array(guide_data.values))
        # perturb_gene_list = reduced_guide_data.columns.tolist()

        # log.info(f"Loading gene symbols from {args.S}")
        # with open(args.S, mode='r', encoding='utf-8') as file:
        #     reader = csv.reader(file)
        #     gene_symbol = [row[0] for row in reader]
        # log.info(f"Loaded {len(gene_symbol)} gene symbols")

        log.info("Running inference...")
        results = infer(x, z_dim=12, l_dim=400, G=g_reduce_sp, A=None, p_prior=0.5, standardize=False, init="random",
                        tau=10, max_iter=500, tol=1e-2)

        np.savetxt(f"{out}/W.txt", results.W)
        np.savetxt(f"{out}/pip.txt", results.pip)
        np.savetxt(f"{out}/pve.txt", results.pve)

        with open(params_file, "wb") as fh:
            pickle.dump(results.params, fh)

        log.info(f"Saved all files to {out}")
        log.info("PerturbVI inference completed.")

    return 0


def run_cli():
    return _main(sys.argv[1:])

if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
