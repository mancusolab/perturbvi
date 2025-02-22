#!/usr/bin/env python3
import argparse as ap
import logging
import sys

import jax
from perturbvi.log import get_logger

def main(args):
    argp = ap.ArgumentParser(description="infer regulatory modules from CRISPR perturbation data")
    # argp.add_argument("--test", "-t", type=str, help="Version test")
    argp.add_argument(
        "--platform",
        "-p",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="platform: cpu, gpu or tpu",
    )
    argp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument("--out", "-o", type=str, help="out file prefix")

    args = argp.parse_args(args)

    jax.config.update("jax_enable_x64", True)  # complaints if using TPU
    jax.config.update("jax_platform_name", args.platform)

    log = get_logger(__name__, args.out)

    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    return 0


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))