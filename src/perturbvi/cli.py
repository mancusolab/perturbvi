#!/usr/bin/env python3
import argparse as ap
import logging
import sys

import jax
from perturbvi.log import get_logger
from perturbvi.sim import generate_sim
from perturbvi.utils import compute_lfsr

def main(args):
    argp = ap.ArgumentParser(description="infer regulatory modules from CRISPR perturbation data")
    # argp.add_argument("--test", "-t", type=str, help="Version test")
    subparsers = argp.add_subparsers(dest="command", required=True)

    # Subparser for generate_sim
    parser_sim = subparsers.add_parser("sim", help="Generate simulated data")
    parser_sim.add_argument("--seed", type=int, required=True, help="Seed for random initialization")
    parser_sim.add_argument("--l_dim", type=int, required=True, help="Number of single effects in each factor")
    parser_sim.add_argument("--n_dim", type=int, required=True, help="Number of samples in the data")
    parser_sim.add_argument("--p_dim", type=int, required=True, help="Number of features in the data")
    parser_sim.add_argument("--z_dim", type=int, required=True, help="Number of latent dimensions")
    parser_sim.add_argument("--g_dim", type=int, required=True, help="Perturbation dimensions")
    parser_sim.add_argument("--b_sparsity", type=float, default=0.2, help="Sparsity of perturbation effects")
    parser_sim.add_argument("--effect_size", type=float, default=1.0,
                            help="Effect size of features contributing to the factor")

    # Subparser for compute_lfsr
    parser_lfsr = subparsers.add_parser("lfsr", help="Compute the LFSR")
    parser_lfsr.add_argument("--seed", type=int, required=True, help="JAX random key seed")
    parser_lfsr.add_argument("--params", type=str, required=True, help="Path to model parameters file")
    parser_lfsr.add_argument("--iters", type=int, default=2000, help="Number of iterations")

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

    args = argp.parse_args()

    if args.command == "sim":
        sim_data = generate_sim(
            seed=args.seed,
            l_dim=args.l_dim,
            n_dim=args.n_dim,
            p_dim=args.p_dim,
            z_dim=args.z_dim,
            g_dim=args.g_dim,
            b_sparsity=args.b_sparsity,
            effect_size=args.effect_size,
        )
        log.info("Simulation completed:", sim_data)

    elif args.command == "lfsr":
        key = jax.random.PRNGKey(args.seed)
        lfsr_result = compute_lfsr(key, args.params, iters=args.iters)
        log.info("LFSR Computed:", lfsr_result)

    return 0


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
