#!/usr/bin/env python3
import argparse as ap
import logging
import sys

import jax
from perturbvi.log import get_logger
from perturbvi.sim import generate_sim
from perturbvi.utils import compute_lfsr
from perturbvi.infer import infer

def main(args):
    argp = ap.ArgumentParser(description="infer regulatory modules from CRISPR perturbation data")
    subparsers = argp.add_subparsers(dest="command", required=True)

    # generate_sim
    arg_sim = subparsers.add_parser("sim", help="generate simulated data")
    arg_sim.add_argument("--seed", type=int, required=True, help="seed")
    arg_sim.add_argument("--l_dim", type=int, required=True, help="num of single effects in each factor")
    arg_sim.add_argument("--n_dim", type=int, required=True, help="num of samples")
    arg_sim.add_argument("--p_dim", type=int, required=True, help="num of features")
    arg_sim.add_argument("--z_dim", type=int, required=True, help="num of latent dimensions")
    arg_sim.add_argument("--g_dim", type=int, required=True, help="perturbation dimensions")
    arg_sim.add_argument("--b_sparsity", type=float, default=0.2, help="sparsity of perturbation effects")
    arg_sim.add_argument("--effect_size", type=float, default=1.0, help="effect size of features")

    # compute_lfsr
    arg_lfsr = subparsers.add_parser("lfsr", help="compute the LFSR")
    arg_lfsr.add_argument("--seed", type=int, required=True, help="seed")
    arg_lfsr.add_argument("--params", type=str, required=True, help="path to model params file")
    arg_lfsr.add_argument("--iters", type=int, default=2000, help="num of iterations")

    # infer
    arg_infer = subparsers.add_parser("infer", help="perform inference using SuSiE PCA")
    arg_infer.add_argument("--X", type=str, required=True, help="path to expression count matrix")
    arg_infer.add_argument("--z_dim", type=int, required=True, help="latent dimension")
    arg_infer.add_argument("--l_dim", type=int, required=True, help="num of single effects per factor")
    arg_infer.add_argument("--G", type=str, required=True, help="path to perturbation density matrix")
    arg_infer.add_argument("--A", type=str, help="path to annotation matrix")
    arg_infer.add_argument("--p_prior", type=float, default=0.5, help="prior probability")
    arg_infer.add_argument("--tau", type=float, default=1.0, help="initial residual precision")
    arg_infer.add_argument("--standardize", action="store_true", help="standardize input data")
    arg_infer.add_argument("--init", type=str, choices=["pca", "random"], default="pca", help="init method")
    arg_infer.add_argument("--learning_rate", type=float, default=1e-2, help="learning rate for inference")
    arg_infer.add_argument("--max_iter", type=int, default=400, help="maximum num of iterations")
    arg_infer.add_argument("--tol", type=float, default=1e-3, help="convergence tolerance")
    arg_infer.add_argument("--seed", type=int, default=0, help="seed")

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
        log.info("simulation completed:", sim_data)

    elif args.command == "lfsr":
        key = jax.random.PRNGKey(args.seed)
        lfsr_result = compute_lfsr(key, args.params, iters=args.iters)
        log.info("LFSR computed:", lfsr_result)

    # TODO: figure out improved IO for all CLI commands and use-cases
    elif args.command == "infer":
        X = jax.numpy.load(args.X)
        G = jax.numpy.load(args.G)
        A = jax.numpy.load(args.A) if args.A else None
        results = infer(
            X=X,
            z_dim=args.z_dim,
            l_dim=args.l_dim,
            G=G,
            A=A,
            p_prior=args.p_prior,
            tau=args.tau,
            standardize=args.standardize,
            init=args.init,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            verbose=args.verbose,
        )
        log.info("inference completed: %s", results)

    return 0


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
