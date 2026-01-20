from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np
import pickle
import os

import equinox as eqx
import jax.scipy.special as jspec
import lineax as lx

from jax import jit, lax, numpy as jnp, random as rdm
from jaxtyping import Array
from typing import Sequence

multi_linear_solve = eqx.filter_vmap(lx.linear_solve, in_axes=(None, 1, None))

_add_ufunc = jnp.frompyfunc(jnp.add, nin=2, nout=1, identity=0)
outer_add = _add_ufunc.outer


def logdet(A: Array) -> Array:
    _, ldet = jnp.linalg.slogdet(A)
    return ldet


def kl_discrete(alpha: Array, pi: Array) -> Array:
    """A function that calculates the Kullback-Leibler divergence for multinomial distributions

    **Arguments:**

    -`alpha` [`Array`]: An array representing the first discrete distribution.

    -`pi` [`Array`]: An array representing the second discrete distribution.

    **Returns:**

    The Kullback-Leibler divergence between the two distributions.
    """
    return jnp.sum(jspec.xlogy(alpha, alpha) - jspec.xlogy(alpha, pi))


@partial(jit, static_argnums=(2, 3, 4))
def prob_pca(rng_key, X, k, max_iter=1000, tol=1e-3):
    """Probabilistic PCA algorithm to initialize latent factors.

    **Arguments:**

    -`rng_key` [`PRNGKey`]: Random key generator.

    -`X` [`Array`]: The observed data.

    -`k` [`int`]: The latent dimension.

    -`max_iter` [`int`]: The maximum number of iterations, default is 1000.

    -`tol` [`float`]: The convergence tolerance, default is 1e-3.

    **Returns:**

    - `Z` [`Array`]: The estimated latent factors.

    -`W` [`Array`]: The estimated loadings.
    """

    n_dim, p_dim = X.shape

    # initial guess for W
    w_key, z_key = rdm.split(rng_key, 2)

    # good enough for initialization
    solver = lx.Cholesky()

    # check if reach the max_iter, or met the norm criterion every 100 iteration
    def _condition(carry):
        i, _, Z, old_Z = carry
        iter_check = i < max_iter
        tol_check = jnp.linalg.norm(Z - old_Z) > tol
        # scaled_tol_check = tol_check / n_dim > tol
        return iter_check & tol_check

    # EM algorithm for PPCA
    def _step(carry):
        i, W, Z, _ = carry

        # E step
        W_op = lx.MatrixLinearOperator(W @ W.T, tags=lx.positive_semidefinite_tag)
        Z_new = multi_linear_solve(W_op, W @ X.T, solver).value

        # M step
        Z_op = lx.MatrixLinearOperator(Z_new.T @ Z_new, tags=lx.positive_semidefinite_tag)
        W = multi_linear_solve(Z_op, Z_new.T @ X, solver).value.T

        return i + 1, W, Z_new, Z

    W = rdm.normal(w_key, shape=(k, p_dim))
    Z = rdm.normal(z_key, shape=(n_dim, k))
    Z_zero = jnp.zeros_like(Z)
    initial_carry = 0, W, Z, Z_zero

    _, W, Z, _ = lax.while_loop(_condition, _step, initial_carry)
    Z, _ = jnp.linalg.qr(Z)

    return Z, W


# Create function to evaluate Local False Sign Rate
# First Create a function to sample single effect matrix based on params.alpha
def bern_sample(alpha):
    """Sample from a Bernoulli distribution with probability alpha.

    **Arguments:**

    - `alpha` [`Array`]: The probability of each row in the L x K matrix.

    **Returns:**

    - `efficient_result_matrix` [`Array`]: The sampled matrix.

    """
    l_dim, z_dim, _ = alpha.shape
    # Generate random numbers for each row in the L x K matrix
    # These random numbers are used as indices for selecting features
    random_indices = np.random.rand(l_dim, z_dim)
    # Calculate the cumulative sum of probabilities along the P dimension
    cumulative_probabilities = np.cumsum(alpha, axis=2)

    # Determine the indices in P where the cumulative probability exceeds the random index
    # This effectively samples from the probability distribution
    feature_indices = np.argmax(cumulative_probabilities > random_indices[..., np.newaxis], axis=2)
    # Initialize the result matrix with zeros
    efficient_result_matrix = np.zeros_like(alpha)

    # Use advanced indexing to set the selected features to 1
    efficient_result_matrix[np.arange(l_dim)[:, np.newaxis], np.arange(z_dim), feature_indices] = 1

    return efficient_result_matrix


def bern_sample_jax(key, alpha):
    """JAX version of bern_sample function.

    Arguments:
        key: JAX random key
        alpha: probability matrix of shape (l_dim, z_dim, p_dim)
    """
    random_values = rdm.uniform(key, shape=alpha.shape[:-1])
    cumsum = jnp.cumsum(alpha, axis=-1)
    return jnp.eye(alpha.shape[-1])[jnp.argmax(cumsum > random_values[..., None], axis=-1)]


@partial(jit, static_argnums=(2,))
def _compute_lfsr_step(key, params, iters):
    """Jitted inner loop of LFSR computation"""
    l_dim, z_dim, p_dim = params.alpha.shape
    g_dim, _ = params.mean_beta.shape
    reshaped_var_w = jnp.repeat(params.var_w[:, :, jnp.newaxis], p_dim, axis=2)

    def _inner_loop(carry, i):  # Modified to accept iteration index
        key, total_pos, total_neg = carry

        # Split keys for different random operations
        key, w_key, a_key, e_key, b_key = rdm.split(key, 5)

        # Sample W
        sample_w = params.mean_w + jnp.sqrt(reshaped_var_w) * rdm.normal(w_key, shape=params.mean_w.shape)
        sample_alpha = bern_sample_jax(a_key, params.alpha)
        sample_W = jnp.sum(sample_w * sample_alpha, axis=0)

        # Sample B
        sample_eta = rdm.bernoulli(e_key, params.p_hat.T)
        sample_beta = params.mean_beta + jnp.sqrt(params.var_beta) * rdm.normal(b_key, shape=params.mean_beta.shape)
        sample_B = sample_beta * sample_eta

        # Compute outer product
        sample_oe = sample_B @ sample_W
        ind_pos = sample_oe >= 0
        ind_neg = sample_oe <= 0

        return (key, total_pos + ind_pos, total_neg + ind_neg), None

    # Initialize
    total_pos_zero = jnp.zeros((g_dim, p_dim))
    total_neg_zero = jnp.zeros((g_dim, p_dim))
    init_carry = (key, total_pos_zero, total_neg_zero)

    # Run the loop
    (_, total_pos_zero, total_neg_zero), _ = lax.scan(_inner_loop, init_carry, jnp.arange(iters))

    return total_pos_zero, total_neg_zero


def compute_lfsr(key, params, iters=2000):
    """Compute the LFSR (Local False Sign Rate) using the given parameters.

    Arguments:
        key: JAX random key
        params: The parameters of the model
        iters: Number of iterations (default=2000)
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start computing LFSR at {current_time}")

    # Split computation into chunks to show progress
    chunk_size = 100
    num_chunks = iters // chunk_size
    remaining = iters % chunk_size

    total_pos = 0
    total_neg = 0

    for i in range(num_chunks):
        pos_chunk = 0
        neg_chunk = 0
        # Process each iteration within the chunk individually
        for j in range(chunk_size):
            iter_key = rdm.fold_in(key, i * chunk_size + j)  # Unique key for each iteration
            pos, neg = _compute_lfsr_step(iter_key, params, 1)  # Process single iteration
            pos_chunk += pos
            neg_chunk += neg
        total_pos += pos_chunk
        total_neg += neg_chunk
        print(f"Completed {(i + 1) * chunk_size}/{iters} iterations")

    # Handle remaining iterations if any
    if remaining > 0:
        pos_rem = 0
        neg_rem = 0
        for j in range(remaining):
            iter_key = rdm.fold_in(key, num_chunks * chunk_size + j)
            pos, neg = _compute_lfsr_step(iter_key, params, 1)
            pos_rem += pos
            neg_rem += neg
        total_pos += pos_rem
        total_neg += neg_rem
        print(f"Completed {iters}/{iters} iterations")

    # Compute final LFSR
    lfsr = jnp.minimum(total_pos, total_neg) / iters

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished computing LFSR at {current_time}")

    return lfsr


def pip_analysis(pip: jnp.ndarray, rho=0.9, rho_prime=0.05):
    """Create a function to give a quick summary of PIPs

    Args:
        pip:the pip matrix, a ndarray from results object returned by
        infer.perturbvi

    """
    z_dim, p_dim = pip.shape
    results = []

    print(f"Of {p_dim} features from the data, SuSiE PCA identifies:")
    for k in range(z_dim):
        num_signal = jnp.where(pip[k, :] >= rho)[0].shape[0]
        num_zero = jnp.where(pip[k, :] < rho_prime)[0].shape[0]
        print(f"Component {k} has {num_signal} features with pip>{rho}; and {num_zero} features with pip<{rho_prime}")
        results.append([num_signal, num_zero])

    df = pd.DataFrame(results, columns=["num_signal", "num_zero"])

    # Calculate and print mean and standard deviation for each column
    mean_signal = df["num_signal"].mean()
    std_signal = df["num_signal"].std()
    mean_zero = df["num_zero"].mean()
    std_zero = df["num_zero"].std()

    print(f"Mean and standard deviation for num_signal: {mean_signal}, {std_signal}")
    print(f"Mean and standard deviation for num_zero: {mean_zero}, {std_zero}")

    return df


def find_top_genes(df, pip_cutoff = 0.9):
    high_value_genes = {}
    for column in df.columns:
        high_values = df[df[column] > pip_cutoff].index.tolist()
        high_value_genes[column] = high_values
    return high_value_genes


def analyze_output(
    dir: str,
    perturb_genes: Sequence[str],
    background_genes: Sequence[str],
):
    """
    Produce results from output files.

    Args:
        dir: Results directory
        perturb_genes: List of perturbed gene symbols (e.g. 14 genes)
        background_genes: List of background gene symbols (e.g. ~6000 genes)
    """

    perturb_genes = list(perturb_genes)
    background_genes = list(background_genes)

    params_path = f"{dir}/params_file.pkl"
    pip_path = f"{dir}/pip.txt"
    W_path = f"{dir}/W.txt"
    pip_df_path = f"{dir}/pip.csv"
    lfsr_path = f"{dir}/lfsr.csv"
    p_hat_path = f"{dir}/p_hat.csv"
    beta_path = f"{dir}/beta_target.csv"
    overall_path = f"{dir}/overall_effect.csv"

    # For Luhmes
    # guide_path = "luhmes_G.csv"
    # gene_path = "luhmes_gene_symbol_v2.csv"
    # G = pd.read_csv(guide_path, index_col=0)
    # G_reduce = G.drop(columns=["Nontargeting"])
    # perturbed = G_reduce.columns.to_list()  # 14 genes perturbed
    # genes = pd.read_csv(gene_path, header=None)[0].to_list()  # 6000 gene symbols (background genes)

    with open(params_path, "rb") as file:
        params = pickle.load(file)

    pip = np.loadtxt(pip_path)
    W = pd.DataFrame(np.loadtxt(W_path))

    z_dim, p_dim = params.W.shape
    g_dim, z_dim = params.mean_beta.shape
    n_dim, z_dim = params.mean_z.shape
    print(f"Dims: z_dim={z_dim}, p_dim={p_dim}, g_dim={g_dim}, n_dim={n_dim}\n")
    column_names_b = [f"b{i}" for i in range(z_dim)]
    column_names_w = [f"w{i}" for i in range(z_dim)]

    beta_sparse = params.mean_beta * params.p_hat.T
    overall_effect = beta_sparse @ params.W

    # pip_df = pd.DataFrame(pip.T, columns=column_names_w)
    pip_df = pd.DataFrame(pip.T, columns=column_names_w, index=background_genes)
    pip_df.to_csv(pip_df_path)
    
    analysis = pip_analysis(jnp.asarray(pip).astype(jnp.float32),rho=0.90,rho_prime=0.10)

    for col in pip_df.columns:
        sig_genes = pip_df.index[pip_df[col] > 0.9].tolist()
        if not os.path.exists(f"{dir}/pip"):
            os.makedirs(f"{dir}/pip")
        np.savetxt(f"{dir}/pip/{col}_sig_genes.txt", sig_genes, fmt="%s")
        print(f"Saved {len(sig_genes)} sig. genes for {col} (PIP > 0.9) in {dir}/pip/{col}_sig_genes.txt")
    skip_lfsr = os.path.exists(lfsr_path)

    if skip_lfsr:
        print("\nlfsr.csv exists. skipping lfsr compute...")
        lfsr_df = pd.read_csv(lfsr_path, index_col=0)
        lfsr_df.index = pd.Index(background_genes)
        lfsr_df.columns = perturb_genes
        lfsr_df.to_csv(lfsr_path)
    else:
        print("\ncomputing lfsr...")
        lfsr = compute_lfsr(key=rdm.PRNGKey(0), params=params, iters=2000)
        lfsr.block_until_ready()
        lfsr_np = np.array(lfsr)
        # lfsr_df = pd.DataFrame(lfsr_np.T)
        lfsr_df = pd.DataFrame(lfsr_np.T, index=background_genes, columns=perturb_genes)
        lfsr_df.to_csv(lfsr_path)

    # number of degs per w from PIP (find top genes with high pip)
    # top_pip_degs = find_top_genes(pip_df,0.90)
    # number of degs per w from PIP
    num_deg_per_w = (pip_df>0.9).sum(axis=0)


    # number of degs per perturbed gene from LFSR
    num_deg_per_gene = (lfsr_df<0.05).sum(axis=0)

    for col in lfsr_df.columns:
        sig_genes = lfsr_df.index[lfsr_df[col] < 0.05].tolist()
        if not os.path.exists(f"{dir}/lfsr"):
            os.makedirs(f"{dir}/lfsr")
        np.savetxt(f"{dir}/lfsr/{col}_sig_genes.txt", sig_genes, fmt="%s")
        print(f"Saved {len(sig_genes)} sig. genes for {col} (LFSR < 0.05) in {dir}/lfsr.")

    # p_hat_df = pd.DataFrame(params.p_hat.T, columns=column_names_b)
    p_hat_df = pd.DataFrame(params.p_hat.T, columns=column_names_b, index=perturb_genes)
    p_hat_df.to_csv(p_hat_path)
        
    # beta_df = pd.DataFrame(beta_sparse, columns=column_names_b)
    beta_df = pd.DataFrame(beta_sparse, columns=column_names_b, index=perturb_genes)
    beta_df.to_csv(beta_path)

    # overall_df = pd.DataFrame(overall_effect.T)
    overall_df = pd.DataFrame(overall_effect.T, columns=perturb_genes, index=background_genes)
    overall_df.to_csv(overall_path)

    print("\nshape of W df", W.shape)
    print("shape of pip df", pip_df.shape)
    print("shape of lfsr df", lfsr_df.shape)
    print("shape of p_hat df", p_hat_df.shape)
    print("shape of beta target df", beta_df.shape)
    print("shape of overall effect df", overall_df.shape)

    print("\nDone!\n")

    print(f"To check significant DEGs per W, see {dir}/pip")
    print(f"To check significant DEGs per perturbed gene, see {dir}/lfsr\n")

    return {
        "params": params,
        "W_df": W,
        "pip_df": pip_df,
        "pip_analysis": analysis,
        "num_deg_per_perturbed_gene": num_deg_per_gene,
        "num_deg_per_w": num_deg_per_w,
        "lfsr_df": lfsr_df,
        "p_hat_df": p_hat_df,
        "beta_df": beta_df,
        "overall_effect_df": overall_df,
    }
