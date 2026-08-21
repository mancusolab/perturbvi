import logging
import pickle

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from adjustText import adjust_text

from .infer import InferResults
from .log import get_logger


log = get_logger("perturbvi")
log.setLevel(logging.INFO)

__all__ = ["save_results", "load_results"]


def _validated_names(names, expected: int, label: str):
    if names is None:
        return None
    names = list(names)
    if len(names) != expected:
        raise ValueError(f"{label} has {len(names)} entries; expected {expected}")
    if np.asarray(pd.isna(names), dtype=bool).any():
        raise ValueError(f"{label} contains missing values")
    values = [str(name) for name in names]
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate names")
    if any("\n" in value or "\r" in value for value in values):
        raise ValueError(f"{label} may not contain newline characters")
    return values


def _write_names(path: Path, values, label: str) -> None:
    (path / f"{label}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")


def save_results(
    results: InferResults,
    path: str,
    *,
    gene_names=None,
    perturbation_names=None,
) -> None:
    """Persist a fit using stable filenames and optional name metadata.

    The directory is created when needed. ``W.txt``, ``pip.txt``, ``pve.txt``,
    and ``params_file.pkl`` retain their historical names. Optional labels are
    written to ``gene_names.txt`` and ``perturbation_names.txt`` for automatic
    saved-results analysis.
    """
    gene_names = _validated_names(gene_names, results.W.shape[1], "gene_names")
    perturbation_names = _validated_names(
        perturbation_names,
        results.params.mean_beta.shape[0],
        "perturbation_names",
    )
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    log.info("Saving PerturbVI results")

    np.savetxt(output / "W.txt", np.asarray(results.W))
    np.savetxt(output / "pip.txt", np.asarray(results.pip))
    np.savetxt(output / "pve.txt", np.asarray(results.pve))
    with (output / "params_file.pkl").open("wb") as params_file:
        pickle.dump(results.params, params_file)

    if gene_names is not None:
        _write_names(output, gene_names, "gene_names")
    if perturbation_names is not None:
        _write_names(output, perturbation_names, "perturbation_names")

    log.info(f"Results saved successfully at {output}")


def load_results(path: str) -> InferResults:
    """Load saved PerturbVI results from a results directory.

    Args:
        path: directory written by save_results (must contain params_file.pkl, pip.txt, pve.txt)

    Returns:
        InferResults with params, pve, pip populated. elbo is None (not persisted).
    """
    result_path = Path(path)
    with (result_path / "params_file.pkl").open("rb") as f:
        params = pickle.load(f)

    expected_w_shape = tuple(int(size) for size in params.W.shape)
    z_dim, gene_count = expected_w_shape

    W = np.asarray(np.loadtxt(result_path / "W.txt")).reshape(expected_w_shape)
    pip = np.asarray(np.loadtxt(result_path / "pip.txt")).reshape((z_dim, gene_count))
    pve = np.asarray(np.loadtxt(result_path / "pve.txt")).reshape((z_dim,))
    if not np.allclose(W, np.asarray(params.W), rtol=1e-5, atol=1e-7):
        raise ValueError("W.txt does not match the W stored in params_file.pkl")
    for name, values in {"W.txt": W, "pip.txt": pip, "pve.txt": pve}.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")

    return InferResults(params=params, elbo=None, pve=pve, pip=pip)


def save_pip(results: InferResults, gene_symbol: Optional[list] = None, filepath: Optional[str] = None) -> pd.DataFrame:
    """Create a DataFrame of posterior inclusion probabilities (PIPs) and optionally save to CSV.

    Args:
        results: InferResults object containing inference results
        gene_symbol: Optional list of gene symbols to use as row indices. If None, uses numeric indices
        filepath: Optional path to save CSV file. If provided, saves DataFrame to this location

    Returns:
        pd.DataFrame: DataFrame containing PIP values with labeled columns and indices
    """
    z_dim, p_dim = results.params.W.shape
    # Create column names in format 'w0', 'w1', etc.
    column_names = [f"w{i}" for i in range(z_dim)]

    # Create DataFrame with appropriate indices
    if gene_symbol is not None:
        if len(gene_symbol) != p_dim:
            raise ValueError(f"Length of gene_symbol ({len(gene_symbol)}) must match number of genes ({p_dim})")
        pip_df = pd.DataFrame(results.pip.T, columns=column_names, index=gene_symbol)
    else:
        pip_df = pd.DataFrame(results.pip.T, columns=column_names)

    # Save to CSV if filepath is provided
    if filepath is not None:
        pip_df.to_csv(filepath)

    return pip_df


# plot beta
def plot_beta(
    results: InferResults,
    factor_idx: int,
    perturb_names: Optional[list] = None,
    n: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    fontsize: int = 8,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """Plot beta values for a specific factor from inference results.
    Highlights the top n points with the highest absolute values.
    Labels are added near the top n data points, with automatic adjustment to avoid overlap.

    Args:
        results: InferResults object containing inference results
        factor_idx: Index of the factor to plot (0-based)
        perturb_names: Optional list of perturbation names to use as labels. If None, uses numeric indices
        n: Number of top values to highlight and label (default: 20)
        figsize: Figure size as (width, height) tuple (default: (10, 6))
        fontsize: Font size for gene labels (default: 8)
        save_path: Optional path to save the plot as PNG file. If None, displays plot instead
        dpi: Resolution of the saved image in dots per inch (default: 300)
    """
    # Extract parameters and compute sparse beta matrix
    params = results.params
    z_dim = params.mean_beta.shape[1]

    # Create beta matrix and dataframe
    beta_sparse = params.mean_beta * params.p_hat.T
    column_names = [f"b{i}" for i in range(z_dim)]
    df = pd.DataFrame(beta_sparse, columns=column_names)

    if perturb_names is not None:
        if len(perturb_names) != len(df):
            raise ValueError(
                f"Length of perturb_names ({len(perturb_names)}) must match number of perturbations ({len(df)})"
            )
        df.index = perturb_names

    column = f"b{factor_idx}"
    if column not in df.columns:
        raise ValueError(f"Factor index {factor_idx} out of range. Should be between 0 and {z_dim - 1}")

    plt.figure(figsize=figsize)

    # Numeric mapping for string indices
    numeric_indices = range(len(df))

    top_values = df[column].abs().nlargest(n)
    top_indices = top_values.index
    # draw the horizontal line
    top_value = df[column].abs().nlargest(n)[-1]

    # Use numeric indices for plotting
    plt.scatter(numeric_indices, df[column], color="grey", alpha=0.7)
    plt.scatter(
        [numeric_indices[df.index.get_loc(i)] for i in top_indices], df.loc[top_indices, column], color="red", zorder=5
    )

    plt.axhline(y=top_value, color="green", linestyle="--")
    plt.axhline(y=0, color="black", linestyle="-")
    plt.axhline(y=0 - top_value, color="green", linestyle="--")

    texts = []
    for index in top_indices:
        x_pos = numeric_indices[df.index.get_loc(index)]
        y_pos = df.loc[index, column]
        # Use perturbation name if provided, otherwise use index
        label = str(index)
        texts.append(plt.text(x_pos, y_pos, label, fontsize=fontsize, color="red"))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="red"))

    plt.xticks([])
    plt.ylabel("Beta Value")
    plt.title(f"Top {n} Beta Values for Factor {factor_idx}")

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def draw_perturb_heatmap(
    results: InferResults,
    perturb_names: Optional[list] = None,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "seismic",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """Draw heatmap of perturbation effects across factors.

    Args:
        results: InferResults object containing inference results
        perturb_names: Optional list of perturbation names for y-axis labels
        figsize: Figure size as (width, height) tuple (default: (10, 6))
        cmap: Color map for heatmap (default: 'seismic')
        save_path: Optional path to save the plot as PNG file. If None, displays plot instead
        dpi: Resolution of the saved image in dots per inch (default: 300)
    """
    # Extract parameters and compute sparse beta matrix
    params = results.params
    z_dim = params.mean_beta.shape[1]

    # Create beta matrix and dataframe
    beta_sparse = params.mean_beta * params.p_hat.T
    column_names = [f"{i}" for i in range(z_dim)]
    df = pd.DataFrame(beta_sparse, columns=column_names)

    if perturb_names is not None:
        if len(perturb_names) != len(df):
            raise ValueError(
                f"Length of perturb_names ({len(perturb_names)}) must match number of perturbations ({len(df)})"
            )
        df.index = perturb_names

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Create symmetric color scale
    vmax = np.max(np.abs(df))

    # Draw heatmap
    sns.heatmap(df, cmap=cmap, center=0, vmin=-vmax, vmax=vmax, ax=ax)

    # Move x-axis labels and ticks to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Set labels
    ax.set_xlabel("Factors")
    ax.set_ylabel("Perturbations")

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
