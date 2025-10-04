import pickle

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from adjustText import adjust_text

from .infer import InferResults


__all__ = ["save_results"]


# save all results as a pickle object
def save_results(results: InferResults, path: str):
    """Create a function to save SuSiE PCA results returned by function
    perturbvi.infer

    Args:
        results: results object returned by perturbvi.infer
        path: local path to save the results subject


    """
    print("Save results from SuSiE PCA")

    np.savetxt(f"{path}/W.txt", results.W)
    np.savetxt(f"{path}/pip.txt", results.pip)
    np.savetxt(f"{path}/pve.txt", results.pve)

    params_file = open(f"{path}/params_file.pkl", "wb")
    pickle.dump(results.params, params_file)
    params_file.close()

    print(f"Results saved successfully at {path}")

    return


# function to find genes with high pip
def find_top_genes(
    results: InferResults, pip_cutoff: float = 0.9, gene_symbol: Optional[list] = None, filepath: Optional[str] = None
) -> dict:
    """Find genes with high posterior inclusion probabilities (PIPs) and optionally save results.

    Args:
        results: InferResults object containing inference results
        pip_cutoff: Threshold for considering a PIP value as significant (default: 0.9)
        gene_symbol: Optional list of gene symbols to use as row indices. If None, uses numeric indices
        filepath: Optional path to save results. If provided, saves results as CSV with columns:
                 'factor', 'gene', 'pip_value'

    Returns:
        dict: Dictionary mapping each factor to list of genes with PIP > cutoff
    """
    # Create PIP DataFrame
    z_dim, p_dim = results.params.W.shape
    column_names = [f"w{i}" for i in range(z_dim)]

    if gene_symbol is not None:
        if len(gene_symbol) != p_dim:
            raise ValueError(f"Length of gene_symbol ({len(gene_symbol)}) must match number of genes ({p_dim})")
        df = pd.DataFrame(results.pip.T, columns=column_names, index=gene_symbol)
    else:
        df = pd.DataFrame(results.pip.T, columns=column_names)

    # Initialize an empty dictionary
    high_value_genes = {}

    # Iterate over each column (factor) in the DataFrame
    for column in df.columns:
        # Find the rows in the current column with values greater than cutoff
        high_values_mask = df[column] > pip_cutoff
        high_values = df.index[high_values_mask].tolist()
        # Store these rows in the dictionary
        high_value_genes[column] = high_values

    # Save to file if filepath is provided
    if filepath is not None:
        # Create a list to store rows for the DataFrame
        rows = []
        for factor, genes in high_value_genes.items():
            for gene in genes:
                rows.append({"factor": factor, "gene": gene, "pip_value": df.loc[gene, factor]})
        # Convert to DataFrame and save
        results_df = pd.DataFrame(rows)
        results_df.to_csv(filepath, index=False)

    return high_value_genes


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
        raise ValueError(f"Factor index {factor_idx} out of range. Should be between 0 and {z_dim-1}")

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


def draw_perturb_heatmap(results: InferResults, perturb_names: Optional[list] = None,
                        figsize: Tuple[int, int] = (10, 6), cmap: str = 'seismic',
                        save_path: Optional[str] = None, dpi: int = 300) -> None:
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
    column_names = [f'{i}' for i in range(z_dim)]
    df = pd.DataFrame(beta_sparse, columns=column_names)
    
    if perturb_names is not None:
        if len(perturb_names) != len(df):
            raise ValueError(f"Length of perturb_names ({len(perturb_names)}) must match number of perturbations ({len(df)})")
        df.index = perturb_names

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create symmetric color scale
    vmax = np.max(np.abs(df))
    
    # Draw heatmap
    sns.heatmap(df, cmap=cmap, center=0, vmin=-vmax, vmax=vmax, ax=ax)
    
    # Move x-axis labels and ticks to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Set labels
    ax.set_xlabel("Factors")
    ax.set_ylabel("Perturbations")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

