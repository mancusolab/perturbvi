"""Heatmaps for PerturbVI results."""

from __future__ import annotations

import math
import re
import warnings

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


__all__ = ["plot_B", "plot_W", "plot_BW"]

_EFFECT_COLORS = ("#2166AC", "#67A9CF", "#FFFFFF", "#EF8A62", "#B2182B")
_GROUP_COLORS = ("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3")
_INK = "#342D38"


def _matplotlib():
    try:
        import matplotlib.pyplot as plt

        from matplotlib import colors
    except ImportError as exc:
        raise ImportError("Plotting requires Matplotlib: uv pip install matplotlib") from exc
    return plt, colors


def _table(table, key):
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError(f"{key} must be a nonempty DataFrame")
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f"{key} identifiers must be unique")
    if table.index.hasnans or table.columns.hasnans:
        raise ValueError(f"{key} identifiers must not be missing")
    if not np.issubdtype(table.to_numpy().dtype, np.number) or not np.isfinite(table.to_numpy()).all():
        raise ValueError(f"{key} must contain finite numeric values")
    return table


def _ids(requested, available, name):
    if requested is None:
        return available.tolist()
    if isinstance(requested, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of IDs, not a single string")
    selected = list(requested)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{name} must be nonempty and contain unique IDs")
    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError(f"Unknown {name}: {missing}")
    return selected


def _select(table, rows, columns):
    return table.loc[_ids(rows, table.index, "rows"), _ids(columns, table.columns, "columns")].copy()


def _mask(estimate, probability, key, matrix, cutoff, direction, transpose=False):
    if cutoff is None:
        return None
    if probability is None:
        raise ValueError(f"Pass {key}= as a DataFrame when show_significance=True")
    probability = _table(probability, key)
    if set(probability.index) != set(estimate.index) or set(probability.columns) != set(estimate.columns):
        raise ValueError(f"{key} identifiers do not match the effect matrix")
    if ((probability.to_numpy() < 0) | (probability.to_numpy() > 1)).any():
        raise ValueError(f"{key} values must be between 0 and 1")
    if transpose:
        probability = probability.T
    selected = probability.loc[matrix.index, matrix.columns]
    return selected > cutoff if direction == "greater" else selected < cutoff


def _factor_label(value):
    match = re.fullmatch(r"factor_(\d+)", str(value))
    return f"Factor {int(match.group(1)) + 1}" if match else str(value)


def _labels(ids, factors=False):
    return [_factor_label(value) if factors else str(value) for value in ids]


def _annotation(matrix, metadata, gene_annotation_columns=None):
    _, colors = _matplotlib()
    if metadata is None:
        return matrix, _labels(matrix.index), None, [], None
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("gene_annotations must be a pandas DataFrame")
    required = ("gene_ID", "gene_name", "annotation")
    columns = {name: name for name in required} if gene_annotation_columns is None else gene_annotation_columns
    if not isinstance(columns, Mapping) or set(columns) != set(required):
        raise ValueError("gene_annotation_columns must map gene_ID, gene_name, and annotation to input columns")
    source_columns = [columns[name] for name in required]
    if len(set(source_columns)) != len(required):
        raise ValueError("gene_annotation_columns must map each role to a different input column")
    missing = set(source_columns) - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing gene annotation columns: {sorted(missing)}")
    metadata = metadata.loc[:, source_columns].copy()
    metadata.columns = required
    if metadata["gene_ID"].isna().any() or not metadata["gene_ID"].is_unique:
        raise ValueError("gene_ID values must be unique and nonmissing")
    selected = metadata.set_index("gene_ID").reindex(matrix.index)
    # Group order and colors follow first appearance in the complete CSV.
    categories = metadata["annotation"].dropna().drop_duplicates().tolist()
    ranks = selected["annotation"].map({group: i for i, group in enumerate(categories)})
    order = np.argsort(ranks.fillna(len(categories)).to_numpy(), kind="stable")
    matrix, selected = matrix.iloc[order], selected.iloc[order]
    labels = [
        str(name) if pd.notna(name) else fallback
        for name, fallback in zip(selected["gene_name"], _labels(matrix.index))
    ]
    palette = {group: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, group in enumerate(categories)}
    group_order = selected["annotation"].dropna().drop_duplicates().tolist()
    row_colors = [
        colors.to_rgba(palette[group]) if pd.notna(group) else colors.to_rgba("#EEEEEE")
        for group in selected["annotation"]
    ]
    legend = [(str(group), palette[group]) for group in group_order]
    if selected["annotation"].isna().any():
        legend.append(("Unannotated", "#EEEEEE"))
    return matrix, labels, np.array(row_colors), legend, selected["annotation"].tolist()


def _leaf_order(values):
    if values.shape[0] < 2:
        return np.arange(values.shape[0])
    from scipy.cluster.hierarchy import leaves_list, linkage

    return leaves_list(linkage(values, method="ward", metric="euclidean"))


def _cluster_matrix(
    matrix, *, row_clustering="none", column_clustering="none", row_labels=None, row_colors=None, row_groups=None
):
    row_options = ("none", "global", "within_annotation")
    column_options = ("none", "global")
    if row_clustering not in row_options:
        raise ValueError(f"row_clustering must be one of {row_options}")
    if column_clustering not in column_options:
        raise ValueError(f"column_clustering must be one of {column_options}")
    if row_clustering == "within_annotation" and row_groups is None:
        raise ValueError("row_clustering='within_annotation' requires gene_annotations")

    if row_clustering == "global":
        order = _leaf_order(matrix.to_numpy())
    elif row_clustering == "within_annotation":
        order = []
        start = 0
        while start < len(row_groups):
            group = row_groups[start]
            end = start + 1
            while end < len(row_groups):
                next_group = row_groups[end]
                same_group = (pd.isna(group) and pd.isna(next_group)) or (
                    pd.notna(group) and pd.notna(next_group) and group == next_group
                )
                if not same_group:
                    break
                end += 1
            order.extend(start + _leaf_order(matrix.iloc[start:end].to_numpy()))
            start = end
        order = np.asarray(order, dtype=int)
    else:
        order = np.arange(len(matrix))

    matrix = matrix.iloc[order]
    if row_labels is not None:
        row_labels = [row_labels[index] for index in order]
    if row_colors is not None:
        row_colors = row_colors[order]

    if column_clustering == "global":
        matrix = matrix.iloc[:, _leaf_order(matrix.to_numpy().T)]
    return matrix, row_labels, row_colors


def _normalization(matrix, ticks):
    _, colors = _matplotlib()
    limit = max(float(np.max(np.abs(matrix.to_numpy()))), 1e-12)
    if ticks is not None:
        ticks = np.asarray(ticks, dtype=float)
        if ticks.ndim != 1 or len(ticks) < 2 or not np.isfinite(ticks).all() or np.any(np.diff(ticks) <= 0):
            raise ValueError("colorbar_ticks must be increasing and finite")
        limit = float(np.max(np.abs(ticks)))
    norm = colors.Normalize(vmin=-limit, vmax=limit)
    if ticks is None:
        ticks = np.asarray(norm.inverse(np.linspace(0, 1, 5)))
        ticks[2] = 0
    return norm, limit, ticks


@lru_cache(maxsize=16384)
def _measure(text, fontsize, family, italic=False):
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    renderer = RendererAgg(1, 1, 72)
    prop = FontProperties(family=family, size=fontsize, style="italic" if italic else "normal")
    width, height, _ = renderer.get_text_width_height_descent(str(text), prop, False)
    return width / 72, height / 72


def _legend(fig, bounds, entries, title, columns, family, size, row_heights):
    from matplotlib.patches import Rectangle

    ax = fig.add_axes(bounds, label=f"annotation_legend_{len(fig.axes)}")
    ax.set_axis_off()
    w, h = bounds[2] * fig.get_figwidth(), bounds[3] * fig.get_figheight()
    ax.text(0, 1, title, ha="left", va="top", fontsize=size, family=family, color=_INK)
    for i, (label, color) in enumerate(entries):
        col, row = i % columns, i // columns
        x = col / columns
        y = 1 - (0.17 + sum(row_heights[:row]) + row_heights[row] / 2) / h
        ax.add_patch(Rectangle((x, y - 0.04 / h), 0.08 / w, 0.08 / h, facecolor=color, edgecolor="none"))
        ax.text(
            x + 0.11 / w, y, label, ha="left", va="center", fontsize=size, family=family, color=_INK, linespacing=1.05
        )
    return ax


def _heatmap(
    matrix,
    mask,
    *,
    row_labels,
    column_labels,
    row_colors=None,
    groups=(),
    colorbar_ticks=None,
    cmap=None,
    ax=None,
    colorbar_label="Effect size",
    xlabel="",
    ylabel="",
    italic_rows=True,
    italic_columns=False,
):
    plt, colors = _matplotlib()
    font_family = plt.rcParams["font.family"][0]
    fontsize, label_size, label_rotation = 9, 7.5, 60
    annotation_title, annotation_columns, annotation_size = "Gene annotation", 2, 7
    norm, limit, ticks = _normalization(matrix, colorbar_ticks)
    colormap = (
        colors.LinearSegmentedColormap.from_list("perturbvi_effect", _EFFECT_COLORS, N=257)
        if cmap is None
        else plt.get_cmap(cmap)
        if isinstance(cmap, str)
        else cmap
    )
    nrows, ncols = matrix.shape
    angle = np.deg2rad(label_rotation)
    col_extent = max(
        w * np.sin(angle) + h * np.cos(angle)
        for w, h in [_measure(x, label_size, font_family, italic_columns) for x in column_labels]
    )
    label_width = max(_measure(x, label_size, font_family, italic_rows)[0] for x in row_labels)
    annotation_columns = min(annotation_columns, max(1, len(groups)))
    top = col_extent + 0.13
    left = 0.26 if ylabel else 0.08
    strip = 0.17 if row_colors is not None else 0
    key_width = max(0.52, _measure(colorbar_label, 7, font_family)[0] + 0.02)
    right = strip + label_width + 0.10 + key_width + 0.09
    # Display annotation text verbatim, including any user-supplied line breaks.
    legend_row_heights = [
        0.15 + 0.11 * (max(label.count("\n") + 1 for label, _ in groups[i : i + annotation_columns]) - 1)
        for i in range(0, len(groups), annotation_columns)
    ]
    annotation_height = 0.19 + sum(legend_row_heights) if groups else 0
    bottom = 0.31 + (annotation_height + 0.09 if groups else 0)
    if ax is None:
        figsize = (max(4.64, 0.23 * ncols + left + right), max(2.35, min(18, 0.135 * nrows + top + bottom)))
        fig = plt.figure(figsize=figsize, facecolor="white")
        outer = [0, 0, figsize[0], figsize[1]]
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    else:
        fig = ax.figure
        pos = ax.get_position()
        outer = [
            pos.x0 * fig.get_figwidth(),
            pos.y0 * fig.get_figheight(),
            pos.width * fig.get_figwidth(),
            pos.height * fig.get_figheight(),
        ]
    x0, y0, width, height = outer
    plot_width, plot_height = width - left - right, height - top - bottom
    if plot_width < 0.3 or plot_height < 0.3:
        if len(fig.axes) == 1:
            plt.close(fig)
        raise ValueError(
            "Figure is too small for these labels; use a larger Matplotlib figure or shorten display labels"
        )
    fw, fh = fig.get_size_inches()

    def bounds(x, y, w, h):
        return [(x0 + x) / fw, (y0 + y) / fh, w / fw, h / fh]

    ax.set_position(bounds(left, bottom, plot_width, plot_height))
    ax.imshow(
        matrix.to_numpy(),
        cmap=colormap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        rasterized=(matrix.size > 100000),
    )
    ax.set(xlim=(-0.5, ncols - 0.5), ylim=(nrows - 0.5, -0.5))
    # Keep all cells; only thin labels when a complete matrix is very dense.
    row_step = max(1, math.ceil(nrows * label_size * 1.15 / (72 * plot_height)))
    col_step = max(1, math.ceil(ncols * label_size * 0.9 / (72 * plot_width)))
    ri, ci = np.arange(0, nrows, row_step), np.arange(0, ncols, col_step)
    if row_step > 1 or col_step > 1:
        warnings.warn(
            "Dense matrix: tick labels were thinned; all selected cells are displayed", UserWarning, stacklevel=3
        )
    ax.set_xticks(
        ci,
        [column_labels[i] for i in ci],
        rotation=label_rotation,
        ha="left" if label_rotation else "center",
        va="bottom",
        rotation_mode="anchor",
    )
    ax.set_yticks(ri, [row_labels[i] for i in ri])
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis="both", which="both", length=0, pad=4, colors=_INK, labelsize=label_size)
    ax.tick_params(axis="y", pad=4 + 72 * strip)
    for tick in ax.get_xticklabels():
        tick.set(fontfamily=font_family, fontstyle="italic" if italic_columns else "normal")
    for tick in ax.get_yticklabels():
        tick.set(ha="left", fontfamily=font_family, fontstyle="italic" if italic_rows else "normal")
    ax.xaxis.set_label_position("bottom")
    ax.yaxis.set_label_position("left")
    ax.set_xlabel(xlabel, fontsize=fontsize, color=_INK, family=font_family, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=fontsize, color=_INK, family=font_family, labelpad=6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set(visible=True, linewidth=0.25 * 72 / 25.4, color="#555555")
    if mask is not None:
        ys, xs = np.nonzero(mask.to_numpy())
        ax.scatter(xs, ys, s=4, c="#242424", linewidths=0)
    if row_colors is not None:
        strip_ax = fig.add_axes(
            bounds(left + plot_width + 0.06, bottom, 0.11, plot_height), label=f"gene_annotation_{len(fig.axes)}"
        )
        strip_ax.imshow(row_colors[:, None, :], aspect="auto", interpolation="nearest")
        strip_ax.set_axis_off()
    if groups:
        _legend(
            fig,
            bounds(left, 0.02, width - left - 0.08, annotation_height),
            groups,
            annotation_title,
            annotation_columns,
            font_family,
            annotation_size,
            legend_row_heights,
        )
    # Fixed physical colorbar size matches the compact publication heatmaps.
    key_length = min(1.0, plot_height)
    key_x = width - key_width - 0.04
    key_y = bottom + (plot_height - key_length) / 2
    key = fig.add_axes(bounds(key_x, key_y, 0.10, key_length), label=f"effect_colorbar_{len(fig.axes)}")
    key.imshow(
        np.linspace(0, 1, 257)[:, None],
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 1],
        cmap=colormap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    key.set_axis_off()
    for value in ticks:
        position = float(norm(value))
        key.plot([0, 0.36], [position, position], color="white", lw=0.65, clip_on=False)
        key.plot([0.64, 1], [position, position], color="white", lw=0.65, clip_on=False)
        label = "0" if round(float(value), 1) == 0 else f"{value:.1f}"
        key.text(
            1.45,
            position,
            label,
            ha="left",
            va="center",
            transform=key.transAxes,
            fontsize=6.5,
            color=_INK,
            family=font_family,
        )
    key.text(
        0,
        1 + 0.09 / key_length,
        colorbar_label,
        ha="left",
        va="bottom",
        transform=key.transAxes,
        fontsize=7,
        color=_INK,
        family=font_family,
    )
    # Triangular caps make explicit color saturation visible.
    from matplotlib.patches import Polygon

    low, high = np.min(matrix.to_numpy()) < -limit, np.max(matrix.to_numpy()) > limit
    for clipped, y, tip, color in [(low, 0, -0.05, colormap(0.0)), (high, 1, 1.05, colormap(1.0))]:
        if clipped:
            key.add_patch(
                Polygon(
                    [[0, y], [1, y], [0.5, tip]],
                    facecolor=color,
                    edgecolor="none",
                    transform=key.transAxes,
                    clip_on=False,
                )
            )
    return fig


def plot_B(
    B: pd.DataFrame,
    *,
    pip: pd.DataFrame | None = None,
    perturbations: Sequence | None = None,
    factors: Sequence | None = None,
    row_clustering: str = "none",
    column_clustering: str = "none",
    show_significance: bool = False,
    cmap=None,
    colorbar_ticks: Sequence | None = None,
    ax=None,
):
    """Plot a perturbation-by-factor posterior mean effect matrix ``B``.

    Read ``B.csv`` with ``pd.read_csv(..., index_col=0)``. Rows identify
    perturbations and columns identify factors. Optional selections use these
    IDs, e.g. ``factors=['factor_0']`` displays ``Factor 1``. Perturbation labels
    are italic automatically. No result dictionary or saved model is needed.

    Dots are off by default. To mark PIP > 0.95, pass ``show_significance=True``
    and ``pip=PIP_B``, a DataFrame with the same identifiers and orientation.
    Supplying ``pip`` alone does not enable dots. Probabilities are aligned by
    identifiers; their row and column order need not match ``B``.

    Rows and columns retain their supplied order by default. Set
    ``row_clustering`` or ``column_clustering`` to ``"global"`` to order that
    axis by Euclidean distance and Ward's method. Clustering changes display
    order only; dendrograms are not drawn.

    Legend labels retain original units. ``cmap`` accepts a Matplotlib colormap name
    or object. By default, five markers are evenly spaced on the displayed
    scale, including zero and both limits, with labels rounded to one
    decimal place. The symmetric color range covers the displayed values;
    supplied ``colorbar_ticks`` set a symmetric range using their largest
    absolute value. Triangular caps mark saturation. Use identical ticks for
    comparable panels. Typography and spacing are automatic.

    Returns a Matplotlib Figure. Use ``ax`` for composition, save with
    ``fig.savefig()``, or customize the returned artists with Matplotlib.
    """
    B = _table(B, "B")
    matrix = _select(B, perturbations, factors)
    matrix, _, _ = _cluster_matrix(matrix, row_clustering=row_clustering, column_clustering=column_clustering)
    mask = _mask(B, pip, "pip", matrix, 0.95 if show_significance else None, "greater")
    return _heatmap(
        matrix,
        mask,
        row_labels=_labels(matrix.index),
        column_labels=_labels(matrix.columns, factors=True),
        cmap=cmap,
        colorbar_ticks=colorbar_ticks,
        ax=ax,
        xlabel="Factors",
        ylabel="Perturbations",
    )


def plot_W(
    W: pd.DataFrame,
    *,
    pip: pd.DataFrame | None = None,
    genes: Sequence | None = None,
    factors: Sequence | None = None,
    gene_annotations: pd.DataFrame | None = None,
    gene_annotation_columns: Mapping[str, str] | None = None,
    row_clustering: str = "none",
    column_clustering: str = "none",
    show_significance: bool = False,
    cmap=None,
    colorbar_ticks: Sequence | None = None,
    ax=None,
):
    """Plot a factor-by-gene posterior mean loading matrix ``W``.

    Read ``W.csv`` with ``pd.read_csv(..., index_col=0)``. The function
    transposes it internally to display genes on rows and factors on columns.
    Loadings are already inclusion-weighted and are not multiplied by PIP again.

    Dots are off by default. Pass ``pip=PIP_W`` and ``show_significance=True``
    to mark PIP > 0.95. ``PIP_W`` must have the same factor-by-gene identifiers
    as ``W``; it is aligned and transposed internally.

    Optional ``gene_annotations`` is a DataFrame with gene identifiers,
    display names, and annotation groups. By default, its columns are named
    ``gene_ID``, ``gene_name``, and ``annotation``. Use
    ``gene_annotation_columns`` to map those roles to any input headers. Genes
    are grouped by annotation in first-seen order; ``genes`` controls order
    within each group unless clustering is requested. Unannotated genes appear
    last. Without annotations, the supplied gene order stays unchanged.
    Annotation text labels the two-column bottom legend verbatim, with no
    automatic wrapping. Gene names are italic automatically.

    ``row_clustering`` may be ``"none"`` (default), ``"global"``, or
    ``"within_annotation"``. The last option requires annotations and keeps
    annotation groups together in their first-seen order while clustering genes
    inside each group. ``column_clustering`` may be ``"none"`` or ``"global"``.
    Clustering uses Euclidean distance and Ward's method, reorders the heatmap,
    and does not draw dendrograms.

    Returned Figure follows :func:`plot_B`. Omitting
    selections includes every gene and factor. Dense labels may be thinned;
    all selected cells remain present in the heatmap.
    """
    W = _table(W, "W")
    matrix = _select(W.T, genes, factors)
    matrix, rows, row_colors, groups, row_groups = _annotation(matrix, gene_annotations, gene_annotation_columns)
    matrix, rows, row_colors = _cluster_matrix(
        matrix,
        row_clustering=row_clustering,
        column_clustering=column_clustering,
        row_labels=rows,
        row_colors=row_colors,
        row_groups=row_groups,
    )
    mask = _mask(W, pip, "pip", matrix, 0.95 if show_significance else None, "greater", transpose=True)
    return _heatmap(
        matrix,
        mask,
        row_labels=rows,
        column_labels=_labels(matrix.columns, factors=True),
        row_colors=row_colors,
        groups=groups,
        cmap=cmap,
        colorbar_ticks=colorbar_ticks,
        ax=ax,
        colorbar_label="Gene loading",
        xlabel="Factors",
        ylabel="Genes",
    )


def plot_BW(
    BW: pd.DataFrame,
    *,
    lfsr: pd.DataFrame | None = None,
    genes: Sequence | None = None,
    perturbations: Sequence | None = None,
    gene_annotations: pd.DataFrame | None = None,
    gene_annotation_columns: Mapping[str, str] | None = None,
    row_clustering: str = "none",
    column_clustering: str = "none",
    show_significance: bool = False,
    cmap=None,
    colorbar_ticks: Sequence | None = None,
    ax=None,
):
    """Plot a perturbation-by-gene overall-effect matrix ``BW``.

    Read ``BW.csv`` with ``pd.read_csv(..., index_col=0)``. The function
    transposes it internally to display genes on rows and perturbations on
    columns. It plots the supplied effects without recomputing a factor
    product, thresholding colors, or subtracting a reference condition.

    Dots are off by default. Pass ``lfsr=LFSR_BW`` and
    ``show_significance=True`` to mark LFSR < 0.05. ``LFSR_BW`` must have the
    same perturbation-by-gene identifiers as ``BW``. LFSR is aligned and
    transposed internally; it is never computed by the plotting function.
    No LFSR table is required when dots are off.

    Annotation, color, and clustering options follow :func:`plot_W`.
    Gene and perturbation names are italic automatically. Returns a Matplotlib
    Figure.
    """
    BW = _table(BW, "BW")
    matrix = _select(BW.T, genes, perturbations)
    matrix, rows, row_colors, groups, row_groups = _annotation(matrix, gene_annotations, gene_annotation_columns)
    matrix, rows, row_colors = _cluster_matrix(
        matrix,
        row_clustering=row_clustering,
        column_clustering=column_clustering,
        row_labels=rows,
        row_colors=row_colors,
        row_groups=row_groups,
    )
    mask = _mask(BW, lfsr, "lfsr", matrix, 0.05 if show_significance else None, "less", transpose=True)
    return _heatmap(
        matrix,
        mask,
        row_labels=rows,
        column_labels=_labels(matrix.columns),
        row_colors=row_colors,
        groups=groups,
        cmap=cmap,
        colorbar_ticks=colorbar_ticks,
        ax=ax,
        colorbar_label="Overall effect",
        xlabel="Perturbations",
        ylabel="Genes",
        italic_columns=True,
    )
