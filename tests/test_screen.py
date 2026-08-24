from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from jax.experimental import sparse as jax_sparse

from perturbvi import PerturbData
from perturbvi.screen import validate_screen


def _make_data(n_cells=24, n_genes=40, n_perturbations=4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_cells, n_genes))
    G = np.zeros((n_cells, n_perturbations))
    G[np.arange(n_cells), np.arange(n_cells) % n_perturbations] = 1
    return PerturbData(
        X=X,
        G=G,
        gene_names=[f"gene_{index}" for index in range(n_genes)],
        perturbation_names=[f"perturbation_{index}" for index in range(n_perturbations)],
    )


def test_dataframe_inputs_infer_names_without_storing_cell_names():
    cells = [f"cell_{index}" for index in range(6)]
    X = pd.DataFrame(np.ones((6, 3)), index=cells, columns=["A", "B", "C"])
    G = pd.DataFrame(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]],
        index=cells,
        columns=["guide_A", "guide_B"],
    )

    data = PerturbData(X=X, G=G)
    validate_screen(data)

    assert data.gene_names == ("A", "B", "C")
    assert data.perturbation_names == ("guide_A", "guide_B")
    assert not hasattr(data, "cell_names")


def test_dataframe_row_order_must_match_when_indexes_are_available():
    X = pd.DataFrame(np.ones((3, 2)), index=["c1", "c2", "c3"])
    G = pd.DataFrame(np.ones((3, 1)), index=["c2", "c1", "c3"])
    with pytest.raises(ValueError, match="row indexes"):
        PerturbData(X=X, G=G)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (PerturbData(X=np.ones(10), G=np.ones((10, 2))), "2D"),
        (PerturbData(X=np.ones((10, 5)), G=np.ones(10)), "2D"),
        (PerturbData(X=np.ones((10, 5)), G=np.ones((8, 2))), "rows"),
    ],
)
def test_matrix_dimensions_are_core_invariants(data, message):
    with pytest.raises(ValueError, match=message):
        validate_screen(data)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_expression_must_be_finite(value):
    data = _make_data()
    X = np.asarray(data.X).copy()
    X[0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        validate_screen(replace(data, X=X))


@pytest.mark.parametrize("value", [np.nan, 0.5, -1, 2])
def test_perturbation_design_must_be_finite_and_binary(value):
    data = _make_data()
    G = np.asarray(data.G).copy()
    G[0, 0] = value
    with pytest.raises(ValueError, match="non-finite|binary"):
        validate_screen(replace(data, G=G))


def test_all_zero_rows_are_valid_but_all_zero_columns_are_not():
    data = _make_data()
    G = np.asarray(data.G).copy()
    G[:3] = 0
    validate_screen(replace(data, G=G))

    G[:, 0] = 0
    with pytest.raises(ValueError, match="all-zero columns"):
        validate_screen(replace(data, G=G))


def test_names_are_required_unique_and_dimensioned():
    data = _make_data()
    with pytest.raises(ValueError, match="gene_names"):
        validate_screen(replace(data, gene_names=None))
    with pytest.raises(ValueError, match="duplicate"):
        validate_screen(replace(data, perturbation_names=["same"] * data.G.shape[1]))


def test_sparse_bcsr_is_valid_without_materializing_the_full_matrix():
    data = _make_data(n_cells=128, n_genes=512, n_perturbations=16)
    sparse_data = replace(
        data,
        X=jax_sparse.BCSR.fromdense(data.X),
        G=jax_sparse.BCSR.fromdense(data.G),
    )
    validate_screen(sparse_data)


def test_covariates_accept_numeric_boolean_category_and_string_columns():
    data = _make_data()
    n = data.X.shape[0]
    covariates = pd.DataFrame(
        {
            "depth": np.linspace(1, 2, n),
            "passed_qc": np.resize([True, False], n),
            "batch": pd.Categorical(np.resize(["A", "B"], n)),
            "donor": np.resize(["D1", "D2", "D3"], n),
        }
    )
    validate_screen(replace(data, covariates=covariates))


def test_covariates_reject_only_broken_alignment_or_values():
    data = _make_data()
    with pytest.raises(ValueError, match="rows"):
        validate_screen(replace(data, covariates=pd.DataFrame({"x": [1, 2]})))

    values = np.arange(data.X.shape[0], dtype=float)
    values[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_screen(replace(data, covariates=pd.DataFrame({"depth": values})))
