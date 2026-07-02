import numpy as np
import pytest

from perturbvi.screen import ScreenData, validate_screen


def _make_screen(n=12, p=20, g=3):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    G = np.zeros((n, g))
    for i in range(n):
        G[i, i % g] = 1.0
    return ScreenData(X=X, G=G, gene_names=None, perturbation_names=None, cell_names=None, source={})


def test_valid_screen_passes():
    validate_screen(_make_screen())


def test_valid_screen_with_names():
    s = _make_screen(n=9, p=20, g=3)
    screen = ScreenData(
        X=s.X,
        G=s.G,
        gene_names=[f"gene_{i}" for i in range(20)],
        perturbation_names=["pert_a", "pert_b", "pert_c"],
        cell_names=[f"cell_{i}" for i in range(9)],
        source={"format": "test"},
    )
    validate_screen(screen)


def test_x_not_2d_raises():
    screen = ScreenData(
        X=np.ones(10), G=np.ones((10, 2)),
        gene_names=None, perturbation_names=None, cell_names=None, source={}
    )
    with pytest.raises(ValueError, match="2D"):
        validate_screen(screen)


def test_g_not_2d_raises():
    screen = ScreenData(
        X=np.ones((10, 5)), G=np.ones(10),
        gene_names=None, perturbation_names=None, cell_names=None, source={}
    )
    with pytest.raises(ValueError, match="2D"):
        validate_screen(screen)


def test_row_mismatch_raises():
    screen = ScreenData(
        X=np.ones((10, 5)), G=np.ones((8, 2)),
        gene_names=None, perturbation_names=None, cell_names=None, source={}
    )
    with pytest.raises(ValueError, match="rows"):
        validate_screen(screen)


def test_nan_in_x_raises():
    X = np.ones((10, 5))
    X[2, 3] = np.nan
    G = np.zeros((10, 2))
    G[:5, 0] = 1.0
    G[5:, 1] = 1.0
    screen = ScreenData(X=X, G=G, gene_names=None, perturbation_names=None, cell_names=None, source={})
    with pytest.raises(ValueError, match="non-finite"):
        validate_screen(screen)


def test_inf_in_x_raises():
    X = np.ones((10, 5))
    X[0, 0] = np.inf
    G = np.zeros((10, 2))
    G[:5, 0] = 1.0
    G[5:, 1] = 1.0
    screen = ScreenData(X=X, G=G, gene_names=None, perturbation_names=None, cell_names=None, source={})
    with pytest.raises(ValueError, match="non-finite"):
        validate_screen(screen)


def test_empty_g_column_raises():
    X = np.ones((10, 5))
    G = np.zeros((10, 3))
    G[:, 0] = 1.0  # columns 1 and 2 are all-zero
    screen = ScreenData(X=X, G=G, gene_names=None, perturbation_names=None, cell_names=None, source={})
    with pytest.raises(ValueError, match="all-zero"):
        validate_screen(screen)


def test_gene_names_mismatch_raises():
    s = _make_screen(p=20)
    screen = ScreenData(
        X=s.X, G=s.G,
        gene_names=["g"] * 5,  # wrong: should be 20
        perturbation_names=None, cell_names=None, source={}
    )
    with pytest.raises(ValueError, match="gene_names"):
        validate_screen(screen)


def test_perturbation_names_mismatch_raises():
    s = _make_screen(g=3)
    screen = ScreenData(
        X=s.X, G=s.G,
        gene_names=None,
        perturbation_names=["p1", "p2"],  # wrong: should be 3
        cell_names=None, source={}
    )
    with pytest.raises(ValueError, match="perturbation_names"):
        validate_screen(screen)


def test_cell_names_mismatch_raises():
    s = _make_screen(n=12)
    screen = ScreenData(
        X=s.X, G=s.G,
        gene_names=None, perturbation_names=None,
        cell_names=["c"] * 5,  # wrong: should be 12
        source={}
    )
    with pytest.raises(ValueError, match="cell_names"):
        validate_screen(screen)
