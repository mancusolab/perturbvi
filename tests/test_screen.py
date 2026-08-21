import numpy as np
import pytest

from jax.experimental import sparse as jax_sparse

from perturbvi.screen import ScreenData, validate_screen


def _make_screen(n=12, p=20, g=3):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    G = np.zeros((n, g))
    for row in range(n):
        G[row, row % g] = 1.0
    return ScreenData(X=X, G=G)


def test_valid_screen_with_names():
    original = _make_screen(n=9, p=20, g=3)
    screen = original._replace(
        gene_names=[f"gene_{index}" for index in range(20)],
        perturbation_names=["pert_a", "pert_b", "pert_c"],
        cell_names=[f"cell_{index}" for index in range(9)],
        source={"format": "test"},
    )
    validate_screen(screen)


@pytest.mark.parametrize(
    ("screen", "message"),
    [
        (ScreenData(X=np.ones(10), G=np.ones((10, 2))), "2D"),
        (ScreenData(X=np.ones((10, 5)), G=np.ones(10)), "2D"),
        (ScreenData(X=np.ones((10, 5)), G=np.ones((8, 2))), "rows"),
    ],
)
def test_invalid_dimensions_raise(screen, message):
    with pytest.raises(ValueError, match=message):
        validate_screen(screen)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_nonfinite_x_raises(nonfinite):
    screen = _make_screen()._replace(X=np.ones((12, 20)))
    values = np.asarray(screen.X).copy()
    values[0, 0] = nonfinite
    with pytest.raises(ValueError, match="non-finite"):
        validate_screen(screen._replace(X=values))


def test_nonfinite_g_raises():
    screen = _make_screen()
    guides = np.asarray(screen.G).copy()
    guides[0, 0] = np.nan
    with pytest.raises(ValueError, match="G contains non-finite"):
        validate_screen(screen._replace(G=guides))


def test_empty_g_column_raises():
    screen = _make_screen()
    guides = np.column_stack([screen.G, np.zeros(screen.G.shape[0])])
    with pytest.raises(ValueError, match="all-zero"):
        validate_screen(screen._replace(G=guides))


def test_zero_column_g_shape_raises():
    screen = _make_screen()
    with pytest.raises(ValueError, match="at least one perturbation"):
        validate_screen(screen._replace(G=np.empty((12, 0)), perturbation_names=[]))


def test_name_dimension_mismatches_raise():
    screen = _make_screen()
    with pytest.raises(ValueError, match="gene_names"):
        validate_screen(screen._replace(gene_names=["g"] * 5))
    with pytest.raises(ValueError, match="perturbation_names"):
        validate_screen(screen._replace(perturbation_names=["p1", "p2"]))
    with pytest.raises(ValueError, match="cell_names"):
        validate_screen(screen._replace(cell_names=["cell"] * 5))


def test_validate_screen_accepts_bcsr_without_materializing():
    screen = _make_screen()
    sparse_screen = screen._replace(
        X=jax_sparse.BCSR.fromdense(screen.X),
        G=jax_sparse.BCSR.fromdense(screen.G),
    )
    validate_screen(sparse_screen)


def test_validate_screen_rejects_nonmapping_source():
    screen = _make_screen()._replace(source="not metadata")
    with pytest.raises(ValueError, match="source must be a mapping"):
        validate_screen(screen)


def test_missing_categorical_covariate_raises():
    screen = _make_screen()._replace(
        covariates=np.array([["A"]] * 11 + [[None]], dtype=object),
        covariate_names=["batch"],
    )
    with pytest.raises(ValueError, match="missing values"):
        validate_screen(screen)


def test_nonfinite_numeric_covariate_raises():
    values = np.arange(12, dtype=float).reshape(-1, 1)
    values[4, 0] = np.inf
    screen = _make_screen()._replace(covariates=values, covariate_names=["depth"])
    with pytest.raises(ValueError, match="non-finite"):
        validate_screen(screen)
