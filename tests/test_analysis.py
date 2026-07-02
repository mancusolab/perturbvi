import numpy as np
import pytest

from perturbvi import analyze, infer, load_results, save_results


@pytest.fixture
def small_results(tmp_path):
    rng = np.random.default_rng(0)
    n, g, p = 50, 15, 4
    X = rng.normal(0, 1, (n, g))
    G = (rng.random((n, p)) > 0.5).astype(float)
    for col in range(p):
        if G[:, col].sum() == 0:
            G[0, col] = 1.0
    results = infer(X, z_dim=2, l_dim=5, G=G, tau=50, max_iter=5, seed=0)
    save_results(results, path=str(tmp_path))
    return results, tmp_path


def test_analyze_lfsr_none_by_default(small_results):
    results, _ = small_results
    tables = analyze(results, genes=None, perturbations=None)
    assert tables["lfsr"] is None
    assert "pip_df" in tables
    assert "pve_df" in tables
    assert "overall_effect_df" in tables


def test_analyze_compute_lfsr(small_results):
    results, _ = small_results
    tables = analyze(results, genes=None, perturbations=None, compute_lfsr=True, lfsr_iters=50)
    assert tables["lfsr"].shape == (15, 4)


def test_analyze_compute_lfsr_saves_to_path(small_results):
    results, tmp_path = small_results
    tables = analyze(
        results, genes=None, perturbations=None, compute_lfsr=True, lfsr_iters=50, path=str(tmp_path)
    )
    assert tables["lfsr"] is not None
    assert (tmp_path / "lfsr.csv").exists()


def test_analyze_reuses_cached_lfsr(small_results):
    results, tmp_path = small_results
    computed = analyze(
        results, genes=None, perturbations=None, compute_lfsr=True, lfsr_iters=50, path=str(tmp_path)
    )

    cached = analyze(results, genes=None, perturbations=None, path=str(tmp_path))

    assert cached["lfsr"] is not None
    np.testing.assert_allclose(cached["lfsr"].to_numpy(), computed["lfsr"].to_numpy())


def test_analyze_warns_when_lfsr_missing_at_path(small_results):
    results, tmp_path = small_results
    with pytest.warns(UserWarning, match="Could not locate lfsr.csv"):
        tables = analyze(results, genes=None, perturbations=None, path=str(tmp_path))
    assert tables["lfsr"] is None


def test_analyze_from_loaded_results(small_results):
    _, tmp_path = small_results
    fitted = load_results(str(tmp_path))
    tables = analyze(fitted, genes=None, perturbations=None)
    assert "pip_df" in tables
    assert "pve_df" in tables


def test_analyze_output_still_importable():
    from perturbvi.utils import analyze_output

    assert callable(analyze_output)


def test_analyze_with_names(small_results):
    results, _ = small_results
    gene_names = [f"gene_{i}" for i in range(15)]
    pert_names = [f"pert_{i}" for i in range(4)]
    tables = analyze(results, genes=gene_names, perturbations=pert_names)
    # pip_df: rows=genes, cols=factors
    assert list(tables["pip_df"].index) == gene_names
    # overall_effect_df: rows=genes, cols=perturbations
    assert list(tables["overall_effect_df"].columns) == pert_names
