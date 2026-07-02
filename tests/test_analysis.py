import numpy as np
import pytest

from perturbvi import analyze_results, analyze_saved_results, infer, save_results


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


def test_analyze_no_lfsr(small_results):
    results, _ = small_results
    tables = analyze_results(results, gene_names=None, perturbation_names=None)
    assert "lfsr_df" not in tables
    assert "pip_df" in tables
    assert "pve_df" in tables
    assert "overall_effect_df" in tables


def test_analyze_with_lfsr(small_results):
    results, _ = small_results
    tables = analyze_results(
        results,
        gene_names=None,
        perturbation_names=None,
        compute_lfsr=True,
        lfsr_iters=50,
        seed=0,
    )
    assert "lfsr_df" in tables


def test_analyze_lfsr_not_called_by_default(small_results):
    results, _ = small_results
    tables = analyze_results(results, gene_names=None, perturbation_names=None, compute_lfsr=False)
    assert "lfsr_df" not in tables


def test_analyze_saved_results(small_results):
    results, tmp_path = small_results
    tables = analyze_saved_results(str(tmp_path), gene_names=None, perturbation_names=None)
    assert "pip_df" in tables
    assert "pve_df" in tables


def test_analyze_output_still_importable():
    from perturbvi.utils import analyze_output

    assert callable(analyze_output)


def test_analyze_with_names(small_results):
    results, _ = small_results
    gene_names = [f"gene_{i}" for i in range(15)]
    pert_names = [f"pert_{i}" for i in range(4)]
    tables = analyze_results(results, gene_names=gene_names, perturbation_names=pert_names)
    # pip_df: rows=genes, cols=factors
    assert list(tables["pip_df"].index) == gene_names
    # overall_effect_df: rows=genes, cols=perturbations
    assert list(tables["overall_effect_df"].columns) == pert_names
