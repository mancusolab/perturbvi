import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def h5ad_path(tmp_path):
    rng = np.random.default_rng(0)
    n, g = 40, 8
    X = rng.poisson(4, (n, g)).astype(float)
    obs = pd.DataFrame(
        {
            "perturbation": rng.choice(["geneA", "geneB", "non-targeting"], n),
            "batch": rng.choice(["A", "B"], n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(g)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    path = tmp_path / "screen.h5ad"
    adata.write_h5ad(path)
    return path


# --- Parser tests (no file I/O, no JAX) ---

def test_fit_missing_required_arg():
    from perturbvi.cli import main

    with pytest.raises(SystemExit):
        main(["fit", "input.h5ad", "--output", "out"])  # missing --z-dim, --l-dim, --tau


def test_fit_invalid_format():
    from perturbvi.cli import main

    with pytest.raises(SystemExit):
        main(["fit", "input.h5ad", "--output", "out", "--z-dim", "2", "--l-dim", "5",
              "--tau", "50", "--format", "bad_format"])


def test_analyze_missing_results_dir():
    from perturbvi.cli import main

    with pytest.raises(SystemExit):
        main(["analyze"])  # no results_dir


def test_fit_multi_guide_choices():
    from perturbvi.cli import main

    with pytest.raises(SystemExit):
        main(["fit", "input.h5ad", "--output", "out", "--z-dim", "2", "--l-dim", "5",
              "--tau", "50", "--guide-key", "x", "--multi-guide", "invalid"])


# --- Smoke tests (real file I/O + inference) ---

def test_fit_smoke_h5ad(h5ad_path, tmp_path):
    from perturbvi.cli import main

    out = tmp_path / "results"
    main([
        "fit", str(h5ad_path),
        "--output", str(out),
        "--z-dim", "2",
        "--l-dim", "5",
        "--tau", "50",
        "--guide-key", "perturbation",
        "--control-label", "non-targeting",
        "--max-iter", "3",
        "--seed", "0",
    ])
    assert (out / "W.txt").exists()
    assert (out / "pip.txt").exists()
    assert (out / "pve.txt").exists()
    assert (out / "params_file.pkl").exists()
    assert (out / "run_config.json").exists()
    assert (out / "input_summary.json").exists()

    config = json.loads((out / "run_config.json").read_text())
    assert config["z_dim"] == 2
    assert config["seed"] == 0


def test_fit_smoke_with_covariates(h5ad_path, tmp_path):
    from perturbvi.cli import main

    out = tmp_path / "results_cov"
    main([
        "fit", str(h5ad_path),
        "--output", str(out),
        "--z-dim", "2",
        "--l-dim", "5",
        "--tau", "50",
        "--guide-key", "perturbation",
        "--covariates", "batch",
        "--categoricals", "batch",
        "--max-iter", "3",
    ])
    config = json.loads((out / "run_config.json").read_text())
    assert config["covariates"] == ["batch"]
    assert config["categoricals"] == ["batch"]


def test_analyze_smoke(h5ad_path, tmp_path):
    from perturbvi.cli import main

    fit_out = tmp_path / "fit_results"
    main([
        "fit", str(h5ad_path),
        "--output", str(fit_out),
        "--z-dim", "2", "--l-dim", "5", "--tau", "50",
        "--guide-key", "perturbation",
        "--max-iter", "3",
    ])

    analyze_out = tmp_path / "analysis"
    main(["analyze", str(fit_out), "--output", str(analyze_out)])

    csvs = list(analyze_out.glob("*.csv"))
    assert len(csvs) > 0
    names = [f.name for f in csvs]
    assert "pip_df.csv" in names
    assert "lfsr_df.csv" not in names


def test_analyze_smoke_with_lfsr(h5ad_path, tmp_path):
    from perturbvi.cli import main

    fit_out = tmp_path / "fit_results"
    main([
        "fit", str(h5ad_path),
        "--output", str(fit_out),
        "--z-dim", "2", "--l-dim", "5", "--tau", "50",
        "--guide-key", "perturbation",
        "--max-iter", "3",
    ])

    analyze_out = tmp_path / "analysis_lfsr"
    main([
        "analyze", str(fit_out),
        "--output", str(analyze_out),
        "--compute-lfsr",
        "--lfsr-iters", "50",
    ])

    assert (analyze_out / "lfsr_df.csv").exists()
