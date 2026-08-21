import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from .helpers import write_10x_h5


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


def test_fit_categorical_covariates_require_covariates():
    from perturbvi.cli import main

    with pytest.raises(SystemExit):
        main([
            "fit", "input.h5ad", "--output", "out", "--z-dim", "2", "--l-dim", "5",
            "--tau", "50", "--guide-key", "x", "--categorical-covariates", "batch",
        ])


def test_delimited_header_and_index_accept_none(monkeypatch):
    from perturbvi import cli

    captured = {}

    def fake_setup(device):
        captured["device"] = device
        return "cpu:0"

    def fake_fit(args, log, selected_device):
        captured["header"] = args.header
        captured["index_col"] = args.index_col

    monkeypatch.setattr(cli, "_setup_jax", fake_setup)
    monkeypatch.setattr(cli, "_cmd_fit", fake_fit)
    cli.main([
        "fit", "expression.csv", "--format", "csv", "--guide-matrix", "guides.csv",
        "--output", "out", "--z-dim", "1", "--l-dim", "1", "--tau", "1",
        "--header", "none", "--index-col", "none",
    ])
    assert captured == {"device": "cpu", "header": None, "index_col": None}


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
    assert config["preprocessing_steps"] == ["standardize"]
    assert config["preprocessing_order"] == "standardize"


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
        "--categorical-covariates", "batch",
        "--max-iter", "3",
    ])
    config = json.loads((out / "run_config.json").read_text())
    assert config["covariates"] == ["batch"]
    assert config["categorical_covariates"] == ["batch"]
    assert config["preprocessing_steps"] == ["residualize", "standardize"]
    assert config["preprocessing_order"] == "residualize_then_standardize"
    summary = json.loads((out / "input_summary.json").read_text())
    assert summary["source"]["residualized"] is True
    assert summary["source"]["residualized_covariate_names"] == ["batch"]


def test_analyze_smoke_without_and_with_lfsr(h5ad_path, tmp_path):
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
    assert "lfsr.csv" not in names
    pip_table = pd.read_csv(analyze_out / "pip_df.csv", index_col=0)
    assert list(pip_table.index) == [f"gene_{index}" for index in range(8)]

    lfsr_out = tmp_path / "analysis_lfsr"
    main([
        "analyze", str(fit_out),
        "--output", str(lfsr_out),
        "--compute-lfsr",
        "--lfsr-iters", "50",
    ])

    assert (lfsr_out / "lfsr.csv").exists()


def test_fit_smoke_10x_h5(tmp_path):
    from perturbvi.cli import main

    input_path = write_10x_h5(tmp_path / "screen.h5")
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"target": ["control", "target_a", "target_b", "target_a", "target_b", "control"]},
        index=[f"cell_{index}" for index in range(6)],
    ).to_csv(metadata_path)
    output = tmp_path / "tenx_results"
    main(
        [
            "fit",
            str(input_path),
            "--format",
            "10x-h5",
            "--metadata",
            str(metadata_path),
            "--guide-key",
            "target",
            "--control-label",
            "control",
            "--output",
            str(output),
            "--z-dim",
            "1",
            "--l-dim",
            "1",
            "--tau",
            "5",
            "--max-iter",
            "1",
        ]
    )

    assert (output / "params_file.pkl").is_file()
    assert (output / "gene_names.txt").read_text().splitlines() == ["gene_1", "gene_2", "gene_3"]
    assert (output / "perturbation_names.txt").read_text().splitlines() == ["target_a", "target_b"]


def test_fit_smoke_csv(tmp_path):
    from perturbvi.cli import main

    cells = [f"c{index}" for index in range(8)]
    expression_path = tmp_path / "expression.csv"
    guide_path = tmp_path / "guides.csv"
    pd.DataFrame(
        np.arange(24, dtype=float).reshape(8, 3),
        index=cells,
        columns=["g1", "g2", "g3"],
    ).to_csv(expression_path)
    pd.DataFrame(
        [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        index=cells,
        columns=["p1", "p2"],
    ).to_csv(guide_path)

    output = tmp_path / "csv_results"
    main([
        "fit", str(expression_path), "--guide-matrix", str(guide_path),
        "--output", str(output), "--z-dim", "1", "--l-dim", "1", "--tau", "5",
        "--max-iter", "1",
    ])
    assert (output / "params_file.pkl").is_file()
    assert json.loads((output / "run_config.json").read_text())["format"] == "csv"
