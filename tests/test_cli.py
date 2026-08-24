import argparse
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from perturbvi._defaults import (
    DEFAULT_INIT,
    DEFAULT_MAX_ITER,
    DEFAULT_STANDARDIZE,
    DEFAULT_TAU,
    DEFAULT_TOL,
    DEFAULT_VERBOSE,
)


@pytest.fixture
def h5ad_path(tmp_path):
    rng = np.random.default_rng(0)
    n_cells, n_genes = 48, 12
    conditions = np.resize(["control", "IRF1", "STAT1"], n_cells)
    adata = ad.AnnData(
        X=rng.normal(size=(n_cells, n_genes)),
        obs=pd.DataFrame(
            {
                "condition": conditions,
                "batch": np.resize(["run_1", "run_2"], n_cells),
                "guide_A": np.resize([0, 1, 0, 1], n_cells),
                "guide_B": np.resize([0, 0, 1, 1], n_cells),
            },
            index=[f"cell_{index}" for index in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(n_genes)]),
    )
    adata.obsm["G"] = (
        adata.obs["condition"].astype("string").str.get_dummies().astype(int)
    )
    path = tmp_path / "screen.h5ad"
    adata.write_h5ad(path)
    return path


def _fit_args(source, output):
    return [
        "fit",
        str(source),
        "--control",
        "control",
        "--output",
        str(output),
        "--z-dim",
        "2",
        "--l-dim",
        "4",
        "--tau",
        "10",
        "--max-iter",
        "2",
    ]


def test_fit_requires_model_arguments():
    from perturbvi.cli import main

    with pytest.raises(SystemExit):
        main(["fit", "screen.h5ad", "--output", "out"])


def test_fit_help_exposes_the_keys_contract(capsys):
    from perturbvi.cli import main

    with pytest.raises(SystemExit) as error:
        main(["fit", "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--x-key" in help_text
    assert "--g-key" in help_text
    assert "--control" in help_text
    assert "--condition" not in help_text
    assert "--design-columns" not in help_text


def test_cli_fit_defaults_match_python_interfaces():
    from perturbvi.cli import _add_fit_args

    parser = argparse.ArgumentParser()
    _add_fit_args(parser)
    args = parser.parse_args(
        [
            "screen.h5ad",
            "--output",
            "results",
            "--z-dim",
            "2",
            "--l-dim",
            "4",
        ]
    )
    assert {
        "standardize": args.standardize,
        "init": args.init,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "verbose": args.verbose,
        "tau": args.tau,
        "x_key": args.x_key,
        "g_key": args.g_key,
        "control": args.control,
    } == {
        "standardize": DEFAULT_STANDARDIZE,
        "init": DEFAULT_INIT,
        "max_iter": DEFAULT_MAX_ITER,
        "tol": DEFAULT_TOL,
        "verbose": DEFAULT_VERBOSE,
        "tau": DEFAULT_TAU,
        "x_key": None,
        "g_key": "G",
        "control": None,
    }


@pytest.mark.parametrize("command", ["validate", "convert"])
def test_cli_has_no_redundant_validation_or_conversion_commands(command):
    from perturbvi.cli import main

    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


def test_fit_saves_one_canonical_fit_bundle(h5ad_path, tmp_path):
    from perturbvi import FitResults, load_results
    from perturbvi.cli import main

    output = tmp_path / "fit"
    main(_fit_args(h5ad_path, output))

    assert {path.name for path in output.iterdir()} == {
        "W.txt",
        "pip.txt",
        "pve.txt",
        "params_file.pkl",
        "run_config.json",
        "input_summary.json",
    }
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["z_dim"] == 2
    assert run_config["l_dim"] == 4
    assert run_config["tau"] == 10.0
    assert run_config["seed"] == 0
    input_summary = json.loads((output / "input_summary.json").read_text(encoding="utf-8"))
    assert input_summary["X_shape"] == [48, 12]
    assert input_summary["G_shape"] == [48, 2]
    loaded = load_results(output)
    assert isinstance(loaded, FitResults)
    assert loaded.gene_names == tuple(f"gene_{index}" for index in range(12))
    assert loaded.perturbation_names == ("IRF1", "STAT1")


def test_baseline_free_fit_needs_no_control_flag(h5ad_path, tmp_path):
    from perturbvi.cli import main

    adata = ad.read_h5ad(h5ad_path)
    G = pd.DataFrame(
        {
            "guide_A": adata.obs["guide_A"].astype(int),
            "guide_B": adata.obs["guide_B"].astype(int),
        },
        index=adata.obs_names,
    )
    adata.obsm["G"] = G
    source = tmp_path / "baseline_free.h5ad"
    adata.write_h5ad(source)

    output = tmp_path / "design_fit"
    args = _fit_args(source, output)
    del args[args.index("--control") : args.index("--control") + 2]
    main(args)
    assert (output / "params_file.pkl").is_file()


def test_g_key_flag_is_passed_through(h5ad_path, tmp_path):
    from perturbvi.cli import main

    adata = ad.read_h5ad(h5ad_path)
    adata.obsm["perturbations"] = adata.obsm["G"].copy()
    del adata.obsm["G"]
    source = tmp_path / "custom_gkey.h5ad"
    adata.write_h5ad(source)

    output = tmp_path / "gkey_fit"
    main(_fit_args(source, output) + ["--g-key", "perturbations"])
    assert (output / "params_file.pkl").is_file()


def test_covariate_strings_are_inferred_without_extra_type_flags(h5ad_path, tmp_path):
    from perturbvi.cli import main

    output = tmp_path / "covariate_fit"
    main(_fit_args(h5ad_path, output) + ["--covariates", "batch"])
    assert (output / "params_file.pkl").is_file()
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["covariates"] == ["batch"]
    assert run_config["categoricals"] == ["batch"]


def test_run_metadata_records_config_and_summary(tmp_path):
    from perturbvi import PerturbData
    from perturbvi.cli import _write_run_metadata

    data = PerturbData(
        X=np.zeros((4, 3)),
        G=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
        gene_names=["g1", "g2", "g3"],
        perturbation_names=["A", "B"],
        covariates=pd.DataFrame(
            {
                "batch": pd.Categorical(["x", "y", "x", "y"]),
                "depth": [1.0, 2.0, 3.0, 4.0],
            }
        ),
    )
    args = argparse.Namespace(
        input="screen.h5ad",
        z_dim=2,
        l_dim=4,
        tau=10.0,
        p_prior=0.5,
        standardize=False,
        init="pca",
        tol=1e-3,
        max_iter=3,
        seed=0,
        verbose=False,
        x_key=None,
        g_key="G",
        control="control",
        covariates=["batch", "depth"],
        device="cpu",
    )

    output = tmp_path / "meta"
    _write_run_metadata(output, data, args)

    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["covariates"] == ["batch", "depth"]
    assert run_config["categoricals"] == ["batch"]
    assert run_config["z_dim"] == 2
    assert run_config["seed"] == 0
    input_summary = json.loads((output / "input_summary.json").read_text(encoding="utf-8"))
    assert input_summary["X_shape"] == [4, 3]
    assert input_summary["G_shape"] == [4, 2]
    assert input_summary["perturbation_names"] == ["A", "B"]
    assert input_summary["residualized"] is True


def test_zarr_has_the_same_fit_behavior(h5ad_path, tmp_path):
    from perturbvi.cli import main

    source = tmp_path / "screen.zarr"
    ad.read_h5ad(h5ad_path).write_zarr(source)
    output = tmp_path / "zarr_fit"
    main(_fit_args(source, output))
    assert (output / "params_file.pkl").is_file()


def test_analyze_writes_only_core_tables_and_optional_lfsr(h5ad_path, tmp_path):
    from perturbvi.cli import main

    output = tmp_path / "fit"
    main(_fit_args(h5ad_path, output))
    main(["analyze", str(output)])

    assert {path.name for path in output.glob("*.csv")} == {
        "pip.csv",
        "pve.csv",
        "perturbation_effect.csv",
        "perturbation_pip.csv",
        "gene_effect.csv",
    }
    pip = pd.read_csv(output / "pip.csv", index_col=0)
    assert list(pip.index) == [f"gene_{index}" for index in range(12)]

    main(["analyze", str(output), "--compute-lfsr", "--lfsr-iters", "20"])
    assert (output / "lfsr.csv").is_file()
