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


def test_fit_saves_one_canonical_fit_bundle(h5ad_path, tmp_path):
    from perturbvi.cli import main

    output = tmp_path / "fit"
    main(_fit_args(h5ad_path, output))

    assert {path.name for path in output.iterdir()} == {
        "W.csv",
        "PIP_W.csv",
        "B.csv",
        "PIP_B.csv",
        "BW.csv",
        "PVE.csv",
        "model.pkl",
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
    BW = pd.read_csv(output / "BW.csv", index_col=0)
    assert BW.index.tolist() == ["IRF1", "STAT1"]
    assert BW.columns.tolist() == [f"gene_{index}" for index in range(12)]


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
    assert (output / "model.pkl").is_file()


def test_g_key_flag_is_passed_through(h5ad_path, tmp_path):
    from perturbvi.cli import main

    adata = ad.read_h5ad(h5ad_path)
    adata.obsm["perturbations"] = adata.obsm["G"].copy()
    del adata.obsm["G"]
    source = tmp_path / "custom_gkey.h5ad"
    adata.write_h5ad(source)

    output = tmp_path / "gkey_fit"
    main(_fit_args(source, output) + ["--g-key", "perturbations"])
    assert (output / "model.pkl").is_file()


def test_covariate_strings_are_inferred_without_extra_type_flags(h5ad_path, tmp_path):
    from perturbvi.cli import main

    output = tmp_path / "covariate_fit"
    main(_fit_args(h5ad_path, output) + ["--covariates", "batch"])
    assert (output / "model.pkl").is_file()
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["covariates"] == ["batch"]
    assert run_config["categoricals"] == ["batch"]


def test_zarr_has_the_same_fit_behavior(h5ad_path, tmp_path):
    from perturbvi.cli import main

    source = tmp_path / "screen.zarr"
    ad.read_h5ad(h5ad_path).write_zarr(source)
    output = tmp_path / "zarr_fit"
    main(_fit_args(source, output))
    assert (output / "model.pkl").is_file()
