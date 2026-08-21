# API

The public API covers loading a screen, optional residualization, model
fitting, saving, and analysis. Use the array interface when `X` and `G` are
already prepared and aligned.

## Load a screen

::: perturbvi.ScreenData

::: perturbvi.load_screen

## Residualize expression

::: perturbvi.residualize_screen

Residualization is optional. If residualization and standardization are both
requested, PerturbVI residualizes expression first and standardizes the
residuals. CLI fits record the applied steps in `run_config.json` and describe
the fitted input in `input_summary.json`.

## Fit the model

::: perturbvi.fit_screen

## Save and analyze results

::: perturbvi.save_results

::: perturbvi.load_results

::: perturbvi.analyze

## Saved files

`save_results` writes the model files below.

| File | Contents |
|---|---|
| `W.txt` | Factor loadings |
| `pip.txt` | Gene-factor posterior inclusion probabilities |
| `pve.txt` | Per-factor proportion of variance explained |
| `params_file.pkl` | Fitted model parameters |

When names are supplied, `gene_names.txt` and `perturbation_names.txt` are
written beside the model. CLI fits also write `run_config.json` and
`input_summary.json`.

`perturbvi analyze` writes these tables to `results/analysis` unless another
output directory is supplied:

| File | Contents |
|---|---|
| `pip_df.csv` | Gene-by-factor PIP table |
| `pve_df.csv` | Per-factor PVE table |
| `beta_df.csv` | Sparse perturbation-by-factor effects |
| `p_hat_df.csv` | Perturbation-factor posterior probabilities |
| `overall_effect_df.csv` | Gene-by-perturbation effects |
| `pip_significant_df.csv` | Genes meeting the PIP threshold |
| `pip_summary_df.csv` | Per-factor PIP and PVE summary |

With `--compute-lfsr`, analysis also writes `lfsr.csv` and
`lfsr_significant_df.csv`. LFSR is not computed by default.

## Array interface

The array interface is for callers that already have aligned dense or JAX
sparse matrices. The `infer(X, z_dim, l_dim, G, ...)` signature and
`InferResults` fields remain stable.

::: perturbvi.infer.compute_elbo

::: perturbvi.infer.infer

::: perturbvi.infer.compute_pip

::: perturbvi.infer.compute_pve

## Supporting calculations

::: perturbvi.utils.kl_discrete

::: perturbvi.utils.prob_pca

::: perturbvi.utils.bern_sample

::: perturbvi.utils.compute_lfsr
