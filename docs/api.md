# API

The public API covers loading a screen, optional residualization, model
fitting, saving, and analysis. Use the array interface when `X` and `G` are
already constructed with matching rows.

## Load a screen

::: perturbvi.ScreenData

::: perturbvi.load_screen

The CLI uses one assignment option for each input layout:

| Input layout | Perturbation assignments |
|---|---|
| H5AD labels in `obs` | `--perturbation-column` and `--control-label` |
| H5AD matrix in `obsm` | `--guide-matrix-key` |
| 10x H5 or MEX | `--cell-metadata`, `--perturbation-column`, and `--control-label` |
| CSV or TSV expression | `--guide-matrix` |

`--expression-layer` selects `X` or a layer from H5AD. `--header` and
`--index-col` apply only to CSV and TSV files. Their defaults expect gene names
in the first row and cell IDs in the first column; use `none` only when either
is absent. Run `perturbvi fit --help` for complete input recipes.

## Residualize expression

::: perturbvi.residualize_screen

Residualization is optional. PerturbVI centers every expression feature before
fitting. With `standardize=True`, it also scales each centered feature to unit
population variance. Residualization runs before centering and scaling.
`perturbvi fit` records these steps in `run_config.json` and describes the
fitted input in `input_summary.json`.

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

`perturbvi fit` writes `run_config.json` and `input_summary.json` beside the
model. `input_summary.json` retains gene and perturbation names for later CLI
analysis; separate name text files are not created.

Gene names label columns of `X`, and perturbation names label columns of `G`.
Names passed to `analyze()` or supplied to `perturbvi analyze` label the output
tables; they must have the correct length and order and do not alter fitted
values. CLI override files contain one name per row.

`perturbvi analyze` writes these tables to the fit directory:

| File | Contents |
|---|---|
| `pip.csv` | Gene-by-factor PIP table |
| `pve.csv` | Per-factor PVE table |
| `beta.csv` | Sparse perturbation-by-factor effects |
| `p_hat.csv` | Perturbation-factor posterior probabilities |
| `overall_effect.csv` | Gene-by-perturbation effects |
| `pip_significant.csv` | Genes with PIP at or above the cutoff |
| `pip_summary.csv` | Per-factor PIP and PVE summary |

If `lfsr.csv` already exists, `perturbvi analyze` reads it without recomputing
LFSR after confirming that its gene and perturbation labels match the fit.
`--compute-lfsr` recomputes LFSR and replaces `lfsr.csv` and
`lfsr_significant.csv`. LFSR values at or below the cutoff are significant.

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
