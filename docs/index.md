---
hide:
  - toc
---

# PerturbVI

PerturbVI infers latent gene programs and their perturbation effects from
single-cell Perturb-seq data.

!!! note
    For the preprint, please see: <br/>
    *PerturbVI: A Scalable Latent Factor Model to Infer Genetic Regulatory Modules through CRISPR Perturbation Data*. <br/>
    [doi.org/10.0000/perturbvi](https://doi.org/10.0000/perturbvi) (placeholder DOI)

!!! important
    To reproduce the analyses in the preprint: <br/>
    [zenodo.org/records/0000000](https://zenodo.org/records/0000000) (placeholder)

## Installation

```bash
uv pip install perturbvi
```

## Quick start

Prepare an H5AD file with transformed expression in `adata.X` and a binary,
named perturbation DataFrame in `adata.obsm["G"]`.

```python
from pathlib import Path
from perturbvi import fit_screen, load_screen, save_results

result_dir = Path("results/my_screen")
data = load_screen(
    "data/screen.h5ad",
)
fit = fit_screen(
    data,
    z_dim=12,
    l_dim=100,
)
save_results(
    fit,
    result_dir,
)
```

This saves the fitted model and labeled result CSVs in `result_dir`.
See the tutorials for plotting and enrichment.

## Tutorials

- [LUHMES Analysis with PerturbVI](luhmes.md): fitting, factor and gene effects, and neuronal GO enrichment.
- Replogle Analysis with PerturbVI: fitting and interpretation (coming soon).
- [Using PerturbVI with Your Data](workflow.md): CSV and AnnData inputs, controls, covariates, and fitting.
- [API](api.md): function arguments, result matrices, and CLI.

## Support

Please report bugs or feature requests in the
[issue tracker](https://github.com/mancusolab/perturbvi/issues). For questions
or comments, contact Abdullah Al Nahid (<alnahid@usc.edu>) or Nicholas Mancuso
(<nmancuso@usc.edu>).

## Other Software

Other software developed by the [Mancuso Lab](https://www.mancusolab.com/):

- [SuShiE](https://github.com/mancusolab/sushie): a Bayesian fine-mapping
  framework for molecular QTL data across multiple ancestries.
- [jaxQTL](https://github.com/mancusolab/jaxqtl): scalable, count-based
  large-scale eQTL mapping.
- [MA-FOCUS](https://github.com/mancusolab/ma-focus): a Bayesian fine-mapping
  framework using [TWAS](https://www.nature.com/articles/ng.3506) statistics
  across multiple ancestries to identify causal genes for complex traits.
- [SuSiE-PCA](https://github.com/mancusolab/susiepca): scalable Bayesian
  variable selection for sparse principal component analysis.
- [twas_sim](https://github.com/mancusolab/twas_sim): simulation of
  [TWAS](https://www.nature.com/articles/ng.3506) statistics.
- [traceax](https://github.com/mancusolab/traceax): stochastic trace
  estimation for linear operators.
- [FactorGo](https://github.com/mancusolab/factorgo): scalable variational
  factor analysis for learning pleiotropic factors from GWAS summary
  statistics.
- [HAMSTA](https://github.com/tszfungc/hamsta): estimation of heritability
  explained by local ancestry data from admixture mapping summary statistics.

---

PerturbVI is distributed under the terms of the
[MIT license](https://spdx.org/licenses/MIT.html).
