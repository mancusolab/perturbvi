---
hide:
  - toc
---

# Single-cell perturbation analysis

PerturbVI learns latent gene programs and perturbation effects from
single-cell perturbation screens.

## Install

```bash
uv pip install perturbvi
```

## Quick start

Fit a screen:

```bash
perturbvi fit screen.h5ad \
  --perturbation-column perturbation \
  --control-label control \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

Analyze the saved fit:

```bash
perturbvi analyze results
```

## Cookbook

- [Shared Python workflow](cookbook.md#shared-function)
- [Preparing raw expression](cookbook.md#preparing-raw-expression)
- [LUHMES CSV fit and analysis](cookbook.md#luhmes-csv-fit-and-analysis)
- [Adamson H5AD and CSV](cookbook.md#adamson-et-al-2016-perturb-seq)
- [Datlinger CROP-seq](cookbook.md#datlinger-et-al-2017-crop-seq)
- [Norman CRISPRa with combination perturbations](cookbook.md#norman-et-al-2019-crispra)
- [Srivatsan sci-Plex](cookbook.md#srivatsan-et-al-2020-sci-plex-2)
- [10x A375 H5](cookbook.md#10x-genomics-a375-crispr)
- [10x A549 MEX](cookbook.md#10x-genomics-a549-crispr)

## Reference

See the [API reference](api.md) for functions, command options, and
[saved files](api.md#saved-files).

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

---

This project has been set up using Hatch. See the
[Hatch documentation](https://hatch.pypa.io/) for usage information.
