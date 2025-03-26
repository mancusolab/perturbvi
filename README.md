[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI-Server](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![Github](https://img.shields.io/github/stars/mancusolab/perturbvi?style=social)](https://github.com/mancusolab/perturbvi)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

# perturbVI
`perturbvi` is a scalable approach to infer regulatory modules through informative latent component model in the single-cell Perturb-seq data.

  [**Installation**](#installation)
  | [**Example**](#get-started-with-example)
  | [**Notes**](#notes)
  | [**Version**](#version-history)
  | [**Support**](#support)
  | [**Other Software**](#other-software)

------------------

## Installation

Users can download the latest repository and then use `pip`:

``` bash
git clone https://github.com/mancusolab/perturbvi.git
cd perturbvi
pip install .
```

## Get Started with `perturbvi`

### 1. `infer`
Perform inference using SuSiE PCA to find the regulatory modules from CRISPR perturbation data
``` bash
perturbvi infer <exp_csv> <guide_csv> <gene_symbol_csv> -o=output --verbose
```

#### Arguments
- `exp_csv`: Path to the experiment CSV file.
- `guide_csv`: Path to the guide CSV file.
- `gene_symbol_csv`: Path to the gene symbol CSV file.
- `-o=output`: Specifies the output directory name or path.
- `--verbose`: For logging (Optional).

#### Example Usage
```bash
perturbvi infer data/exp.csv data/guide.csv data/symbol.csv -o=data/out --verbose
```

This will save the all the output files (including the parameter file `params.pkl`) into the `data/out` folder, which can be used for the downstream tasks outlined below.

## Notes

-   `perturbvi` uses [JAX](https://github.com/google/jax) with [Just In
    Time](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    compilation to achieve high-speed computation. However, there are
    some [issues](https://github.com/google/jax/issues/5501) for JAX
    with Mac M1 chip. To solve this, users need to initiate conda using
    [miniforge](https://github.com/conda-forge/miniforge), and then
    install `perturbvi` using `pip` in the desired environment.

## Version History

TBD

## Support

Please report any bugs or feature requests in the [Issue
Tracker](https://github.com/mancusolab/perturbvi/issues). If users have
any questions or comments, please contact Dong Yuan (<dongyuan@usc.edu>)
and Nicholas Mancuso (<nmancuso@usc.edu>).

## Other Software

Feel free to use other software developed by [Mancuso
Lab](https://www.mancusolab.com/):

-   [SuShiE](https://github.com/mancusolab/sushie): a Bayesian
    fine-mapping framework for molecular QTL data across multiple
    ancestries.
-   [MA-FOCUS](https://github.com/mancusolab/ma-focus): a Bayesian
    fine-mapping framework using
    [TWAS](https://www.nature.com/articles/ng.3506) statistics across
    multiple ancestries to identify the causal genes for complex traits.
-   [SuSiE-PCA](https://github.com/mancusolab/susiepca): a scalable
    Bayesian variable selection technique for sparse principal component
    analysis
-   [twas_sim](https://github.com/mancusolab/twas_sim): a Python
    software to simulate [TWAS](https://www.nature.com/articles/ng.3506)
    statistics.
-   [FactorGo](https://github.com/mancusolab/factorgo): a scalable
    variational factor analysis model that learns pleiotropic factors
    from GWAS summary statistics.
-   [HAMSTA](https://github.com/tszfungc/hamsta): a Python software to
    estimate heritability explained by local ancestry data from
    admixture mapping summary statistics.

------------------------------------------------------------------------

`perturbvi` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.


------------------------------------------------------------------------

This project has been set up using Hatch. For details and usage
information on Hatch see <https://github.com/pypa/hatch>.
