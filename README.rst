.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

.. image:: https://img.shields.io/badge/Docs-Available-brightgreen
    :alt: Documentation-webpage
    :target: https://mancusolab.github.io/perturbvi/

.. image:: https://img.shields.io/pypi/v/perturbvi.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/perturbvi/

.. image:: https://img.shields.io/github/stars/mancusolab/perturbvi?style=social
    :alt: Github
    :target: https://github.com/mancusolab/perturbvi

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :alt: License
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg
    :alt: Project generated with Hatch
    :target: https://github.com/pypa/hatch

======
perturbVI
======
``perturbvi`` is a Python library to compute p-values of scores computed under exponential family models
using saddlepoint approximation of the sampling distribution.


|Installation|_ | |Example|_ | |Notes|_ | |Version|_ | |Support|_ | |Other Software|_

=================

.. _Installation:
.. |Installation| replace:: **Installation**

Installation
============
Users can download the latest repository and then use ``pip``:

.. code:: bash

    git clone https://github.com/mancusolab/perturbvi.git
    cd perturbvi
    pip install .

.. _Example:
.. |Example| replace:: **Example**

Get Started with Example
========================
TBD

.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====
* ``perturbvi`` uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compilation to achieve high-speed computation. However, there are some `issues <https://github.com/google/jax/issues/5501>`_ for JAX with Mac M1 chip. To solve this, users need to initiate conda using `miniforge <https://github.com/conda-forge/miniforge>`_, and then install ``perturbvi`` using ``pip`` in the desired environment.

.. _Version:
.. |Version| replace:: **Version**

Version History
===============
TBD

.. _Support:
.. |Support| replace:: **Support**


Support
=======

Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/perturbvi/issues>`_.
If users have any questions or comments, please contact Eric Zhao (yzhao532@usc.edu) and Nicholas Mancuso (nmancuso@usc.edu).

.. _OtherSoftware:
.. |Other Software| replace:: **Other Software**

Other Software
==============

Feel free to use other software developed by `Mancuso Lab <https://www.mancusolab.com/>`_:

* `SuShiE <https://github.com/mancusolab/sushie>`_: a Bayesian fine-mapping framework for molecular QTL data across multiple ancestries.

* `MA-FOCUS <https://github.com/mancusolab/ma-focus>`_: a Bayesian fine-mapping framework using `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics across multiple ancestries to identify the causal genes for complex traits.

* `SuSiE-PCA <https://github.com/mancusolab/susiepca>`_: a scalable Bayesian variable selection technique for sparse principal component analysis

* `twas_sim <https://github.com/mancusolab/twas_sim>`_: a Python software to simulate `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics.

* `FactorGo <https://github.com/mancusolab/factorgo>`_: a scalable variational factor analysis model that learns pleiotropic factors from GWAS summary statistics.

* `HAMSTA <https://github.com/tszfungc/hamsta>`_: a Python software to  estimate heritability explained by local ancestry data from admixture mapping summary statistics.

---------------------

.. _license:

``perturbvi`` is distributed under the terms of the `MIT <https://spdx.org/licenses/MIT.html>`_ license.


---------------------

.. _hatch-notes:

This project has been set up using Hatch. For details and usage
information on Hatch see https://github.com/pypa/hatch.
