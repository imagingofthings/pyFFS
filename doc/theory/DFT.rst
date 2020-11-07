.. ############################################################################
.. DFT.rst
.. =======
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


.. _DFT_def:

The Discrete Fourier Transform
==============================

Let :math:`X \in \mathbb{C}^{N}`.

The length-:math:`N` *Discrete Fourier Transform* :math:`\text{DFT}_{N}\{X\} \in \mathbb{C}^{N}` is
defined as:

.. math::

   \text{DFT}_{N}\{X\}[k] = \sum_{n = 0}^{N - 1} X[n] W_{N}^{n k}, \qquad  k \in \{ 0, \ldots, N - 1 \},

with :math:`W_{N} = \exp\left( -j \frac{2 \pi}{N} \right)`.

The length-:math:`N` *inverse Discrete Fourier Transform* :math:`\text{iDFT}_{N}\{X\} \in
\mathbb{C}^{N}` is defined as:

.. math::

   \text{iDFT}_{N}\{X\}[n] = \frac{1}{N} \sum_{k = 0}^{N - 1} X[k] W_{N}^{-n k}, \qquad n \in \{ 0, \ldots, N - 1 \}.

Moreover, :math:`\text{(i)DFT}_{N}` is reversible:

.. math::

   (\text{iDFT}_{N} \circ \text{DFT}_{N})\{X\} = (\text{DFT}_{N} \circ \text{iDFT}_{N})\{X\} = X.

Particularly efficient :math:`\mathcal{O}(N \log N)` algorithms exist to compute
:math:`\text{(i)DFT}_{N}`, especially if :math:`N` is highly composite.
