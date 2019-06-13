.. ############################################################################
.. CZT.rst
.. =======
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


.. _CZT_def:

The Chirp Z-Transform
=====================

Let :math:`X \in \mathbb{C}^{N}`.

The length-:math:`M` `Chirp Z-Transform <https://ieeexplore.ieee.org/document/1162034/>`_
:math:`\text{CZT}_{N}^{M}\{X\} \in \mathbb{C}^{M}` of parameters :math:`A, W \in \mathbb{C}^{*}` is
defined as:

.. math::

   \text{CZT}_{N}^{M}\{ X \}[k] = \sum_{n = 0}^{N - 1} X[n] A^{-n} W^{n k}, \qquad k \in \{ 0, \ldots, M - 1 \}.


:math:`\text{CZT}_{N}^{M}` can be efficiently computed using :math:`\text{(i)DFT}` in
:math:`\mathcal{O}(L \log L)` operations, where :math:`L \ge N + M - 1` can be arbitrarily chosen.


Implementation Notes
********************

:py:func:`~pyffs.czt` can be used to compute :math:`\text{CZT}_{N}^{M}` as
defined above, with :math:`L \ge N + M - 1` optimally chosen.
