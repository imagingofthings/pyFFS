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

For a :math:`D`-dimensional Chirp Z-Transform, namely a :math:`(M_1, M_2, \ldots, M_D)` length
transform of :math:`X \in \mathbb{C}^{N_1 \times N_2 \times \cdots \times N_D}`, the above operation can be
performed one dimension at a time.


Implementation Notes
********************

:py:func:`~pyffs.czt.czt` can be used to compute :math:`\text{CZT}_{N}^{M}` as
defined above, with :math:`L \ge N + M - 1` optimally chosen.

:py:func:`~pyffs.czt.cztn` can be used to compute :math:`\text{CZT}_{N_1, N_2,
\ldots, N_D}^{M_1, M_2, \ldots, M_D}`, with :math:`L_d \ge N_d + M_d - 1` optimally chosen for
:math:`d = 1, 2, \ldots, D`. In our implementation, we opt for a more efficient approach than
applying the 1D :math:`\text{CZT}` along each dimension.
