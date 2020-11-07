.. ############################################################################
.. func_interpolation_BL_signals.rst
.. =================================
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


.. _fp_interp_def:

Fast Function Interpolation for Bandlimited Periodic Signals
============================================================

Theory
******

Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a :math:`T`-periodic function of bandwidth
:math:`N_{FS} = 2 N + 1`.  Then :math:`\phi` is fully characterized by its :math:`N_{FS}` Fourier
Series coefficients :math:`\{ \phi_{n}^{FS}, n = -N, \ldots, N \}` such that

.. math::
   :label: BL_FS_exp

   \phi(t) = \sum_{n = -N}^{N} \phi_{n}^{FS} \exp \left( j \frac{2 \pi}{T} n t \right).

Using :eq:`BL_FS_exp`, we can obtain :math:`M` values of :math:`\phi` at arbitrary positions
:math:`\{t_{k}, k = 0, \ldots, M-1 \}` with :math:`\mathcal{O}(N_{FS} M)` operations.

In the case of :math:`M` regularly-spaced samples however, an efficient algorithm based on the
*Chirp Z-Transform* can be used.


Complex-valued Functions
------------------------

.. admonition:: Theorem

   Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a :math:`T`-periodic function of bandwidth
   :math:`N_{FS} = 2 N + 1`.
   Let :math:`\{ \phi_{n}^{FS}, n = -N, \ldots, N \}` be its Fourier Series coefficients and
   :math:`a \lt b \in \mathbb{R}` be the end-points of an interval on which we want to evaluate
   :math:`M` equi-spaced samples of :math:`\phi`.

   Then the following holds:

   .. math::

      \Phi = A^{N} \text{CZT}_{N_{FS}}^{M} \{ \Phi^{FS} \} \circ W^{-N E},

   where :math:`\text{CZT}` of parameters :math:`A, W` is as defined in :ref:`CZT_def` and

   .. math::

      \Phi & = \left[ \phi(t[0]), \ldots, \phi(t[M - 1]) \right] \in \mathbb{C}^{M},

      \Phi^{FS} & = \left[ \phi_{-N}^{FS}, \ldots, \phi_{N}^{FS} \right] \in \mathbb{C}^{N_{FS}},

      E & = \left[ 0, \ldots, M - 1 \right] \in \mathbb{N}^{M},

      t[k] & = \left( a + \frac{b - a}{M - 1} k \right) 1_{[0, \ldots, M-1]}[k],

      A & = \exp \left( -j \frac{2 \pi}{T} a \right),

      W & = \exp \left( j \frac{2 \pi}{T} \frac{b - a}{M - 1}\right).


.. admonition:: Proof

   Replacing :math:`t[k]` in :eq:`BL_FS_exp`, we get

   .. math::

      \phi(t[k]) & = \sum_{n = -N}^{N} \phi_{n}^{FS} A^{-n} W^{n k}

      & = A^{N} W^{-N k} \sum_{n = 0}^{2 N} \phi_{n - N}^{FS} A^{-n} W^{n k}

      & = A^{N} W^{-N k} \text{CZT}_{N_{FS}}^{M} \{ \Phi^{FS} \}[k],

   with :math:`A, W, \Phi^{FS}` as defined above.

   Rearranging the equation into vector form proves the theorem.


Real-valued Functions
---------------------

.. admonition:: Theorem

   Let :math:`\phi: \mathbb{R} \to \mathbb{R}` be a :math:`T`-periodic function of bandwidth
   :math:`N_{FS} = 2 N + 1`.
   Let :math:`\{ \phi_{n}^{FS}, n = -N, \ldots, N \}` be its Fourier Series coefficients and
   :math:`a \lt b \in \mathbb{R}` be the end-points of an interval on which we want to evaluate
   :math:`M` equi-spaced samples of :math:`\phi`.

   Then the following holds:

   .. math::

      \Phi = \phi_{0}^{FS} + 2 \Re \left[ A^{-1} \text{CZT}_{N}^{M} \{ \Phi_{+}^{FS} \} \circ W^{E} \right],

   where :math:`\text{CZT}` of parameters :math:`A, W` is as defined in :ref:`CZT_def` and

   .. math::

      \Phi & = \left[ \phi(t[0]), \ldots, \phi(t[M - 1]) \right] \in \mathbb{C}^{M},

      \Phi_{+}^{FS} & = \left[ \phi_{1}^{FS}, \ldots, \phi_{N}^{FS} \right] \in \mathbb{C}^{N},

      E & = \left[ 0, \ldots, M - 1 \right] \in \mathbb{N}^{M},

      t[k] & = \left( a + \frac{b - a}{M - 1} k \right) 1_{[0, \ldots, M-1]}[k],

      A & = \exp \left( -j \frac{2 \pi}{T} a \right),

      W & = \exp \left( j \frac{2 \pi}{T} \frac{b - a}{M - 1}\right).


.. admonition:: Proof

   Leverage the conjugate symmetry of the :math:`\phi_{k}^{FS}` in the previous proof.


Multi-dimensional Functions
---------------------------

For a multi-dimensional signal :math:`\phi: \mathbb{R}^D \to \mathbb{C}` that is
:math:`[T_1, T_2, \ldots, T_D]`-periodic and :math:`[N_{FS, 1}, N_{FS, 2}, \ldots,
N_{FS, D}]`-bandlimited, we can also obtain arbitrary values of :math:`\phi` on a uniform grid in a similarly
efficient manner. To do so, we need to perform a multi-dimensional :math:`\text{CZT}` and then
modulate along each dimension as done in the complex-valued theorem above.


Implementation Notes
********************

:py:func:`~pyffs.interp.fs_interp` can be used to obtain samples of a function using the algorithms
above.

:py:func:`~pyffs.interp.fs_interpn` can be used to obtain samples of a :math:`D`-dimensional
function.
