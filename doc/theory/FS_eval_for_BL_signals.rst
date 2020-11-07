.. ############################################################################
.. FS_eval_for_BL_signals.rst
.. ==========================
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


.. _FFS_def:

Fast Fourier Series Evaluation for Bandlimited Periodic Signals
===============================================================

Theory
******

Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a :math:`T`-periodic function of bandwidth
:math:`N_{FS} = 2 N + 1`.  Then :math:`\phi` is fully characterized by its :math:`N_{FS}` Fourier
Series coefficients :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}` such that

.. math::
   :label: BL_FS_expansion

   \phi(t) = \sum_{k = -N}^{N} \phi_{k}^{FS} \exp\left( j \frac{2 \pi}{T} k t \right),

where :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}` is defined as

.. math::
   :label: FS_def

   \phi_{k}^{FS} = \frac{1}{T} \int_{T_{c} - \frac{T}{2}}^{T_{c} + \frac{T}{2}} \phi(t) \exp\left( -j \frac{2 \pi}{T} k t \right) dt,

with :math:`T_{c}` being one of :math:`\phi`'s period mid-points.

Computing the :math:`\phi_{k}^{FS}` with :eq:`FS_def` can be prohibitive when closed-form solutions
are unavailable.  However, it is possible to calculate the FS coefficients exactly from :math:`N_{s}
= N_{FS} + Q` judiciously-placed samples of :math:`\phi`, where :math:`Q \in \mathbb{N}` can be
arbitrarily chosen.


:math:`N_{s} \in 2 \mathbb{N} + 1`
----------------------------------

.. admonition:: Theorem

   Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a :math:`T`-periodic function of bandwidth
   :math:`N_{FS} = 2 N + 1`, with

   * :math:`T_{c}`: mid-point of one period,
   * :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}`: Fourier Series coefficients of :math:`\phi`.

   Moreover, let :math:`Q \in 2 \mathbb{N}` be an arbitrary even integer such that

   * :math:`N_{s} = N_{FS} + Q`,
   * :math:`M = (N_{s} - 1) / 2`.

   Then the following holds:

   .. math::

      \Phi & = N_{s} \text{iDFT}_{N_{s}}\left\{ \Phi^{FS} \circ B_{1}^{E_{1}} \right\} \circ B_{2}^{N E_{2}},

      \Phi^{FS} & = \frac{1}{N_{s}} \text{DFT}_{N_{s}}\left\{ \Phi \circ B_{2}^{- N E_{2}} \right\} \circ B_{1}^{- E_{1}},

   where :math:`\text{(i)DFT}` is as defined in :ref:`DFT_def` and

   .. math::

      \Phi      & = \left[ \phi(t[0]), \ldots, \phi(t[M]), \phi(t[-M]), \ldots, \phi(t[-1]) \right] \in \mathbb{C}^{N_{s}},

      \Phi^{FS} & = \left[ \phi_{-N}^{FS}, \ldots, \phi_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}},

      E_{1}     & = \left[ -N, \ldots, N, 0, \ldots, 0 \right] \in \mathbb{Z}^{N_{s}},

      E_{2}     & = \left[ 0, \ldots, M, -M, \ldots, -1 \right] \in \mathbb{Z}^{N_{s}},

      t[n]      & = \left( T_{c} + \frac{T}{N_{s}} n \right) 1_{[-M, \ldots, M]}[n],

      B_{1}     & = \exp\left( j \frac{2 \pi}{T} T_{c} \right),

      B_{2}     & = \exp\left( -j \frac{2 \pi}{N_{s}} \right).


:math:`N_{s} \in 2 \mathbb{N}`
------------------------------

.. admonition:: Theorem

   Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a :math:`T`-periodic function of bandwidth
   :math:`N_{FS} = 2 N + 1`, with

   * :math:`T_{c}`: mid-point of one period,
   * :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}`: Fourier Series coefficients of :math:`\phi`.

   Moreover, let :math:`Q \in 2 \mathbb{N} + 1` be an arbitrary odd integer such that

   * :math:`N_{s} = N_{FS} + Q`,
   * :math:`M = N_{s} / 2`.

   Then the following holds:

   .. math::

      \Phi & = N_{s} \; \text{iDFT}_{N_{s}}\left\{ \Phi^{FS} \circ B_{1}^{E_{1}} \right\} \circ B_{2}^{N E_{2}},

      \Phi^{FS} & = \frac{1}{N_{s}} \text{DFT}_{N_{s}}\left\{ \Phi \circ B_{2}^{- N E_{2}} \right\} \circ B_{1}^{- E_{1}},

   where :math:`\text{(i)DFT}` is as defined in :ref:`DFT_def` and

   .. math::

      \Phi & = \left[ \phi(t[0]), \ldots, \phi(t[M - 1]), \phi(t[-M]), \ldots, \phi(t[-1]) \right] \in \mathbb{C}^{N_{s}},

      \Phi^{FS} & = \left[ \phi_{-N}^{FS}, \ldots, \phi_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}},

      E_{1} & = \left[ -N, \ldots, N, 0, \ldots, 0 \right] \in \mathbb{Z}^{N_{s}},

      E_{2} & = \left[ 0, \ldots, M - 1, -M, \ldots, -1 \right] \in \mathbb{Z}^{N_{s}},

      t[n] & = \left( T_{c} + \frac{T}{N_{s}} \left[ \frac{1}{2} + n \right] \right) 1_{[-M, \ldots, M - 1]}[n],

      B_{1} & = \exp\left( j \frac{2 \pi}{T} \left[ T_{c} + \frac{T}{2 N_{s}} \right] \right),

      B_{2} & = \exp\left( -j \frac{2 \pi}{N_{s}} \right).


.. admonition:: Proof(s)

   Replace :math:`t[n]` in :eq:`BL_FS_expansion` and rearrange terms.


Extension to multi-dimensional case
-----------------------------------

For a multi-dimensional signal :math:`\phi: \mathbb{R}^D \to \mathbb{C}` that is
:math:`[T_1, T_2, \ldots, T_D]`-periodic and :math:`[N_{FS, 1}, N_{FS, 2}, \ldots,
N_{FS, D}]`-bandlimited, we can obtain its Fourier Series coefficients by applying the above approach along each
dimension.


Implementation Notes
********************

:py:func:`~pyffs.ffs.ffs` and :py:func:`~pyffs.ffs.iffs` can be used to obtain Fourier Series
coefficients / spatial samples of a function using the algorithms above.  Due to the reliance on
:math:`\text{(i)DFT}_{N_{s}}`, it is recommended to choose :math:`N_{s}` highly-composite.

:py:func:`~pyffs.ffs.ffsn` and :py:func:`~pyffs.ffs.iffsn` can be used to obtain Fourier Series
coefficients / spatial samples of a :math:`D`-dimensional function. In our implementation, we opt
for a more efficient approach than applying the above method along each dimension.
