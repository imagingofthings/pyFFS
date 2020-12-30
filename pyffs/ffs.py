# #############################################################################
# ffs.py
# ======
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric Bezzam [ebezzam@gmail.com]
# #############################################################################

"""
Methods for computing Fast Fourier Series.
"""

__all__ = ["ffs", "ffsn", "iffs", "iffsn", "_ffsn", "_iffsn"]

import numpy as np
from scipy import fftpack as fftpack

from pyffs.util import _create_modulation_vectors, _verify_ffsn_input


def ffs(x, T, T_c, N_FS, axis=-1):
    r"""
    Fourier Series coefficients from signal samples of a 1D function.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        (..., N_s, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffs_sample`.
    T : float
        Function period.
    T_c : float
        Period mid-point.
    N_FS : int
        Function bandwidth.
    axis : int
        Dimension of `x` along which function samples are stored.

    Returns
    -------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_s, ...) vectors containing entries :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS},
        0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}}`.

    Examples
    --------
    Let :math:`\phi(t)` be a shifted Dirichlet kernel of period :math:`T` and bandwidth
    :math:`N_{FS} = 2 N + 1`:

    .. math::

       \phi(t) = \sum_{k = -N}^{N} \exp\left( j \frac{2 \pi}{T} k (t - T_{c}) \right)
               = \frac{\sin\left( N_{FS} \pi [t - T_{c}] / T \right)}{\sin\left( \pi [t - T_{c}]
               / T \right)}.

    Its Fourier Series (FS) coefficients :math:`\phi_{k}^{FS}` can be analytically evaluated using
    the shift-modulation theorem:

    .. math::

       \phi_{k}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pyffs.ffs.ffs` to numerically evaluate
    :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}`:

    .. testsetup::

       import math

       import numpy as np

       from pyffs import ffs_sample, ffs
       from pyffs.func import dirichlet, dirichlet_fs

    .. doctest::

       >>> T, T_c, N_FS = math.pi, math.e, 15
       >>> N_samples = 16  # Any >= N_FS will do, but highly-composite best.

       # Sample the kernel and do the transform.
       >>> sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
       >>> diric_samples = dirichlet(sample_points, T, T_c, N_FS)
       >>> diric_FS = ffs(diric_samples, T, T_c, N_FS)

       # Compare with theoretical result.
       >>> np.allclose(diric_FS[:N_FS], dirichlet_fs(N_FS, T, T_c))
       True

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffs_sample`, :py:func:`~pyffs.ffs.iffs`
    """
    return ffsn(x=x, T=[T], T_c=[T_c], N_FS=[N_FS], axes=(axis,))


def iffs(x_FS, T, T_c, N_FS, axis=-1):
    r"""
    Signal samples from Fourier Series coefficients of a 1D function.

    :py:func:`~pyffs.ffs.iffs` is basically the inverse of :py:func:`~pyffs.ffs.ffs`.

    Parameters
    ----------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_s, ...) FS coefficients in the order :math:`\left[ x_{-N}^{FS}, \ldots,
        x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}}`.
    T : float
        Function period.
    T_c : float
        Period mid-point.
    N_FS : int
        Function bandwidth.
    axis : int
        Dimension of `x_FS` along which FS coefficients are stored.

    Returns
    -------
    x : :py:class:`~numpy.ndarray`
        (..., N_s, ...) vectors containing original function samples given to
        :py:func:`~pyffs.ffs.ffs`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ x \} = x`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffs_sample`, :py:func:`~pyffs.ffs.ffs`
    """
    return iffsn(x_FS=x_FS, T=[T], T_c=[T_c], N_FS=[N_FS], axes=(axis,))


def ffsn(x, T, T_c, N_FS, axes=None):
    r"""
    Fourier Series coefficients from signal samples of a D-dimension signal.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffsn_sample`.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Period mid-point for each dimension.
    N_FS : list(int)
        Function bandwidth along each dimension.
    axes : tuple
        Dimensions of `x` along which function samples are stored.

    Returns
    -------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) array containing Fourier Series
        coefficients in ascending order (top-left of matrix).

    Examples
    --------
    Let :math:`\phi(x, y)` be a shifted Dirichlet kernel of periods :math:`(T_x,
    T_y)` and bandwidths :math:`N_{FS, x} = 2 N_x + 1, N_{FS, y} = 2 N_y + 1`:

    .. math::

       \phi(x, y) &= \sum_{k_x = -N_x}^{N_x} \sum_{k_y = -N_y}^{N_y}
                \exp\left( j \frac{2 \pi}{T_x} k_x (x - T_{c,x}) \right)
                \exp\left( j \frac{2 \pi}{T_y} k_y (y - T_{c,y}) \right) \\
               &= \frac{\sin\left( N_{FS, x} \pi [x - T_{c,x}] / T_x \right)}{\sin\left( \pi
               [x - T_{c, x}] / T_x \right)} \frac{\sin\left( N_{FS, y} \pi [y - T_{c,y}] / T_y
               \right)}{\sin\left( \pi [y - T_{c, y}] / T_y \right)}.

    Its Fourier Series (FS) coefficients :math:`\phi_{k_x, k_y}^{FS}` can be
    analytically evaluated using the shift-modulation theorem:

    .. math::

       \phi_{k_x, k_y}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T_x} k_x T_{c,x} \right) \exp\left( -j \frac{2 \pi}{T_y} k_y
           T_{c,y} \right) & -N_x \le k_x \le N_x, -N_y \le k_y \le N_y,  \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pyffs.ffs.ffsn` to numerically
    evaluate :math:`\{\phi_{k_x, k_y}^{FS}, k_x = -N_x, \ldots, N_x, k_y = -N_y,
    \ldots, N_y\}`:

    .. testsetup::

       import math

       import numpy as np

       from pyffs import ffsn_sample, ffsn
       from pyffs.func import dirichlet_2D, dirichlet_fs

    .. doctest::

       >>> T = [1, 1]
       >>> T_c = [0, 0]
       >>> N_FS = [3, 3]
       >>> N_s = [4, 3]

       # Sample the kernel and do the transform.
       >>> sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
       >>> diric_samples = dirichlet_2D(sample_points, T, T_c, N_FS)
       >>> diric_FS = ffsn(x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)

       # Compare with theoretical result.
       >>> diric_FS_exact = np.outer(
       ... dirichlet_fs(N_FS[0], T[0], T_c[0]), dirichlet_fs(N_FS[1], T[1], T_c[1])
       ... )
       >>> np.allclose(diric_FS[: N_FS[0], : N_FS[1]], diric_FS_exact)
       True

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffsn_sample`, :py:func:`~pyffs.ffs.iffsn`
    """
    axes, N_s = _verify_ffsn_input(x, T, T_c, N_FS, axes)

    # check for input type
    if (x.dtype == np.dtype("complex64")) or (x.dtype == np.dtype("float32")):
        is_complex64 = True
        x_FS = x.copy().astype(np.complex64)
    else:
        is_complex64 = False
        x_FS = x.copy().astype(np.complex128)

    # apply pre-FFT modulation
    A = []
    for d, N_sd in enumerate(N_s):
        A_d, B_d = _create_modulation_vectors(N_sd, N_FS[d], T[d], T_c[d])
        A.append(A_d.conj())
        sh = [1] * x.ndim
        sh[axes[d]] = N_s[d]
        C_2 = B_d.conj().reshape(sh)
        if is_complex64:
            C_2 = C_2.astype(np.complex64)
        x_FS *= C_2

    # apply post-FFT modulation
    x_FS = fftpack.fftn(x_FS, axes=axes)

    # apply modulation after FFT
    for d, ax in enumerate(axes):
        sh = [1] * x.ndim
        sh[ax] = N_s[d]
        C_1 = A[d].reshape(sh)
        x_FS *= C_1 / N_s[d]

    return x_FS


def iffsn(x_FS, T, T_c, N_FS, axes=None):
    r"""
    Signal samples from Fourier Series coefficients of a D-dimension signal.

    :py:func:`~pyffs.ffs.iffsn` is basically the inverse of :py:func:`~pyffs.ffs.ffsn`.

    Parameters
    ----------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) FS coefficients in ascending order.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Period mid-point for each dimension.
    N_FS : list(int)
        Function bandwidth along each dimension.
    axes : tuple
        Dimensions of `x_FS` along which FS coefficients are stored.

    Returns
    -------
    x : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) array containing original function
        samples given to :py:func:`~pyffs.ffs.ffsn`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ x \} = x`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffsn_sample`, :py:func:`~pyffs.ffs.ffsn`
    """
    axes, N_s = _verify_ffsn_input(x_FS, T, T_c, N_FS, axes)

    # check for input type
    if (x_FS.dtype == np.dtype("complex64")) or (x_FS.dtype == np.dtype("float32")):
        is_complex64 = True
        x = x_FS.copy().astype(np.complex64)
    else:
        is_complex64 = False
        x = x_FS.copy().astype(np.complex128)

    # apply pre-iFFT modulation
    B = []
    for d, N_sd in enumerate(N_s):
        A_d, B_d = _create_modulation_vectors(N_sd, N_FS[d], T[d], T_c[d])
        B.append(B_d)
        sh = [1] * x.ndim
        sh[axes[d]] = N_s[d]
        C_1 = A_d.reshape(sh)
        if is_complex64:
            C_1 = C_1.astype(np.complex64)
        x *= C_1

    # apply FFT
    x = fftpack.ifftn(x, axes=axes)

    # apply post-iFFT modulation
    for d, ax in enumerate(axes):
        sh = [1] * x.ndim
        sh[ax] = N_s[d]
        C_2 = B[d].reshape(sh)
        x *= C_2 * N_s[d]

    return x


def _ffsn(x, T, T_c, N_FS, axes=None):
    """
    [Slow] Fourier Series coefficients from signal samples of a D-dimension signal.

    For testing purposes only.

    Parameters
    ----------
    See :py:func:`~pyffs.ffs.ffsn`

    Returns
    -------
    See :py:func:`~pyffs.ffs.ffsn`
    """
    axes, _ = _verify_ffsn_input(x, T, T_c, N_FS, axes)

    # sequence of 1D FFS
    x_FS = x.copy()
    for d, ax in enumerate(axes):
        x_FS = ffs(x_FS, T[d], T_c[d], N_FS[d], axis=ax)

    return x_FS


def _iffsn(x_FS, T, T_c, N_FS, axes=None):
    """
    [Slow] Signal samples from Fourier Series coefficients of a D-dimension signal.

    For testing purposes only.

    Parameters
    ----------
    See :py:func:`~pyffs.ffs.iffsn`

    Returns
    -------
    See :py:func:`~pyffs.ffs.iffsn`
    """
    axes, _ = _verify_ffsn_input(x_FS, T, T_c, N_FS, axes)

    # sequence of 1D iFFS
    x = x_FS.copy()
    for d, ax in enumerate(axes):
        x = iffs(x, T[d], T_c[d], N_FS[d], axis=ax)

    return x
