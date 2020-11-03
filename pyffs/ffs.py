# #############################################################################
# ffs.py
# ===========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric Bezzam [ebezzam@gmail.com]
# #############################################################################

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
    return ffsn(Phi=x, T=[T], T_c=[T_c], N_FS=[N_FS], axes=tuple([axis]))


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
    return iffsn(Phi_FS=x_FS, T=[T], T_c=[T_c], N_FS=[N_FS], axes=tuple([axis]))


def ffs2(Phi, T_x, T_y, T_cx, T_cy, N_FSx, N_FSy, axes=(-2, -1)):
    r"""
    Fourier Series coefficients from signal samples of a 2D function.

    Parameters
    ----------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffs2_sample`.
    T_x : float
        Function period along x-axis.
    T_y : float
        Function period along y-axis.
    T_cx : float
        Period mid-point, x-axis.
    T_cy : float
        Period mid-point, y-axis.
    N_FSx : int
        Function bandwidth, x-axis.
    N_FSy : int
        Function bandwidth, y-axis.
    axes : tuple
        Dimensions of `Phi` along which function samples are stored.

    Returns
    -------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) array containing Fourier Series coefficients in ascending order
        (top-left of matrix).

    Examples
    --------
    Let :math:`\phi(x, y)` be a shifted Dirichlet kernel of periods :math:`(T_x, T_y)` and
    bandwidths :math:`N_{FS, x} = 2 N_x + 1, N_{FS, y} = 2 N_y + 1`:

    .. math::

       \phi(x, y) &= \sum_{k_x = -N_x}^{N_x} \sum_{k_y = -N_y}^{N_y}
                \exp\left( j \frac{2 \pi}{T_x} k_x (x - T_{c,x}) \right)
                \exp\left( j \frac{2 \pi}{T_y} k_y (y - T_{c,y}) \right) \\
               &= \frac{\sin\left( N_{FS, x} \pi [x - T_{c,x}] / T_x \right)}{\sin\left( \pi
               [x - T_{c, x}] / T_x \right)} \frac{\sin\left( N_{FS, y} \pi [y - T_{c,y}] / T_y
               \right)}{\sin\left( \pi [y - T_{c, y}] / T_y \right)}.

    Its Fourier Series (FS) coefficients :math:`\phi_{k_x, k_y}^{FS}` can be analytically evaluated
    using the shift-modulation theorem:

    .. math::

       \phi_{k_x, k_y}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T_x} k_x T_{c,x} \right) \exp\left( -j \frac{2 \pi}{T_y} k_y
           T_{c,y} \right) & -N_x \le k_x \le N_x, -N_y \le k_y \le N_y,  \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pyffs.ffs.ffs2` to numerically evaluate
    :math:`\{\phi_{k_x, k_y}^{FS}, k_x = -N_x, \ldots, N_x, k_y = -N_y, \ldots, N_y\}`:

    .. testsetup::

       import math

       import numpy as np

       from pyffs import ffs2_sample, ffs2
       from pyffs.func import dirichlet_2D, dirichlet_fs

    .. doctest::

       >>> T_x = T_y = 1
       >>> T_cx = T_cy = 0
       >>> N_FSx = N_FSy = 3
       >>> N_sx = 4
       >>> N_sy = 3

       # Sample the kernel and do the transform.
       >>> sample_points, _ = ffs2_sample(T_x, T_y, N_FSx, N_FSy, T_cx, T_cy, N_sx, N_sy)
       >>> diric_samples = dirichlet_2D(sample_points, [T_x, T_y], [T_cx, T_cy], [N_FSx, N_FSy])
       >>> diric_FS = ffs2(diric_samples, T_x, T_y, T_cx, T_cy, N_FSx, N_FSy)

       # Compare with theoretical result.
       >>> diric_FS_exact = np.outer(dirichlet_fs(N_FSx, T_x, T_cx), dirichlet_fs(N_FSy, T_y, T_cy))
       >>> np.allclose(diric_FS[:N_FSx, :N_FSy], diric_FS_exact)
       True

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffs2_sample`, :py:func:`~pyffs.ffs.iffs2`
    """

    return ffsn(Phi=Phi, T=[T_x, T_y], T_c=[T_cx, T_cy], N_FS=[N_FSx, N_FSy], axes=axes)


def iffs2(Phi_FS, T_x, T_y, T_cx, T_cy, N_FSx, N_FSy, axes=(-2, -1)):
    r"""
    Signal samples from Fourier Series coefficients of a 2D function.

    :py:func:`~pyffs.ffs.iffs2` is basically the inverse of :py:func:`~pyffs.ffs.ffs2`.

    Parameters
    ----------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) FS coefficients in ascending order, namely in the top-left corner.
    T_x : float
        Function period along x-axis.
    T_y : float
        Function period along y-axis.
    T_cx : float
        Period mid-point, x-axis.
    T_cy : float
        Period mid-point, y-axis.
    N_FSx : int
        Function bandwidth, x-axis.
    N_FSy : int
        Function bandwidth, y-axis.
    axes : tuple
        Dimensions of `Phi_FS` along which FS coefficients are stored.

    Returns
    -------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) matrices containing original function samples given to
        :py:func:`~pyffs.ffs.ffs2`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ x \} = x`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffs2_sample`, :py:func:`~pyffs.ffs.ffs2`
    """

    return iffsn(Phi_FS=Phi_FS, T=[T_x, T_y], T_c=[T_cx, T_cy], N_FS=[N_FSx, N_FSy], axes=axes)


def ffsn_comp(Phi, T, T_c, N_FS, axes=None):
    r"""
    Fourier Series coefficients from signal samples of a D-dimension signal by performing D 1D FFTs
    along each dimension.

    Parameters
    ----------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffsn_sample`.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Function bandwidth along each dimension.
    N_FS : list(int)
        Period mid-point for each dimension.
    axes : tuple
        Dimensions of `Phi` along which function samples are stored.

    Returns
    -------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) array containing Fourier Series coefficients in ascending
        order (top-left of matrix).

    Examples
    --------
    Let :math:`\phi(x, y)` be a shifted Dirichlet kernel of periods :math:`(T_x, T_y)` and
    bandwidths :math:`N_{FS, x} = 2 N_x + 1, N_{FS, y} = 2 N_y + 1`:

    .. math::

       \phi(x, y) &= \sum_{k_x = -N_x}^{N_x} \sum_{k_y = -N_y}^{N_y}
                \exp\left( j \frac{2 \pi}{T_x} k_x (x - T_{c,x}) \right)
                \exp\left( j \frac{2 \pi}{T_y} k_y (y - T_{c,y}) \right) \\
               &= \frac{\sin\left( N_{FS, x} \pi [x - T_{c,x}] / T_x \right)}{\sin\left( \pi
               [x - T_{c, x}] / T_x \right)} \frac{\sin\left( N_{FS, y} \pi [y - T_{c,y}] / T_y
               \right)}{\sin\left( \pi [y - T_{c, y}] / T_y \right)}.

    Its Fourier Series (FS) coefficients :math:`\phi_{k_x, k_y}^{FS}` can be analytically evaluated
    using the shift-modulation theorem:

    .. math::

       \phi_{k_x, k_y}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T_x} k_x T_{c,x} \right) \exp\left( -j \frac{2 \pi}{T_y} k_y
           T_{c,y} \right) & -N_x \le k_x \le N_x, -N_y \le k_y \le N_y,  \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pyffs.ffs.ffsn_comp` to numerically evaluate
    :math:`\{\phi_{k_x, k_y}^{FS}, k_x = -N_x, \ldots, N_x, k_y = -N_y, \ldots, N_y\}`:

    .. testsetup::

       import math

       import numpy as np

       from pyffs import ffsn_sample, ffsn_comp
       from pyffs.func import dirichlet_2D, dirichlet_fs

    .. doctest::

       >>> T = [1, 1]
       >>> T_c = [0, 0]
       >>> N_FS = [3, 3]
       >>> N_s = [4, 3]

       # Sample the kernel and do the transform.
       >>> sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
       >>> diric_samples = dirichlet_2D(sample_points, T, T_c, N_FS)
       >>> diric_FS = ffsn_comp(Phi=diric_samples, T=T, N_FS=N_FS, T_c=T_c)

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
    :py:func:`~pyffs.util.ffsn_sample`, :py:func:`~pyffs.ffs.iffsn_comp`
    """

    axes, _ = _verify_ffsn_input(Phi, T, T_c, N_FS, axes)

    # sequence of 1D FFS
    Phi_FS = Phi.copy()
    for d, ax in enumerate(axes):
        Phi_FS = ffs(Phi_FS, T[d], T_c[d], N_FS[d], axis=ax)

    return Phi_FS


def iffsn_comp(Phi_FS, T, T_c, N_FS, axes=None):
    r"""
    Signal samples from Fourier Series coefficients of a D-dimension signal by performing D 1D iFFTs
    along each dimension.

    :py:func:`~pyffs.ffs.iffsn_comp` is basically the inverse of :py:func:`~pyffs.ffs.ffsn_comp` (
    and :py:func:`~pyffs.ffs.ffsn`).

    Parameters
    ----------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) FS coefficients in ascending order.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Function bandwidth along each dimension.
    N_FS : list(int)
        Period mid-point for each dimension.
    axes : tuple
        Dimensions of `Phi` along which function samples are stored.

    Returns
    -------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) matrices containing original function samples given to
        :py:func:`~pyffs.ffs.ffsn`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ x \} = x`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffsn_sample`, :py:func:`~pyffs.ffs.ffsn_comp`
    """

    axes, _ = _verify_ffsn_input(Phi_FS, T, T_c, N_FS, axes)

    # sequence of 1D iFFS
    Phi = Phi_FS.copy()
    for d, ax in enumerate(axes):
        Phi = iffs(Phi, T[d], T_c[d], N_FS[d], axis=ax)

    return Phi


def ffsn(Phi, T, T_c, N_FS, axes=None):
    r"""
    Fourier Series coefficients from signal samples of a D-dimension signal by performing a
    D-dimensional FFT.

    Parameters
    ----------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffsn_sample`.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Function bandwidth along each dimension.
    N_FS : list(int)
        Period mid-point for each dimension.
    axes : tuple
        Dimensions of `Phi` along which function samples are stored.

    Returns
    -------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) array containing Fourier Series coefficients in ascending
        order (top-left of matrix).

    Examples
    --------
    Let :math:`\phi(x, y)` be a shifted Dirichlet kernel of periods :math:`(T_x, T_y)` and
    bandwidths :math:`N_{FS, x} = 2 N_x + 1, N_{FS, y} = 2 N_y + 1`:

    .. math::

       \phi(x, y) &= \sum_{k_x = -N_x}^{N_x} \sum_{k_y = -N_y}^{N_y}
                \exp\left( j \frac{2 \pi}{T_x} k_x (x - T_{c,x}) \right)
                \exp\left( j \frac{2 \pi}{T_y} k_y (y - T_{c,y}) \right) \\
               &= \frac{\sin\left( N_{FS, x} \pi [x - T_{c,x}] / T_x \right)}{\sin\left( \pi
               [x - T_{c, x}] / T_x \right)} \frac{\sin\left( N_{FS, y} \pi [y - T_{c,y}] / T_y
               \right)}{\sin\left( \pi [y - T_{c, y}] / T_y \right)}.

    Its Fourier Series (FS) coefficients :math:`\phi_{k_x, k_y}^{FS}` can be analytically evaluated
    using the shift-modulation theorem:

    .. math::

       \phi_{k_x, k_y}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T_x} k_x T_{c,x} \right) \exp\left( -j \frac{2 \pi}{T_y} k_y
           T_{c,y} \right) & -N_x \le k_x \le N_x, -N_y \le k_y \le N_y,  \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pyffs.ffs.ffsn` to numerically evaluate
    :math:`\{\phi_{k_x, k_y}^{FS}, k_x = -N_x, \ldots, N_x, k_y = -N_y, \ldots, N_y\}`:

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
       >>> diric_FS = ffsn(Phi=diric_samples, T=T, N_FS=N_FS, T_c=T_c)

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

    axes, N_s = _verify_ffsn_input(Phi, T, T_c, N_FS, axes)

    # check for input type
    if (Phi.dtype == np.dtype("complex64")) or (Phi.dtype == np.dtype("float32")):
        is_complex64 = True
        Phi_FS = Phi.copy().astype(np.complex64)
    else:
        is_complex64 = False
        Phi_FS = Phi.copy().astype(np.complex128)

    # apply modulation before FFT
    A = []
    for d, N_sd in enumerate(N_s):
        A_d, B_d = _create_modulation_vectors(N_sd, N_FS[d], T[d], T_c[d])
        A.append(A_d.conj())
        sh = [1] * Phi.ndim
        sh[axes[d]] = N_s[d]
        C_2 = B_d.conj().reshape(sh)
        if is_complex64:
            C_2 = C_2.astype(np.complex64)
        Phi_FS *= C_2

    # apply FFT
    Phi_FS = fftpack.fftn(Phi_FS, axes=axes)

    # apply modulate after FFT
    for d, ax in enumerate(axes):
        sh = [1] * Phi.ndim
        sh[ax] = N_s[d]
        C_1 = A[d].reshape(sh)
        Phi_FS *= C_1 / N_s[d]

    return Phi_FS


def iffsn(Phi_FS, T, T_c, N_FS, axes=None):
    r"""
    Signal samples from Fourier Series coefficients of a D-dimension signal by performing a
    D-dimensional iFFT.

    :py:func:`~pyffs.ffs.iffsn` is basically the inverse of :py:func:`~pyffs.ffs.ffsn` (and
    :py:func:`~pyffs.ffs.ffsn_comp`).

    Parameters
    ----------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) FS coefficients in ascending order.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Function bandwidth along each dimension.
    N_FS : list(int)
        Period mid-point for each dimension.
    axes : tuple
        Dimensions of `Phi` along which function samples are stored.

    Returns
    -------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) matrices containing original function samples given to
        :py:func:`~pyffs.ffs.ffsn`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ x \} = x`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.util.ffsn_sample`, :py:func:`~pyffs.ffs.ffsn`
    """

    axes, N_s = _verify_ffsn_input(Phi_FS, T, T_c, N_FS, axes)

    # check for input type
    if (Phi_FS.dtype == np.dtype("complex64")) or (Phi_FS.dtype == np.dtype("float32")):
        is_complex64 = True
        Phi = Phi_FS.copy().astype(np.complex64)
    else:
        is_complex64 = False
        Phi = Phi_FS.copy().astype(np.complex128)

    # apply modulation before iFFT
    B = []
    for d, N_sd in enumerate(N_s):
        A_d, B_d = _create_modulation_vectors(N_sd, N_FS[d], T[d], T_c[d])
        B.append(B_d)
        sh = [1] * Phi.ndim
        sh[axes[d]] = N_s[d]
        C_1 = A_d.reshape(sh)
        if is_complex64:
            C_1 = C_1.astype(np.complex64)
        Phi *= C_1

    # apply FFT
    Phi = fftpack.ifftn(Phi, axes=axes)

    # apply modulate after iFFT
    for d, ax in enumerate(axes):
        sh = [1] * Phi.ndim
        sh[ax] = N_s[d]
        C_2 = B[d].reshape(sh)
        Phi *= C_2 * N_s[d]

    return Phi
