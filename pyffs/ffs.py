# #############################################################################
# ffs.py
# ===========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric Bezzam [ebezzam@gmail.com]
# #############################################################################

import numpy as np
from scipy import fftpack as fftpack

from pyffs.util import _create_modulation_vectors


def ffs(x, T, T_c, N_FS, axis=-1):
    r"""
    Fourier Series coefficients from signal samples.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        (..., N_s, ...) function values at sampling points specified by :py:func:`~pyffs.ffs_sample`.
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
        (..., N_s, ...) vectors containing entries :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}}`.

    Examples
    --------
    Let :math:`\phi(t)` be a shifted Dirichlet kernel of period :math:`T` and bandwidth :math:`N_{FS} = 2 N + 1`:

    .. math::

       \phi(t) = \sum_{k = -N}^{N} \exp\left( j \frac{2 \pi}{T} k (t - T_{c}) \right)
               = \frac{\sin\left( N_{FS} \pi [t - T_{c}] / T \right)}{\sin\left( \pi [t - T_{c}] / T \right)}.

    Its Fourier Series (FS) coefficients :math:`\phi_{k}^{FS}` can be analytically evaluated using
    the shift-modulation theorem:

    .. math::

       \phi_{k}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pyffs.ffs` to numerically evaluate
    :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}`:

    .. testsetup::

       import math

       import numpy as np

       from pyffs import ffs_sample, ffs

       def dirichlet(x, T, T_c, N_FS):
           y = x - T_c

           n, d = np.zeros((2, len(x)))
           nan_mask = np.isclose(np.fmod(y, np.pi), 0)
           n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
           d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
           n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
           d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

           return n / d

    .. doctest::

       >>> T, T_c, N_FS = math.pi, math.e, 15
       >>> N_samples = 16  # Any >= N_FS will do, but highly-composite best.

       # Sample the kernel and do the transform.
       >>> sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
       >>> diric_samples = dirichlet(sample_points, T, T_c, N_FS)
       >>> diric_FS = ffs(diric_samples, T, T_c, N_FS)

       # Compare with theoretical result.
       >>> N = (N_FS - 1) // 2
       >>> diric_FS_exact = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])

       >>> np.allclose(diric_FS[:N_FS], diric_FS_exact)
       True

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.ffs_sample`, :py:func:`~pyffs.iffs`
    """
    N_s = x.shape[axis]

    if T <= 0:
        raise ValueError("Parameter[T] must be positive.")
    if not (3 <= N_FS <= N_s):
        raise ValueError(f"Parameter[N_FS] must lie in {{3, ..., N_s}}.")

    # compute modulatation vectors
    A, B = _create_modulation_vectors(N_s, N_FS, T, T_c)
    sh = [1] * x.ndim
    sh[axis] = N_s
    C_1 = np.reshape(A.conj(), sh)
    C_2 = np.reshape(B.conj(), sh)

    # Cast C_2 to 32 bits if x is 32 bits. (Allows faster transforms.)
    if (x.dtype == np.dtype("complex64")) or (x.dtype == np.dtype("float32")):
        C_2 = C_2.astype(np.complex64)

    x_FS = fftpack.fft(x * C_2, axis=axis)
    x_FS *= C_1 / N_s
    return x_FS


def iffs(x_FS, T, T_c, N_FS, axis=-1):
    r"""
    Signal samples from Fourier Series coefficients.

    :py:func:`~pyffs.iffs` is basically the inverse of :py:func:`~pyffs.ffs`.

    Parameters
    ----------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_s, ...) FS coefficients in the order :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}}`.
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
        (..., N_s, ...) vectors containing original function samples given to :py:func:`~pyffs.ffs`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ x \} = x`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.ffs_sample`, :py:func:`~pyffs.ffs`
    """
    N_s = x_FS.shape[axis]

    if T <= 0:
        raise ValueError("Parameter[T] must be positive.")
    if not (3 <= N_FS <= N_s):
        raise ValueError(f"Parameter[N_FS] must lie in {{3, ..., N_s}}.")

    # compute modulatation vectors
    A, B = _create_modulation_vectors(N_s, N_FS, T, T_c)
    sh = [1] * x_FS.ndim
    sh[axis] = N_s
    C_1 = np.reshape(A, sh)
    C_2 = np.reshape(B, sh)

    # Cast C_1 to 32 bits if x_FS is 32 bits. (Allows faster transforms.)
    if (x_FS.dtype == np.dtype("complex64")) or (x_FS.dtype == np.dtype("float32")):
        C_1 = C_1.astype(np.complex64)

    x = fftpack.ifft(x_FS * C_1, axis=axis)
    x *= C_2 * N_s
    return x


def ffs2(Phi, Tx, Ty, T_cx, T_cy, N_FSx, N_FSy, axes=(-2, -1)):
    r"""
    Fourier Series coefficients from signal samples of a 2D function.

    Parameters
    ----------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) function values at sampling points specified by
        :py:func:`~pyffs.ffs2_sample`.
    Tx : float
        Function period along x-axis.
    Ty : float
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
        (..., N_sx, N_sy, ...) array containing Fourier Series coefficients in
        ascending order (top-left of matrix).

    Examples
    --------
    Let :math:`\phi(x, y)` be a shifted Dirichlet kernel of periods
    :math:`(T_x, T_y)` and bandwidths :math:`N_{FS, x} = 2 N_x + 1,
    N_{FS, y} = 2 N_y + 1`:

    .. math::

       \phi(x, y) &= \sum_{k_x = -N_x}^{N_x} \sum_{k_y = -N_y}^{N_y}
                \exp\left( j \frac{2 \pi}{T_x} k_x (x - T_{c,x}) \right)
                \exp\left( j \frac{2 \pi}{T_y} k_y (y - T_{c,y}) \right) \\
               &= \frac{\sin\left( N_{FS, x} \pi [x - T_{c,x}] / T_x \right)}{\sin\left( \pi [x - T_{c, x}] / T_x \right)}
               \frac{\sin\left( N_{FS, y} \pi [y - T_{c,y}] / T_y \right)}{\sin\left( \pi [y - T_{c, y}] / T_y \right)}.

    Its Fourier Series (FS) coefficients :math:`\phi_{k_x, k_y}^{FS}` can be
    analytically evaluated using the shift-modulation theorem:

    .. math::

       \phi_{k_x, k_y}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T_x} k_x T_{c,x} \right)
           \exp\left( -j \frac{2 \pi}{T_y} k_y T_{c,y} \right)
           & -N_x \le k_x \le N_x, -N_y \le k_y \le N_y,  \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pyffs.ffs2` to numerically
    evaluate :math:`\{\phi_{k_x, k_y}^{FS}, k_x = -N_x, \ldots, N_x,
    k_y = -N_y, \ldots, N_y\}`:

    .. testsetup::

       import math

       import numpy as np

       from pyffs import ffs2_sample, ffs2

       def dirichlet(x, T, T_c, N_FS):
           y = x - T_c

           n, d = np.zeros((2, len(x)))
           nan_mask = np.isclose(np.fmod(y, np.pi), 0)
           n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
           d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
           n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
           d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

           return n / d

       def dirichlet_2D(sample_points, Tx, Ty, T_cx, T_cy, N_FSx, N_FSy):

           # compute along x and y, then combine
           x_vals = dirichlet(x=np.squeeze(sample_points[0]), T=Tx, T_c=T_cx, N_FS=N_FSx)
           y_vals = dirichlet(x=np.squeeze(sample_points[1]), T=Ty, T_c=T_cy, N_FS=N_FSy)
           return np.outer(x_vals, y_vals)

       def dirichlet_fs(N_FS, T, T_c):
           N = (N_FS - 1) // 2
           return np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N : N + 1])

    .. doctest::

       >>> Tx = Ty = 1
       >>> T_cx = T_cy = 0
       >>> N_FSx = N_FSy = 3
       >>> N_sx = 4
       >>> N_sy = 3

       # Sample the kernel and do the transform.
       >>> sample_points, _ = ffs2_sample(
       ... Tx, Ty, N_FSx, N_FSy, T_cx, T_cy, N_sx, N_sy,
       ... )
       >>> diric_samples = dirichlet_2D(
       ... sample_points, Tx, Ty, T_cx, T_cy, N_FSx, N_FSy,
       ... )
       >>> diric_FS = ffs2(
       ... diric_samples, Tx, Ty, T_cx, T_cy, N_FSx, N_FSy,
       ... )

       # Compare with theoretical result.
       >>> diric_FS_exact = np.outer(
       ... dirichlet_fs(N_FSx, Tx, T_cx), dirichlet_fs(N_FSy, Ty, T_cy)
       ... )
       >>> np.allclose(diric_FS[:N_FSx, :N_FSy], diric_FS_exact)
       True

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.ffs2_sample`, :py:func:`~pyffs.iffs2`
    """
    if len(Phi.shape) > 2:
        raise NotImplementedError

    N_sx = Phi.shape[axes[0]]
    N_sy = Phi.shape[axes[1]]
    N_s = N_sx * N_sy

    if Tx <= 0:
        raise ValueError("Parameter[Tx] must be positive.")
    if Ty <= 0:
        raise ValueError("Parameter[Ty] must be positive.")
    if not (3 <= N_FSx <= N_sx):
        raise ValueError(f"Parameter[N_FSx] must lie in {{3, ..., N_sx}}.")
    if not (3 <= N_FSy <= N_sy):
        raise ValueError(f"Parameter[N_FSy] must lie in {{3, ..., N_sy}}.")

    # create modulation vectors for each dimension
    Ax, Bx = _create_modulation_vectors(N_sx, N_FSx, Tx, T_cx)
    Ay, By = _create_modulation_vectors(N_sy, N_FSy, Ty, T_cy)
    C_1 = np.outer(Ax.conj(), Ay.conj())
    C_2 = np.outer(Bx.conj(), By.conj())

    # Cast C_2 to 32 bits if x is 32 bits. (Allows faster transforms.)
    if (Phi.dtype == np.dtype("complex64")) or (Phi.dtype == np.dtype("float32")):
        C_2 = C_2.astype(np.complex64)

    x_FS = fftpack.fft2(Phi * C_2, axes=axes)
    x_FS *= C_1 / N_s
    return x_FS


def iffs2(Phi_FS, Tx, Ty, T_cx, T_cy, N_FSx, N_FSy, axes=(-2, -1)):
    r"""
    Signal samples from Fourier Series coefficients of a 2D function.

    :py:func:`~pyffs.iffs2` is basically the inverse of :py:func:`~pyffs.ffs2`.

    Parameters
    ----------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) FS coefficients in ascending order, namely in
        the top-left corner.
    Tx : float
        Function period along x-axis.
    Ty : float
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
        (..., N_sx, N_sy, ...) matrices containing original function samples
        given to :py:func:`~pyffs.ffs2`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ x \} = x`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pyffs.ffs2_sample`, :py:func:`~pyffs.ffs2`
    """

    if len(Phi_FS.shape) > 2:
        raise NotImplementedError

    N_sx = Phi_FS.shape[axes[0]]
    N_sy = Phi_FS.shape[axes[1]]
    N_s = N_sx * N_sy

    if Tx <= 0:
        raise ValueError("Parameter[Tx] must be positive.")
    if Ty <= 0:
        raise ValueError("Parameter[Ty] must be positive.")
    if not (3 <= N_FSx <= N_sx):
        raise ValueError(f"Parameter[N_FSx] must lie in {{3, ..., N_sx}}.")
    if not (3 <= N_FSy <= N_sy):
        raise ValueError(f"Parameter[N_FSy] must lie in {{3, ..., N_sy}}.")

    # create modulation vectors for each dimension
    Ax, Bx = _create_modulation_vectors(N_sx, N_FSx, Tx, T_cx)
    Ay, By = _create_modulation_vectors(N_sy, N_FSy, Ty, T_cy)
    C_1 = np.outer(Ax, Ay)
    C_2 = np.outer(Bx, By)

    # Cast C_1 to 32 bits if x_FS is 32 bits. (Allows faster transforms.)
    if (Phi_FS.dtype == np.dtype("complex64")) or (Phi_FS.dtype == np.dtype("float32")):
        C_1 = C_1.astype(np.complex64)

    Phi = fftpack.ifft2(Phi_FS * C_1, axes=axes)
    Phi *= C_2 * N_s
    return Phi
