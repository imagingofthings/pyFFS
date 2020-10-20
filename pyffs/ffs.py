# #############################################################################
# ffs.py
# ===========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric Bezzam [ebezzam@gmail.com]
# #############################################################################

import numpy as np
from scipy import fftpack as fftpack


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

    It's Fourier Series (FS) coefficients :math:`\phi_{k}^{FS}` can be analytically evaluated using
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
       >>> sample_points = ffs_sample(T, N_FS, T_c, N_samples)
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

    M, N = np.r_[N_s, N_FS] // 2
    E_1 = np.r_[-N : (N + 1), np.zeros((N_s - N_FS,), dtype=int)]
    B_2 = np.exp(-1j * 2 * np.pi / N_s)
    if N_s % 2 == 1:
        B_1 = np.exp(1j * (2 * np.pi / T) * T_c)
        E_2 = np.r_[0 : (M + 1), -M:0]
    else:
        B_1 = np.exp(1j * (2 * np.pi / T) * (T_c + T / (2 * N_s)))
        E_2 = np.r_[0:M, -M:0]

    sh = [1] * x.ndim
    sh[axis] = N_s
    C_1 = np.reshape(B_1 ** (-E_1), sh)
    C_2 = np.reshape(B_2 ** (-N * E_2), sh)

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

    M, N = np.r_[N_s, N_FS] // 2
    E_1 = np.r_[-N : (N + 1), np.zeros((N_s - N_FS,), dtype=int)]
    B_2 = np.exp(-1j * 2 * np.pi / N_s)
    if N_s % 2 == 1:
        B_1 = np.exp(1j * (2 * np.pi / T) * T_c)
        E_2 = np.r_[0 : (M + 1), -M:0]
    else:
        B_1 = np.exp(1j * (2 * np.pi / T) * (T_c + T / (2 * N_s)))
        E_2 = np.r_[0:M, -M:0]

    sh = [1] * x_FS.ndim
    sh[axis] = N_s
    C_1 = np.reshape(B_1 ** (E_1), sh)
    C_2 = np.reshape(B_2 ** (N * E_2), sh)

    # Cast C_1 to 32 bits if x_FS is 32 bits. (Allows faster transforms.)
    if (x_FS.dtype == np.dtype("complex64")) or (
        x_FS.dtype == np.dtype("float32")
    ):
        C_1 = C_1.astype(np.complex64)

    x = fftpack.ifft(x_FS * C_1, axis=axis)
    x *= C_2 * N_s
    return x
