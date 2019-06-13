# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Fast Fourier Series library.
"""

import cmath

import numpy as np
import scipy.fftpack as fftpack

import pyffs._util as _util


def ffs_sample(T, N_FS, T_c, N_s):
    r"""
    Signal sample positions for :py:func:`~pyffs.ffs`.

    Return the coordinates at which a signal must be sampled to use
    :py:func:`~pyffs.ffs`.

    Parameters
    ----------
    T : float
        Function period.
    N_FS : int
        Function bandwidth.
    T_c : float
        Period mid-point.
    N_s : int
        Number of samples.

    Returns
    -------
    sample_point : :py:class:`~numpy.ndarray`
        (N_s,) coordinates at which to sample a signal (in the right order).

    Examples
    --------
    Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a bandlimited periodic function of period
    :math:`T = 1`, bandwidth :math:`N_{FS} = 5`, and with one period centered at :math:`T_{c} = \pi`.
    The sampling points :math:`t[n] \in \mathbb{R}` at which :math:`\phi` must be evaluated to
    compute the Fourier Series coefficients :math:`\left\{ \phi_{k}^{FS}, k = -2, \ldots, 2 \right\}`
    with :py:func:`~pyffs.ffs` are obtained as follows:

    .. testsetup::

       import numpy as np

       from pyffs import ffs_sample

    .. doctest::

       # Ideally choose N_s to be highly-composite for ffs().
       >>> sample_points = ffs_sample(T=1, N_FS=5, T_c=np.pi, N_s=8)
       >>> np.around(sample_points, 2)  # Notice points are not sorted.
       array([3.2 , 3.33, 3.45, 3.58, 2.7 , 2.83, 2.95, 3.08])

    See Also
    --------
    :py:func:`~pyffs.ffs`
    """
    if T <= 0:
        raise ValueError("Parameter[T] must be positive.")
    if N_FS < 3:
        raise ValueError("Parameter[N_FS] must be at least 3.")
    if N_s < N_FS:
        raise ValueError("Parameter[N_s] must be greater or equal to the signal bandwidth.")

    if N_s % 2 == 1:  # Odd-valued
        M = (N_s - 1) // 2
        idx = np.r_[0 : (M + 1), -M:0]
        sample_points = T_c + (T / N_s) * idx
    else:  # Even case
        M = N_s // 2
        idx = np.r_[0:M, -M:0]
        sample_points = T_c + (T / N_s) * (0.5 + idx)

    return sample_points


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
    if (x_FS.dtype == np.dtype("complex64")) or (x_FS.dtype == np.dtype("float32")):
        C_1 = C_1.astype(np.complex64)

    x = fftpack.ifft(x_FS * C_1, axis=axis)
    x *= C_2 * N_s
    return x


def czt(x, A, W, M, axis=-1):
    """
    Chirp Z-Transform.

    This implementation follows the semantics defined in :ref:`CZT_def`.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        (..., N, ...) input array.
    A : float or complex
        Circular offset from the positive real-axis.
    W : float or complex
        Circular spacing between transform points.
    M : int
        Length of the transform.
    axis : int
        Dimension of `x` along which the samples are stored.

    Returns
    -------
    X : :py:class:`~numpy.ndarray`
        (..., M, ...) transformed input along the axis indicated by `axis`.

    Notes
    -----
    Due to numerical instability when using large `M`, this implementation only supports transforms
    where `A` and `W` have unit norm.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from pyffs import czt

    Implementation of the DFT:

    .. doctest::

       >>> N = M = 10
       >>> x = np.random.randn(N, 3) + 1j * np.random.randn(N, 3)  # multi-dim

       >>> dft_x = np.fft.fft(x, axis=0)
       >>> czt_x = czt(x, A=1, W=np.exp(-1j * 2 * np.pi / N), M=M, axis=0)

       >>> np.allclose(dft_x, czt_x)
       True

    Implementation of the iDFT:

    .. doctest::

       >>> N = M = 10
       >>> x = np.random.randn(N) + 1j * np.random.randn(N)

       >>> idft_x = np.fft.ifft(x)
       >>> czt_x = czt(x, A=1, W=np.exp(1j * 2 * np.pi / N), M=M)

       >>> np.allclose(idft_x, czt_x / N)  # czt() does not do the scaling.
       True
    """
    A = complex(A)
    W = complex(W)

    if not cmath.isclose(abs(A), 1):
        raise ValueError("Parameter[A] must lie on the unit circle for numerical stability.")
    if not cmath.isclose(abs(W), 1):
        raise ValueError("Parameter[W] must lie on the unit circle.")
    if M <= 0:
        raise ValueError("Parameter[M] must be positive.")

    # Shape Parameters
    N = x.shape[axis]
    sh_N = [1] * x.ndim
    sh_N[axis] = N
    sh_M = [1] * x.ndim
    sh_M[axis] = M

    L = fftpack.next_fast_len(N + M - 1)
    sh_L = [1] * x.ndim
    sh_L[axis] = L
    sh_Y = list(x.shape)
    sh_Y[axis] = L

    y_dtype = (
        np.complex64
        if ((x.dtype == np.dtype("complex64")) or (x.dtype == np.dtype("float32")))
        else np.complex128
    )

    n = np.arange(L)
    y = np.zeros(sh_Y, dtype=y_dtype)
    y_mod = (A ** -n[:N]) * np.float_power(W, (n[:N] ** 2) / 2)
    y[_util._index(y, axis, slice(N))] = x
    y[_util._index(y, axis, slice(N))] *= y_mod.reshape(sh_N)
    Y = fftpack.fft(y, axis=axis)

    v = np.zeros(L, dtype=complex)
    v[:M] = np.float_power(W, -(n[:M] ** 2) / 2)
    v[L - N + 1 :] = np.float_power(W, -((L - n[L - N + 1 :]) ** 2) / 2)
    V = fftpack.fft(v).reshape(sh_L)

    G = Y
    G *= V
    g = fftpack.ifft(G, axis=axis)
    g_mod = np.float_power(W, (n[:M] ** 2) / 2)
    g[_util._index(g, axis, slice(M))] *= g_mod.reshape(sh_M)

    X = g[_util._index(g, axis, slice(M))]
    return X


def fs_interp(x_FS, T, a, b, M, axis=-1, real_x=False):
    r"""
    Interpolate bandlimited periodic signal.

    If `x_FS` holds the Fourier Series coefficients of a bandlimited periodic function
    :math:`x(t): \mathbb{R} \to \mathbb{C}`, then :py:func:`~pyffs.fs_interp`
    computes the values of :math:`x(t)` at points :math:`t[k] = (a + \frac{b - a}{M - 1} k) 1_{[0,\ldots,M-1]}[k]`.

    Parameters
    ----------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_FS, ...) FS coefficients in the order :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}\right]`.
    T : float
        Function period.
    a : float
        Interval LHS.
    b : float
        Interval RHS.
    M : int
        Number of points to interpolate.
    axis : int
        Dimension of `x_FS` along which the FS coefficients are stored.
    real_x : bool
        If True, assume that `x_FS` is conjugate symmetric and use a more efficient algorithm.
        In this case, the FS coefficients corresponding to negative frequencies are not used.

    Returns
    -------
    x : :py:class:`~numpy.ndarray`
        (..., M, ...) interpolated values :math:`\left[ x(t[0]), \ldots, x(t[M-1]) \right]` along
        the axis indicated by `axis`. If `real_x` is :py:obj:`True`, the output is real-valued,
        otherwise it is complex-valued.

    Examples
    --------
    .. testsetup::

       import math

       import numpy as np

       from pyffs import fs_interp

       def dirichlet(x, T, T_c, N_FS):
           y = x - T_c

           n, d = np.zeros((2, len(x)))
           nan_mask = np.isclose(np.fmod(y, np.pi), 0)
           n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
           d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
           n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
           d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

           return n / d

       # Parameters of the signal.
       T, T_c, N_FS = math.pi, math.e, 15
       N = (N_FS - 1) // 2

       # Generate interpolated signal
       a, b = T_c + (T / 2) *  np.r_[-1, 1]
       M = 100  # We want lots of points.
       diric_FS = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])


    Let :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}` be the Fourier Series (FS) coefficients of a
    shifted Dirichlet kernel of period :math:`T`:

    .. math::

       \phi_{k}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
           0 & \text{otherwise}.
       \end{cases}

    .. doctest::

       # Parameters of the signal.
       >>> T, T_c, N_FS = math.pi, math.e, 15
       >>> N = (N_FS - 1) // 2

       # And the kernel's FS coefficients.
       >>> diric_FS = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])


    Being bandlimited, we can use :py:func:`~pyffs.fs_interp` to numerically
    evaluate :math:`\phi(t)` on the interval :math:`\left[ T_{c} - \frac{T}{2}, T_{c} + \frac{T}{2} \right]`:

    .. doctest::

       # Generate interpolated signal
       >>> a, b = T_c + (T / 2) *  np.r_[-1, 1]
       >>> M = 100  # We want lots of points.
       >>> diric_sig = fs_interp(diric_FS, T, a, b, M)

       # Compare with theoretical result.
       >>> t = a + (b - a) / (M - 1) * np.arange(M)
       >>> diric_sig_exact = dirichlet(t, T, T_c, N_FS)

       >>> np.allclose(diric_sig, diric_sig_exact)
       True


    The Dirichlet kernel is real-valued, so we can set `real_x` to use the accelerated algorithm
    instead:

    .. doctest::

       # Generate interpolated signal
       >>> a, b = T_c + (T / 2) *  np.r_[-1, 1]
       >>> M = 100  # We want lots of points.
       >>> diric_sig = fs_interp(diric_FS, T, a, b, M, real_x=True)

       # Compare with theoretical result.
       >>> t = a + (b - a) / (M - 1) * np.arange(M)
       >>> diric_sig_exact = dirichlet(t, T, T_c, N_FS)

       >>> np.allclose(diric_sig, diric_sig_exact)
       True


    Notes
    -----
    Theory: :ref:`fp_interp_def`.

    See Also
    --------
    :py:func:`~pyffs.czt`
    """
    if T <= 0:
        raise ValueError("Parameter[T] must be positive.")
    if not (a < b):
        raise ValueError(f"Parameter[a] must be smaller than Parameter[b].")
    if M <= 0:
        raise ValueError("Parameter[M] must be positive.")

    # Shape Parameters
    N_FS = x_FS.shape[axis]
    N = (N_FS - 1) // 2
    sh = [1] * x_FS.ndim
    sh[axis] = M

    A = np.exp(-1j * 2 * np.pi / T * a)
    W = np.exp(1j * (2 * np.pi / T) * (b - a) / (M - 1))
    E = np.arange(M)

    if real_x:  # Real-valued functions.
        x0_FS = x_FS[_util._index(x_FS, axis, slice(N, N + 1))]
        xp_FS = x_FS[_util._index(x_FS, axis, slice(N + 1, N_FS))]
        C = np.reshape(W ** E, sh) / A

        x = czt(xp_FS, A, W, M, axis=axis)
        x *= 2 * C
        x += x0_FS
        x = x.real

    else:  # Complex-valued functions.
        C = np.reshape(W ** (-N * E), sh) * (A ** N)
        x = czt(x_FS, A, W, M, axis=axis)
        x *= C

    return x
