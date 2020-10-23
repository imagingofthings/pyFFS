# #############################################################################
# interp.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################


import numpy as np

from pyffs import util as _util
from pyffs.czt import czt


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
    :py:func:`~pyffs.czt.czt`
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
