# #############################################################################
# czt.py
# ======
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Methods for computing the chirp Z-transform.
"""

__all__ = ["czt", "cztn", "_cztn"]

import numpy as np
from scipy import fftpack as fftpack

from pyffs.util import _verify_cztn_input, _index_n


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
    Due to numerical instability when using large `M`, this implementation only
    supports transforms where `A` and `W` have unit norm.

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
    return cztn(x=x, A=[A], W=[W], M=[M], axes=(axis,))


def cztn(x, A, W, M, axes=None):
    """
    Multi-dimensional Chirp Z-transform.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        (..., N_1, N_2, ..., N_D, ...) input values.
    A : list(float or complex)
        Circular offset from the positive real-axis, for each dimension.
    W : list(float or complex)
        Circular spacing between transform points, for each dimension.
    M : list(int)
        Length of transform for each dimension.
    axes : tuple
        Dimensions of `x` along which transform should be applied.

    Returns
    -------
    x_czt : :py:class:`~numpy.ndarray`
        (..., M_1, M_2, ..., M_D, ...) transformed input along the axes indicated by `axes`.

    Notes
    -----
    Due to numerical instability when using large `M`, this implementation only supports transforms
    where each element of `A` and `W` has unit norm.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from pyffs import cztn

    Implementation of N-dimensional DFT:

    .. doctest::

       >>> N = M = 10
       >>> W = np.exp(-1j * 2 * np.pi / N)
       >>> x = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)  # extra dimension

       >>> dft_x = np.fft.fftn(x, axes=(1, 2))
       >>> czt_x = cztn(x, A=[1, 1], W=[W, W], M=[M, M], axes=(1, 2))

       >>> np.allclose(dft_x, czt_x)
       True
    """
    axes, A, W = _verify_cztn_input(x, A, W, M, axes)

    # Initialize variables
    D = len(axes)
    N = np.array(x.shape)[axes]
    L = []
    n = []
    for d in range(D):
        _L = fftpack.next_fast_len(N[d] + M[d] - 1)
        L.append(_L)
        n.append(np.arange(_L))

    # Initialize input
    sh_U = list(x.shape)
    for d in range(D):
        sh_U[axes[d]] = L[d]
    dtype_u = (
        np.complex64
        if ((x.dtype == np.dtype("complex64")) or (x.dtype == np.dtype("float32")))
        else np.complex128
    )
    u = np.zeros(sh_U, dtype=dtype_u)
    idx = _index_n(u, axes, [slice(n) for n in N])
    u[idx] = x

    # Modulate along each dimension
    for d in range(D):
        _n = n[d]
        _N = N[d]
        sh_N = [1] * x.ndim
        sh_N[axes[d]] = N[d]
        u_mod_d = (A[d] ** -_n[:_N]) * np.float_power(W[d], (_n[:_N] ** 2) / 2)
        u[idx] *= u_mod_d.reshape(sh_N)
    U = fftpack.fftn(u, axes=axes)

    # Convolve along each dimension -> multiply in frequency domain
    for d in range(D):
        _N = N[d]
        sh_L = [1] * x.ndim
        sh_L[axes[d]] = L[d]
        v = np.zeros(L[d], dtype=complex)
        v[: M[d]] = np.float_power(W[d], -(n[d][: M[d]] ** 2) / 2)
        v[L[d] - _N + 1 :] = np.float_power(W[d], -((L[d] - n[d][L[d] - _N + 1 :]) ** 2) / 2)
        V = fftpack.fft(v).reshape(sh_L)
        U *= V
    g = fftpack.ifftn(U, axes=axes)

    # Final modulation in time
    time_idx = _index_n(g, axes, [slice(m) for m in M])
    for d in range(D):
        _n = n[d]
        _M = M[d]
        sh_M = [1] * x.ndim
        sh_M[axes[d]] = _M
        g_mod = np.float_power(W[d], (_n[:_M] ** 2) / 2)
        g[time_idx] *= g_mod.reshape(sh_M)

    x_czt = g[time_idx]
    return x_czt


def _cztn(x, A, W, M, axes=None):
    """
    [Slow] Multi-dimensional Chirp Z-transform based on 1D composition.

    For testing purposes only.

    Parameters
    ----------
    See :py:func:`~pyffs.czt.cztn`

    Returns
    -------
    See :py:func:`~pyffs.czt.cztn`
    """
    axes, A, W = _verify_cztn_input(x, A, W, M, axes)

    # sequence of 1D CZT
    x_czt = x.copy()
    for d, ax in enumerate(axes):
        x_czt = czt(x_czt, A[d], W[d], M[d], axis=ax)

    return x_czt
