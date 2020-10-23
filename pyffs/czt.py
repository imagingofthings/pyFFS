# #############################################################################
# czt.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################


import cmath

import numpy as np
from scipy import fftpack as fftpack

from pyffs import util as _util


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
