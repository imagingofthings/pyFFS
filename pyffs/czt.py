# #############################################################################
# czt.py
# ===========
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [eric.bezzam@gmail.com]
# #############################################################################


import cmath

import numpy as np
from scipy import fftpack as fftpack
from pyffs.util import _verify_cztn_input, _index, _index_n


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
    y[_index(y, axis, slice(N))] = x
    y[_index(y, axis, slice(N))] *= y_mod.reshape(sh_N)
    Y = fftpack.fft(y, axis=axis)

    v = np.zeros(L, dtype=complex)
    v[:M] = np.float_power(W, -(n[:M] ** 2) / 2)
    v[L - N + 1 :] = np.float_power(W, -((L - n[L - N + 1 :]) ** 2) / 2)
    V = fftpack.fft(v).reshape(sh_L)

    G = Y
    G *= V
    g = fftpack.ifft(G, axis=axis)
    g_mod = np.float_power(W, (n[:M] ** 2) / 2)
    g[_index(g, axis, slice(M))] *= g_mod.reshape(sh_M)

    X = g[_index(g, axis, slice(M))]
    return X


def czt2(Phi, Ax, Ay, Wx, Wy, Mx, My, axes=(-2, -1)):
    return cztn(Phi, A=[Ax, Ay], W=[Wx, Wy], M=[Mx, My], axes=axes)


def cztn(Phi, A, W, M, axes=None):
    """
    Perform multi-dimensional CZT from signal samples, using the multi-dimensional FFT.

    Parameters
    ----------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) input values.
    A : list(float or complex)
        Circular offset from the positive real-axis, for each dimension.
    W : list(float or complex)
        Circular spacing between transform points, for each dimension.
    M : list(int)
        Length of transform for each dimension.
    axes : tuple
        Dimensions of `Phi` along which transform should be applied.

    Returns
    -------
    X : :py:class:`~numpy.ndarray`
        (..., M_1, M_2, ..., M_D, ...) transformed input along the axis indicated by `axes`.

    Notes
    -----
    Due to numerical instability when using large `M`, this implementation only supports transforms
    where `A` and `W` have unit norm.
    """

    axes, A, W = _verify_cztn_input(Phi, A, W, M, axes)

    # Initialize variables
    D = len(axes)
    N = np.array(Phi.shape)[axes]
    L = []
    n = []
    for d in range(D):
        _L = fftpack.next_fast_len(N[d] + M[d] - 1)
        L.append(_L)
        n.append(np.arange(_L))

    # Initialize input
    sh_U = list(Phi.shape)
    for d in range(D):
        sh_U[axes[d]] = L[d]
    u_dtype = (
        np.complex64
        if ((Phi.dtype == np.dtype("complex64")) or (Phi.dtype == np.dtype("float32")))
        else np.complex128
    )
    u = np.zeros(sh_U, dtype=u_dtype)
    idx = _index_n(u, axes, [slice(n) for n in N])
    u[idx] = Phi

    # Modulate along each dimension
    for d in range(D):
        _n = n[d]
        _N = N[d]
        sh_N = [1] * Phi.ndim
        sh_N[axes[d]] = N[d]
        u_mod_d = (A[d] ** -_n[:_N]) * np.float_power(W[d], (_n[:_N] ** 2) / 2)
        u[idx] *= u_mod_d.reshape(sh_N)
    U = fftpack.fft2(u, axes=axes)

    # Convolve along each dimension -> multiply in frequency domain
    for d in range(D):
        _N = N[d]
        sh_L = [1] * Phi.ndim
        sh_L[axes[d]] = L[d]
        v = np.zeros(L[d], dtype=complex)
        v[: M[d]] = np.float_power(W[d], -(n[d][: M[d]] ** 2) / 2)
        v[L[d] - _N + 1 :] = np.float_power(W[d], -((L[d] - n[d][L[d] - _N + 1 :]) ** 2) / 2)
        V = fftpack.fft(v).reshape(sh_L)
        U *= V
    g = fftpack.ifft2(U, axes=axes)

    # Final modulation in time
    time_idx = _index_n(g, axes, [slice(m) for m in M])
    for d in range(D):
        _n = n[d]
        _M = M[d]
        sh_M = [1] * Phi.ndim
        sh_M[axes[d]] = _M
        g_mod = np.float_power(W[d], (_n[:_M] ** 2) / 2)
        g[time_idx] *= g_mod.reshape(sh_M)

    Phi_czt = g[time_idx]
    return Phi_czt


def cztn_comp(Phi, A, W, M, axes=None):
    """
    Perform multi-dimensional CZT from signal samples of a D-dimension signal by performing D 1D
    CZTs along each dimension.

    Parameters
    ----------
    Phi : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) input values.
    A : list(float or complex)
        Circular offset from the positive real-axis, for each dimension.
    W : list(float or complex)
        Circular spacing between transform points, for each dimension.
    M : list(int)
        Length of transform for each dimension.
    axes : tuple
        Dimensions of `Phi` along which transform should be applied.

    Returns
    -------
    X : :py:class:`~numpy.ndarray`
        (..., M_1, M_2, ..., M_D, ...) transformed input along the axis indicated by `axes`.

    Notes
    -----
    Due to numerical instability when using large `M`, this implementation only supports transforms
    where `A` and `W` have unit norm.
    """

    axes, A, W = _verify_cztn_input(Phi, A, W, M, axes)

    # sequence of 1D FFS
    Phi_czt = Phi.copy()
    for d, ax in enumerate(axes):
        Phi_czt = czt(Phi_czt, A[d], W[d], M[d], axis=ax)

    return Phi_czt
