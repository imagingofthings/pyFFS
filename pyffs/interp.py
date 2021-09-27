# #############################################################################
# interp.py
# =========
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Methods for interpolating functions using Fourier Series.
"""

from pyffs.czt import czt, cztn
from pyffs.util import (
    _index,
    _index_n,
    _verify_fs_interp_input,
)
from pyffs.backend import get_array_module


def fs_interp(x_FS, T, a, b, M, axis=-1, real_x=False):
    r"""
    Interpolate bandlimited periodic signal.

    If `x_FS` holds the Fourier Series coefficients of a bandlimited periodic function :math:`x(t):
    \mathbb{R} \to \mathbb{C}`, then :py:func:`~pyffs.fs_interp` computes the values of :math:`x(t)`
    at points :math:`t[k] = (a + \frac{b - a}{M - 1} k) 1_{[0,\ldots,M-1]}[k]`.

    Parameters
    ----------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_FS, ...) FS coefficients in the order :math:`\left[ x_{-N}^{FS}, \ldots,
        x_{N}^{FS}\right]`.
    T : float
        Function period.
    a : float
        Interval LHS.
    b : float
        Interval RHS.
    M : int
        Number of points to interpolate.
    axis : int, optional
        Dimension of `x_FS` along which the FS coefficients are stored.
    real_x : bool, optional
        If True, assume that `x_FS` is conjugate symmetric and use a more efficient algorithm. In
        this case, the FS coefficients corresponding to negative frequencies are not used.

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
       from pyffs.func import dirichlet

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


    Being bandlimited, we can use :py:func:`~pyffs.interp.fs_interp` to numerically evaluate
    :math:`\phi(t)` on the interval :math:`\left[ T_{c} - \frac{T}{2}, T_{c} + \frac{T}{2} \right]`:

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
    :py:func:`~pyffs.czt.czt`, :py:func:`~pyffs.interp.fs_interpn`
    """
    return fs_interpn(x_FS=x_FS, T=[T], a=[a], b=[b], M=[M], axes=(axis,), real_x=real_x)


def fs_interpn(x_FS, T, a, b, M, axes=None, real_x=False):
    r"""
    Interpolate D-dimensional bandlimited periodic signal.

    Parameters
    ----------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_FSx, N_FSy, ...) FS coefficients in ascending order.
    T : list(float)
        Function period along each dimension.
    a : list(float)
        Interval LHS for each dimension.
    b : list(float)
        Interval RHS for each dimension.
    M : list(int)
        Number of points to interpolate for each dimension.
    axes : tuple, optional
        Dimensions of `x_FS` along which the FS coefficients are stored.
    real_x : bool, optional
        If True, assume that `x_FS` is conjugate symmetric in each dimension
        and use a more efficient algorithm. In this case, the FS coefficients
        corresponding to negative frequencies are not used. Note that this
        approach is only available for D < 3, and will raise an error otherwise.

    Returns
    -------
    x : :py:class:`~numpy.ndarray`
        (..., M_1, M_2, ..., M_D, ...) interpolated values along the axes indicated by `axes`.
        If `real_x` is :py:obj:`True`, the output is real-valued, otherwise it is complex-valued.

    Notes
    -----
    Theory: :ref:`fp_interp_def`.

    See Also
    --------
    :py:func:`~pyffs.czt.cztn`

    """
    axes = _verify_fs_interp_input(x_FS, T, a, b, M, axes)
    D = len(axes)

    xp = get_array_module(x_FS)

    # precompute modulation terms
    N_FS = [x_FS.shape[d] for d in axes]
    N = [(nfs - 1) // 2 for nfs in N_FS]
    A = []
    W = []
    sh = []
    E = []
    for d in range(D):
        A.append(xp.exp(-1j * 2 * xp.pi / T[d] * a[d]))
        W.append(xp.exp(1j * (2 * xp.pi / T[d]) * (b[d] - a[d]) / (M[d] - 1)))
        sh.append([1] * x_FS.ndim)
        sh[d][axes[d]] = M[d]
        E.append(xp.arange(M[d]))

    if real_x:
        x0_FS = x_FS[_index_n(x_FS, axes, [slice(n, n + 1) for n in N])]

        if D == 1:
            x_FS_p = x_FS[_index(x_FS, axes[0], slice(N[0] + 1, N_FS[0]))]
            C = xp.reshape(W[0] ** E[0], sh[0]) / A[0]
            x = czt(x_FS_p, A[0], W[0], M[0], axis=axes[0])

            # exploit conjugate symmetry
            x = 2 * C * x + x0_FS

        elif D == 2:
            # positive / positive
            x_FS_pp = x_FS[_index_n(x_FS, axes, [slice(N[d], N_FS[d]) for d in range(D)])]
            x_pp = cztn(x_FS_pp, A, W, M, axes=axes)

            # negative / positive
            x_FS_np = x_FS[_index_n(x_FS, axes, [slice(0, N[0]), slice(N[1] + 1, N_FS[1])])]
            x_np = cztn(x_FS_np, A, W, M, axes=axes)
            x_np *= xp.reshape(W[0] ** (-N[0] * E[0]), sh[0]) * (A[0] ** N[0])
            x_np *= xp.reshape(W[1] ** E[1], sh[1]) / A[1]

            # exploit conjugate symmetry
            x = 2 * x_pp + 2 * x_np - x0_FS
        else:
            raise NotImplementedError("[real_x] approach not available for D > 2.")

        return x.real
    else:  # General complex case.
        x = cztn(x_FS, A, W, M, axes=axes)

        # modulate along each dimension
        for d in range(D):
            C = xp.reshape(W[d] ** (-N[d] * E[d]), sh[d]) * (A[d] ** N[d])
            x *= C

        return x
