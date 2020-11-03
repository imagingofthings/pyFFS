# #############################################################################
# interp.py
# ===========
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import numpy as np

from pyffs.util import _index, _index_n, _verify_fs_interp_input
from pyffs.czt import czt, cztn


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


    Being bandlimited, we can use :py:func:`~pyffs.fs_interp` to numerically evaluate
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
    return fs_interpn(Phi_FS=x_FS, T=[T], a=[a], b=[b], M=[M], axes=(axis,), real_Phi=real_x)


def fs_interp2(Phi_FS, T_x, T_y, a_x, a_y, b_x, b_y, M_x, M_y, axes=(-2, -1), real_Phi=False):
    r"""
    Interpolate 2D bandlimited periodic signal.

    If `Phi_FS` holds the Fourier Series coefficients of a 2D bandlimited periodic function
    :math:`\phi(t): \mathbb{R}^2 \to \mathbb{C}`, then :py:func:`~pyffs.fs_interp2` computes the
    values of :math:`\phi(t)` at points :math:`x_m = (a_x + \frac{b_x - a_x}{M_x - 1} k_x)
    1_{[0,\ldots,M_x-1]}[k_x], y_n = (a_y + \frac{b_y - a_y}{M_y - 1} k_y)
    1_{[0,\ldots,M_y-1]}[k_y]`.

    Parameters
    ----------
    Phi_FS : :py:class:`~numpy.ndarray`
        (..., N_FSx, N_FSy, ...) FS coefficients in ascending order.
    T_x : float
        Function period, x-axis.
    T_y : float
        Function period, y-axis.
    a_x : float
        Interval LHS, x-axis
    a_y : float
        Interval LHS, y-axis
    b_x : float
        Interval RHS, x-axis.
    b_y : float
        Interval RHS, y-axis.
    M_x : int
        Number of points to interpolate, x-axis.
    M_y : int
        Number of points to interpolate, y-axis.
    axes : tuple, optional
        Dimension of `Phi_FS` along which the FS coefficients are stored.
    real_Phi : bool, optional
        If True, assume that `Phi_FS` is conjugate symmetric and use a more efficient algorithm. In
        this case, the FS coefficients corresponding to negative frequencies are not used.

    Returns
    -------
    Phi : :py:class:`~numpy.ndarray`
        (..., M_x, M_y, ...) interpolated values along the axes indicated by `axes`. If `real_x` is
        :py:obj:`True`, the output is real-valued, otherwise it is complex-valued.

    Notes
    -----
    Theory: :ref:`fp_interp_def`.

    See Also
    --------
    :py:func:`~pyffs.czt.cztn`, :py:func:`~pyffs.interp.fs_interpn`
    """

    return fs_interpn(
        Phi_FS=Phi_FS,
        T=[T_x, T_y],
        a=[a_x, a_y],
        b=[b_x, b_y],
        M=[M_x, M_y],
        axes=axes,
        real_Phi=real_Phi,
    )


def fs_interpn(Phi_FS, T, a, b, M, axes=None, real_Phi=False):
    r"""
    Interpolate D-dimensional bandlimited periodic signal.

    Parameters
    ----------
    Phi_FS : :py:class:`~numpy.ndarray`
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
        Dimensions of `Phi_FS` along which the FS coefficients are stored.
    real_Phi : bool, optional
        Whether time samples are real-valued.

    Returns
    -------
    Phi : :py:class:`~numpy.ndarray`
        (..., M_1, M_2, ..., M_D, ...) interpolated values along the axes indicated by `axes`.
        If `real_Phi` is :py:obj:`True`, the output is real-valued, otherwise it is complex-valued.

    Notes
    -----
    Theory: :ref:`fp_interp_def`.

    See Also
    --------
    :py:func:`~pyffs.czt.cztn`

    """

    axes = _verify_fs_interp_input(Phi_FS, T, a, b, M, axes)
    D = len(axes)

    # precompute modulation terms
    N_FS = np.array(Phi_FS.shape)[list(axes)]
    N = (N_FS - 1) // 2
    A = []
    W = []
    sh = []
    E = []
    for d in range(D):
        A.append(np.exp(-1j * 2 * np.pi / T[d] * a[d]))
        W.append(np.exp(1j * (2 * np.pi / T[d]) * (b[d] - a[d]) / (M[d] - 1)))
        sh.append([1] * Phi_FS.ndim)
        sh[d][axes[d]] = M[d]
        E.append(np.arange(M[d]))

    if real_Phi and D < 3:

        Phi0_FS = Phi_FS[_index_n(Phi_FS, axes, [slice(n, n + 1) for n in N])]

        if D == 1:
            Phi_pos_FS = Phi_FS[_index(Phi_FS, axes[0], slice(N[0] + 1, N_FS[0]))]
            C = np.reshape(W[0] ** E[0], sh[0]) / A[0]
            Phi = czt(Phi_pos_FS, A[0], W[0], M[0], axis=axes[0])
            Phi *= 2 * C
            Phi += Phi0_FS

        elif D == 2:

            # positive / positive
            Phi_pos_pos_FS = Phi_FS[
                _index_n(Phi_FS, axes, [slice(N[d], N_FS[d]) for d in range(D)])
            ]
            Phi_pos_pos = cztn(Phi_pos_pos_FS, A, W, M, axes=axes)

            # negative / positive
            Phi_neg_pos_FS = Phi_FS[
                _index_n(Phi_FS, axes, [slice(0, N[0]), slice(N[1] + 1, N_FS[1])])
            ]
            Phi_neg_pos = cztn(Phi_neg_pos_FS, A, W, M, axes=axes)
            Phi_neg_pos *= np.reshape(W[0] ** (-N[0] * E[0]), sh[0]) * (A[0] ** N[0])
            Phi_neg_pos *= np.reshape(W[1] ** E[1], sh[1]) / A[1]

            # exploit conjugate symmetry
            Phi = 2 * Phi_pos_pos - Phi0_FS + 2 * Phi_neg_pos

    else:

        # apply CZT
        Phi = cztn(Phi_FS, A, W, M, axes=axes)

        # modulate along each dimension
        for d in range(D):
            C = np.reshape(W[d] ** (-N[d] * E[d]), sh[d]) * (A[d] ** N[d])
            Phi *= C

    if real_Phi:
        return Phi.real
    else:
        return Phi
