# #############################################################################
# func.py
# =======
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric Bezzam [ebezzam@gmail.com]
# #############################################################################

"""
Methods for computing samples and Fourier Series coefficients of specific
functions.
"""

import numpy as np


def dirichlet(x, T, T_c, N_FS):
    r"""
    Return samples of a shifted Dirichlet kernel of period :math:`T` and
    bandwidth :math:`N_{FS} = 2 N + 1`:

    .. math::

       \phi(t) = \sum_{k = -N}^{N} \exp\left( j \frac{2 \pi}{T} k (t - T_{c}) \right)
               = \frac{\sin\left( N_{FS} \pi [t - T_{c}] / T \right)}{\sin\left( \pi [t - T_{c}]
               / T \right)}.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Sampling points.
    T : float
        Function period.
    T_c : float
        Period mid-point.
    N_FS : int
        Function bandwidth.

    Returns
    -------
    vals : :py:class:`~numpy.ndarray`
        Function values.

    See Also
    --------
    :py:func:`~pyffs.func.dirichlet_fs`
    """
    y = x - T_c
    n, d = np.zeros((2, len(x)))
    nan_mask = np.isclose(np.fmod(y, np.pi), 0)
    n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
    d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
    n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
    d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

    return n / d


def dirichlet_fs(N_FS, T, T_c):
    """
    Return Fourier Series coefficients of a shifted Dirichlet kernel of period
    :math:`T` and bandwidth :math:`N_{FS} = 2 N + 1`.

    Parameters
    ----------
    N_FS : int
        Function bandwidth.
    T : float
        Function period.
    T_c : float
        Period mid-point.

    Returns
    -------
    vals : :py:class:`~numpy.ndarray`
        Fourier Series coefficients in ascending order.

    See Also
    --------
    :py:func:`~pyffs.func.dirichlet`
        (N_FS,) Fourier Series coefficients.
    """
    N = (N_FS - 1) // 2
    return np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N : N + 1])


def dirichlet_2D(sample_points, T, T_c, N_FS):
    r"""
    Return samples of a shifted 2D Dirichlet kernel of period :math:`(T_x, T_y)`
    and bandwidth :math:`N_{FS, x} = 2 N_x + 1, N_{FS, y} = 2 N_y + 1`:

    .. math::

       \phi(x, y) &= \sum_{k_x = -N_x}^{N_x} \sum_{k_y = -N_y}^{N_y}
                \exp\left( j \frac{2 \pi}{T_x} k_x (x - T_{c,x}) \right)
                \exp\left( j \frac{2 \pi}{T_y} k_y (y - T_{c,y}) \right) \\
               &= \frac{\sin\left( N_{FS, x} \pi [x - T_{c,x}] / T_x \right)}{\sin\left( \pi
               [x - T_{c, x}] / T_x \right)} \frac{\sin\left( N_{FS, y} \pi [y - T_{c,y}] / T_y
               \right)}{\sin\left( \pi [y - T_{c, y}] / T_y \right)}.

    Parameters
    ----------
    sample_points : list(:py:class:`~numpy.ndarray`)
        (2,) coordinates at which to sample the function in the x- and
        y-dimensions respectively.
    T : list(float)
        Function period.
    T_c : list(float)
        Period mid-point.
    N_FS : list(int)
        Function bandwidth.

    Returns
    -------
    vals : :py:class:`~numpy.ndarray`
        Function values at `sample_points`.

    See Also
    --------
    :py:func:`~pyffs.util.ffsn_sample`, :py:func:`~pyffs.func.dirichlet_fs`
    """
    # compute along x and y, then combine
    x_vals = dirichlet(x=sample_points[0][:, 0], T=T[0], T_c=T_c[0], N_FS=N_FS[0])
    y_vals = dirichlet(x=sample_points[1][0, :], T=T[1], T_c=T_c[1], N_FS=N_FS[1])
    return np.outer(x_vals, y_vals)
