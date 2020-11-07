# #############################################################################
# util.py
# ========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Helper functions.
"""

import cmath

import numpy as np


def _index(x, axis, index_spec):
    """
    Form indexing tuple for NumPy arrays.

    Given an array `x`, generates the indexing tuple that has :py:class:`slice`
    in each axis except `axis`, where `index_spec` is used instead.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Array to index.
    axis : int
        Dimension along which to apply `index_spec`.
    index_spec : int or :py:class:`slice`
        Index/slice to use.

    Returns
    -------
    indexer : tuple
        Indexing tuple.
    """
    return _index_n(x=x, axes=[axis], index_spec=[index_spec])


def _index_n(x, axes, index_spec):
    """
    Form indexing tuple for NumPy arrays.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Array to index.
    axes : list(int)
        Dimensions along which to apply `index_spec`.
    index_spec : list(:py:class:`slice`)
        Index/slice to use.

    Returns
    -------
    indexer : tuple
        Indexing tuple.
    """
    assert len(axes) == len(index_spec)
    idx = [slice(None)] * x.ndim
    for i, ax in enumerate(axes):
        idx[ax] = index_spec[i]

    indexer = tuple(idx)
    return indexer


def _verify_axes(x, axes):
    axes = [a + x.ndim if a < 0 else a for a in axes]
    if any(a >= x.ndim or a < 0 for a in axes):
        raise ValueError("axes exceeds dimensionality of input")
    if len(set(axes)) != len(axes):
        raise ValueError("all axes must be unique")
    return axes


def _verify_cztn_input(x, A, W, M, axes):
    """
    Verify input values to :py:func:`~pyffs.czt.cztn`, and return axes values.

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
    axes : tuple
        Transform dimensions.
    A : complex
    W : complex
    """
    D = len(A)
    if not (len(W) == len(M) == D):
        raise ValueError("Length of [A], [W], and [M] must match.")

    if x.ndim < D:
        raise ValueError("[x] does not have enough dimensions.")

    if axes is not None:
        assert len(axes) == D, "Length of [axes] must match [A], [W], and [M]."
        axes = _verify_axes(x, axes)
    else:
        axes = list(range(D))

    # check valid values
    for d in range(D):
        A[d] = complex(A[d])
        W[d] = complex(W[d])
        if not cmath.isclose(abs(A[d]), 1):
            raise ValueError("Parameter[A[d]] must lie on the unit circle for numerical stability.")
        if not cmath.isclose(abs(W[d]), 1):
            raise ValueError("Parameter[W[d]] must lie on the unit circle.")
        if M[d] <= 0:
            raise ValueError("Parameter[M[d]] must be positive.")

    return axes, A, W


def _verify_fs_interp_input(x_FS, T, a, b, M, axes):
    """
    Verify input values to :py:func:`~pyffs.interp.fs_interpn`, and return axes
    values.

    Parameters
    ----------
    x_FS : :py:class:`~numpy.ndarray`
        (..., N_1, N_2, ..., N_D, ...) input values.
    T : list(float)
        Function period along each dimension.
    a : list(float)
        Interval LHS for each dimension.
    b : list(float)
        Interval RHS for each dimension.
    M : list(int)
        Number of points to interpolate for each dimension.
    axes : tuple
        Dimensions of `x` along which transform should be applied.

    Returns
    -------
    axes : tuple
        Indexing tuple.
    """
    D = len(T)
    if not (len(a) == len(b) == len(M) == D):
        raise ValueError("Length of [T], [a], [b], and [M] must match.")

    if x_FS.ndim < D:
        raise ValueError("[x_FS] does not have enough dimensions.")

    if axes is not None:
        assert len(axes) == D, "Length of [axes] must match [T], [a], [b], and [M]."
        axes = _verify_axes(x_FS, axes)
    else:
        axes = list(range(D))

    # check valid values
    for d in range(D):
        if T[d] <= 0:
            raise ValueError("Parameter[T[d]] must be positive.")
        if not (a[d] < b[d]):
            raise ValueError(f"Parameter[a[d]] must be smaller than Parameter[b[d]].")
        if M[d] <= 0:
            raise ValueError("Parameter[M[d]] must be positive.")

    return axes


def _verify_ffsn_input(x, T, T_c, N_FS, axes):
    """
    Verify input values to :py:func:`~pyffs.ffs.ffsn`,
    :py:func:`~pyffs.ffs.iffsn`, and return axes values.

    Parameters
    ----------
    x : :py: class: `~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) input values; either for FFSn or iFFSn.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Period mid-point for each dimension.
    N_FS : list(int)
        Function bandwidth along each dimension.
    axes : tuple
        Dimensions of `x` along which transform should be applied.

    Returns
    -------
    axes : tuple
        Indexing tuple.
    N_s : :py:class:`~numpy.ndarray`
        Number of samples per dimension.
    """
    D = len(T)
    if not (len(T_c) == len(N_FS) == D):
        raise ValueError("Length of [T], [T_c], and [N_FS] must match.")

    if x.ndim < D:
        raise ValueError("[x] does not have enough dimensions.")

    if axes is not None:
        assert len(axes) == D, "Length of [axes] must match [T], [T_c], and [N_FS]."
        axes = _verify_axes(x, axes)
    else:
        axes = list(range(D))

    N_s = np.array(x.shape)[axes]

    # check valid values
    for d in range(D):
        if T[d] <= 0:
            raise ValueError("Parameter[T[d]] must be positive.")
        if not (3 <= N_FS[d] <= N_s[d]):
            raise ValueError(f"Parameter[N_FS[d]] must lie in {{3, ..., N_s[d]}}.")

    return tuple(axes), N_s


def cartesian_product(x1, x2):
    """
    Return
    `Cartesian product <https://en.wikipedia.org/wiki/Cartesian_product>`_ of two arrays.

    Parameters
    ----------
    x1 : :py:class:`~numpy.ndarray`
        (M,) array.
    x2 : :py:class:`~numpy.ndarray`
        (N,) array.

    Returns
    -------
    y : :py:class:`~numpy.ndarray`
        (M, N, 2) array.
    """
    sh = len(x1), len(x2)
    y = np.stack(
        [np.broadcast_to(x1.reshape(-1, 1), sh), np.broadcast_to(x2.reshape(1, -1), sh)], axis=-1
    )
    return y


def ffs_sample(T, N_FS, T_c, N_s):
    r"""
    Signal sample positions for :py:func:`~pyffs.ffs.ffs`.

    Return the coordinates at which a signal must be sampled to use :py:func:`~pyffs.ffs`.

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
    idx : :py:class:`~numpy.ndarray`
        (N_s,) index array; could be used to reorder samples.

    Examples
    --------
    Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a bandlimited periodic function of period
    :math:`T = 1`, bandwidth :math:`N_{FS} = 5`, and with one period centered at :math:`T_{c} =
    \pi`. The sampling points :math:`t[n] \in \mathbb{R}` at which :math:`\phi` must be evaluated to
    compute the Fourier Series coefficients :math:`\left\{ \phi_{k}^{FS}, k = -2, \ldots, 2
    \right\}` with :py:func:`~pyffs.ffs` are obtained as follows:

    .. testsetup::

       import numpy as np

       from pyffs import ffs_sample

    .. doctest::

       # Ideally choose N_s to be highly-composite for ffs().
       >>> sample_points, idx = ffs_sample(T=1, N_FS=5, T_c=np.pi, N_s=8)
       >>> np.around(sample_points, 2)  # Notice points are not sorted.
       array([3.2 , 3.33, 3.45, 3.58, 2.7 , 2.83, 2.95, 3.08])
       >>> idx
       array([ 0,  1,  2,  3, -4, -3, -2, -1])


    See Also
    --------
    :py:func:`~pyffs.ffs.ffs`
    """
    if T <= 0:
        raise ValueError("Parameter[T] must be positive.")
    if N_FS < 3:
        raise ValueError("Parameter[N_FS] must be at least 3.")
    if N_s < N_FS:
        raise ValueError("Parameter[N_s] must be greater or equal to the signal bandwidth.")
    assert N_FS % 2 == 1, "Parameter[N_FS] must be odd."

    if N_s % 2 == 1:  # Odd-valued
        M = (N_s - 1) // 2
        idx = np.r_[0 : (M + 1), -M:0]
        sample_points = T_c + (T / N_s) * idx
    else:  # Even case
        M = N_s // 2
        idx = np.r_[0:M, -M:0]
        sample_points = T_c + (T / N_s) * (0.5 + idx)

    return sample_points, idx


def ffsn_sample(T, N_FS, T_c, N_s):
    r"""
    Signal sample positions for :py:func:`~pyffs.ffs.ffsn`.

    Return the coordinates at which a signal must be sampled to use :py:func:`~pyffs.ffs.ffsn`.

    Parameters
    ----------
    T : list(float)
        Function period along each dimension.
    N_FS : list(int)
        Function bandwidth along each dimension.
    T_c : list(float)
        Period mid-point for each dimension.
    N_s : list(int)
        Number of sample points for each dimension.

    Returns
    -------
    S0, ..., SD : list(:py:class:`~numpy.ndarray`)
        (N_D,) coordinates at which to sample a signal in the d-th dimension (in the right order).
    i1, ..., iD : list(:py:class:~numpy.ndarray)
        (N_D,) sample indices in the d-th dimension. May be useful to reorder samples.

    Examples
    --------
    Let :math:`\phi: \mathbb{R}^2 \to \mathbb{C}` be a bandlimited periodic function with periods
    :math:`T_x = 1` and :math:`T_y = 1`, bandwidths :math:`N_{FS,x} = 3` and :math:`N_{FS,y} = 3`,
    and with one period centered at :math:`(T_{c,x}, T_{c,y}) = (0, 0)`. The sampling points
    :math:`[x[m], y[n]] \in \mathbb{R}^2` at which :math:`\phi` must be evaluated to compute the
    Fourier Series coefficients :math:`\left\{ \phi_{k_x, k_y}^{FS}, k_x, k_y = -1, \ldots, 1
    \right\}` with :py:func:`~pyffs.ffs.ffsn` are obtained as follows:

    .. testsetup::

       from pyffs import ffsn_sample
       from numpy.testing import assert_array_equal

    .. doctest::

       # Ideally choose number of samples to be highly-composite for ffsn().
       >>> sample_points, idx = ffsn_sample(T=[1, 1], N_FS=[3, 3], T_c=[0, 0], N_s=[4, 3])
       >>> assert_array_equal(sample_points[0][:, 0], np.array([0.125, 0.375, -0.375, -0.125]))
       >>> assert_array_equal(sample_points[1][0, :], np.array([0, 1 / 3, -1 / 3]))
       >>> assert_array_equal(idx[0][:, 0], np.array([0, 1, -2, -1]))
       >>> assert_array_equal(idx[1][0, :], np.array([0, 1, -1]))

    See Also
    --------
    :py:func:`~pyffs.ffs.ffsn`
    """
    D = len(T)
    assert len(N_FS) == D, "Length of [N_FS] must match that of [T]."
    assert len(T_c) == D, "Length of [T_c] must match that of [T]."
    assert len(N_s) == D, "Length of [N_s] must match that of [T]."

    # loop over dimensions
    sample_points = []
    idx = []
    for d in range(D):
        # get values for d-dimension
        _sample_points, _idx = ffs_sample(T=T[d], N_FS=N_FS[d], T_c=T_c[d], N_s=N_s[d])

        # reshape for sparse array
        sh = [1] * D
        sh[d] = N_s[d]
        sample_points.append(_sample_points.reshape(sh))
        idx.append(_idx.reshape(sh))

    return sample_points, idx


def _create_modulation_vectors(N_s, N_FS, T, T_c):
    """
    Compute modulation vectors for FFS.

    Parameters
    ----------
    N_s : int
        Number of samples.
    N_FS : int
        Function bandwidth.
    T : float
        Function period.
    T_c : float
        Period mid-point.

    Returns
    -------
    A : :py:class:`~numpy.ndarray`
    B : :py:class:`~numpy.ndarray`

    See Also
    --------
    :py:func:`~pyffs.ffs.ffs`, :py:func:`~pyffs.ffs.iffs`,
    :py:func:`~pyffs.ffs.ffsn`, :py:func:`~pyffs.ffs.iffsn`
    """
    M, N = np.r_[N_s, N_FS] // 2
    E_1 = np.r_[-N : (N + 1), np.zeros((N_s - N_FS,), dtype=int)]
    B_2 = np.exp(-1j * 2 * np.pi / N_s)

    if N_s % 2 == 1:
        B_1 = np.exp(1j * (2 * np.pi / T) * T_c)
        E_2 = np.r_[0 : (M + 1), -M:0]
    else:
        B_1 = np.exp(1j * (2 * np.pi / T) * (T_c + T / (2 * N_s)))
        E_2 = np.r_[0:M, -M:0]

    return B_1 ** E_1, B_2 ** (N * E_2)
