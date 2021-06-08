# #############################################################################
# conv.py
# ========
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


from pyffs.ffs import ffs, iffs, ffsn, iffsn


def convolve(f, h, T, T_c, N_FS, return_coef=False, axis=-1):
    """
    Convolve two functions via a multiplication of their Fourier Series coefficients estimated from
    discrete samples. The two functions must have the same period, period center, and number of
    Fourier Series coefficients.

    Parameters
    ----------
    f : :py:class:`~numpy.ndarray`
        (..., N_s, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffs_sample`.
    h : :py:class:`~numpy.ndarray`
        (..., N_s, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffs_sample`.
    T : float
        Function period.
    T_c : float
        Period mid-point.
    N_FS : int
        Function bandwidth.
    return_coef : bool
        Whether to return coefficients or samples.
    axis : int
        Dimension of `f` and `h` along which function samples are stored.


    Returns
    -------
    g : :py:class:`~numpy.ndarray`
        (..., N_s, ...) vectors containing convolution between `f` and `h` at the same sampling
        points.

    """
    F = ffs(f, T, T_c, N_FS, axis=axis)
    H = ffs(h, T, T_c, N_FS, axis=axis)
    if return_coef:
        return F * H
    else:
        return iffs(F * H, T=T, T_c=T_c, N_FS=N_FS, axis=axis)


def convolve2d(f, h, T, T_c, N_FS, return_coef=False, axes=None):
    """
    Convolve two 2D functions via a multiplication of their Fourier Series coefficients estimated
    from discrete samples. The two functions must have the same period, period center, and number of
    Fourier Series coefficients.

    Parameters
    ----------
    f : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffsn_sample`.
    h : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffsn_sample`.
    T : list(float)
        Function period along each dimension.
    T_c : list(float)
        Period mid-point for each dimension.
    N_FS : list(int)
        Function bandwidth along each dimension.
    return_coef : bool
        Whether to return coefficients or samples.
    axes : tuple
        Dimensions of `f` and `h` along which function samples are stored.


    Returns
    -------
    g : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) vectors containing convolution between `f` and `h` at the same
        sampling points.

    """
    F = ffsn(f, T, T_c, N_FS, axes=axes)
    H = ffsn(h, T, T_c, N_FS, axes=axes)
    if return_coef:
        return F * H
    else:
        return iffsn(F * H, T=T, T_c=T_c, N_FS=N_FS, axes=axes)
