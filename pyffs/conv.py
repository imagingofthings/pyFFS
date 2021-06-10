# #############################################################################
# conv.py
# ========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import numpy as np
from pyffs.ffs import ffs, iffs, ffsn, iffsn
from pyffs.util import ffs_sample, ffsn_sample, _verify_ffsn_input, ffsn_shift


def convolve(f, h, T, T_c, N_FS, return_coef=False, reorder=True, axis=-1):
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
    reorder : bool
        Whether samples need to be reordered with `ffs_sample`.
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
    # reorder samples
    idx = None
    if reorder:
        _, idx = ffs_sample(T, N_FS, T_c, f.shape[axis])
        f = f[idx]
        # same for h, as we have same FFS parameters
        h = h[idx]

    F = ffs(f, T, T_c, N_FS, axis=axis)
    H = ffs(h, T, T_c, N_FS, axis=axis)
    if return_coef:
        return F * H
    else:
        output_samples = iffs(F * H, T=T, T_c=T_c, N_FS=N_FS, axis=axis)
        if reorder:
            output_samples[:] = output_samples[np.argsort(idx)]
        return output_samples


def convolve2d(f, h, T, T_c, N_FS, return_coef=False, reorder=True, axes=None):
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
    reorder : bool
        Whether samples need to be reordered with `ffsn_sample`.
    axes : tuple
        Dimensions of `f` and `h` along which function samples are stored.


    Returns
    -------
    g : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) vectors containing convolution between `f` and `h` at the same
        sampling points.

    """
    axes, N_s = _verify_ffsn_input(f, T, T_c, N_FS, axes)

    # reorder samples
    idx = None
    if reorder:
        _, idx = ffsn_sample(T, N_FS, T_c, N_s)
        f = ffsn_shift(f, idx)
        # same for h, as we have same FFS parameters
        h = ffsn_shift(h, idx)

    F = ffsn(f, T, T_c, N_FS, axes=axes)
    H = ffsn(h, T, T_c, N_FS, axes=axes)
    if return_coef:
        return F * H
    else:
        output_samples = iffsn(F * H, T=T, T_c=T_c, N_FS=N_FS, axes=axes)
        if reorder:
            output_samples = ffsn_shift(output_samples, idx)
        return output_samples
