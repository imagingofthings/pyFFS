# #############################################################################
# conv.py
# ========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import numpy as np
from pyffs.ffs import ffsn, iffsn
from pyffs.util import ffsn_sample, _verify_ffsn_input, ffsn_shift


def convolve(f, h, T, T_c, N_FS, return_coef=False, reorder=True, axes=None):
    """
    Convolve two N-dimensional functions using FFS on their discrete samples.

    The Fourier Series coefficients of both functions are estimated via FFS and multiplied in order
    to perform the convolution.

    The two functions must have the same period, period center, and number of Fourier Series
    coefficients. If a scalar is passed for each these, 1-D is assumed and the convolution is
    performed on the last axis.

    Parameters
    ----------
    f : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffsn_sample`.
    h : :py:class:`~numpy.ndarray`
        (..., N_s1, N_s2, ..., N_sD, ...) function values at sampling points specified by
        :py:func:`~pyffs.util.ffsn_sample`.
    T : int or array_like of floats
        Function period along each dimension.
    T_c : int or array_like of floats
        Period mid-point for each dimension.
    N_FS : int or array_like of ints
        Function bandwidth along each dimension.
    return_coef : bool
        Whether to return coefficients or samples.
    reorder : bool
        Whether samples need to be reordered with `ffsn_sample`.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution. The default is over all axes.


    Returns
    -------
    g : :py:class:`~numpy.ndarray`
        (..., N_sx, N_sy, ...) vectors containing convolution between `f` and `h` at the same
        sampling points.

    """

    if (
        not isinstance(T, (list, tuple, np.ndarray))
        or not isinstance(T_c, (list, tuple, np.ndarray))
        or isinstance(N_FS, int)
        or isinstance(axes, int)
    ):
        assert not isinstance(T, (list, tuple, np.ndarray))
        assert not isinstance(T_c, (list, tuple, np.ndarray))
        assert isinstance(N_FS, int)
        if axes is None:
            axes = (-1,)
        else:
            assert isinstance(axes, int)
        T = [T]
        T_c = [T_c]
        N_FS = [N_FS]
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
