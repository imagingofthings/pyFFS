# #############################################################################
# backend.py
# ==========
# Authors :
# Eric Bezzam [ebezzam@gmail.com]
# #############################################################################

"""
Methods for identifying the appropriate backend between `numpy` and `cupy`.

Code is heavily inspired by `PyLops` library: https://github.com/PyLops/pylops/blob/master/pylops/utils/backend.py
"""

from importlib import util

import os

import numpy as np
import scipy.fft


CUPY_ENABLED = (util.find_spec("cupy") is not None) and (int(os.getenv("CUPY_PYFFS", 1)) == 1)
AVAILABLE_MOD = [np]
if CUPY_ENABLED:
    try:
        import cupy as cp
        import cupyx
        AVAILABLE_MOD.append(cp)
    except ImportError:
        # CuPy is installed, but GPU drivers probably missing.
        CUPY_ENABLED = False


def get_module(backend="numpy"):
    """
    Returns correct numerical module based on backend string.

    Parameters
    ----------
    backend : :obj:`str`, optional
        Backend used for computations (``numpy`` or ``cupy``).

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process arrays (:mod:`numpy` or :mod:`cupy`).
    """
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        xp = cp
    else:
        raise ValueError(f"Unsupported backend '{backend}': choose amongst {{numpy,cupy}}.")
    return xp


def get_module_name(mod):
    """
    Returns name of numerical module based on input numerical module.

    Parameters
    ----------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`).

    Returns
    -------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``).
    """
    if mod == np:
        backend = "numpy"
    elif mod == cp:
        backend = "cupy"
    else:
        raise ValueError(f"Unsupported module '{mod}': choose amongst {{numpy, cupy}}.")
    return backend


def get_backend():
    if CUPY_ENABLED:
        return cp
    else:
        return np


def get_array_module(x):
    """
    Returns correct numerical module based on input.

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)
    """
    if CUPY_ENABLED:
        return cp.get_array_module(x)
    else:
        return np


def fftn(x, axes=None):
    """
    Applies correct fftn method based on input.

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array
    axes : tuple
        Dimension of `x` along which to perform FFT.

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)
    """
    if get_array_module(x) == np:
        func = scipy.fft.fftn
    else:
        func = cupyx.scipy.fft.fftn
    return func(x, axes=axes)


def fft(x):
    """
    Applies correct fft method based on input.

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)
    """
    if get_array_module(x) == np:
        func = scipy.fft.fft
    else:
        func = cupyx.scipy.fft.fft
    return func(x)


def ifftn(x, axes=None):
    """
    Apply correct ifftn method based on input.

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array
    axes : tuple
        Dimension of `x` along which to perform IFFT.

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)
    """
    if get_array_module(x) == np:
        func = scipy.fft.ifftn
    else:
        func = cupyx.scipy.fft.ifftn
    return func(x, axes=axes)


def next_fast_len(target, mod=None):
    """
    Apply correct next_fast_len method based on backend.

    Parameters
    ----------
    target : int
        Desired FFT length.
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`).

    Returns
    -------
    next_fast_len : int
        The smallest fast length greater than or equal to `target`.
    """
    if mod is None:
        mod = get_backend()

    if mod == np:
        func = scipy.fft.next_fast_len
    else:
        func = cupyx.scipy.fft.next_fast_len
    return func(target)
