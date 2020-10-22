# #############################################################################
# __init__.py
# ===========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Fast Fourier Series library.
============================

Efficient computation of Fourier Series (FS) coefficients and interpolation
with them through use of the Fast Fourier Transformation (FFT).

Available submodules
--------------------
:py:obj:`pyfft.czt`
    Methods for computing chirp Z-transform.

:py:obj:`pyfft.ffs`
    Methods for computing Fast Fourier Series.

:py:obj:`pyfft.interpolation`
    Methods for interpolating with Fourier Series coefficients.

:py:obj:`pyfft.util`
    Utility functions.

"""

from .czt import *
from .ffs import *
from .interpolation import *
from .util import *
