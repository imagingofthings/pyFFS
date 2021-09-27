# #############################################################################
# __init__.py
# ===========
# Authors :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Fast Fourier Series library
===========================

Efficient computation of Fourier Series (FS) coefficients and interpolation with
them through use of the Fast Fourier Transform (FFT).
"""

from .czt import *
from .ffs import *
from .interp import *
from .util import *
from . import func
from .conv import *

from .backend import *
