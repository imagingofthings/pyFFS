# #############################################################################
# test_czt.py
# ===========
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import numpy as np
from pyffs import czt, cztn, _cztn


class TestCZT:
    """
    Test :py:module:`~pyffs.czt`.
    """

    def test_czt_dft(self):
        N = M = 10
        x = np.random.randn(N, 3) + 1j * np.random.randn(N, 3)
        dft_x = np.fft.fft(x, axis=0)
        czt_x = czt(x, A=1, W=np.exp(-1j * 2 * np.pi / N), M=M, axis=0)
        assert np.allclose(dft_x, czt_x)

    def test_czt_idft(self):
        N = M = 10
        x = np.random.randn(N) + 1j * np.random.randn(N)
        idft_x = np.fft.ifft(x)
        czt_x = czt(x, A=1, W=np.exp(1j * 2 * np.pi / N), M=M)
        assert np.allclose(idft_x, czt_x / N)  # czt() does not do the scaling.

    def test_cztn(self):
        N = M = 10
        W = np.exp(-1j * 2 * np.pi / N)
        x = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)  # extra dimension
        dft_x = np.fft.fftn(x, axes=(1, 2))
        czt_x = cztn(x, A=[1, 1], W=[W, W], M=[M, M], axes=(1, 2))
        assert np.allclose(dft_x, czt_x)

    def test_cztn_ref(self):
        N = M = 10
        W = np.exp(-1j * 2 * np.pi / N)
        x = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        dft_x = np.fft.fft2(x)
        czt_x = _cztn(x, A=[1, 1], W=[W, W], M=[M, M])
        assert np.allclose(dft_x, czt_x)
