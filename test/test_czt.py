# #############################################################################
# test_czt.py
# ===========
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

from pyffs import czt, cztn, _cztn
from pyffs.backend import AVAILABLE_MOD


class TestCZT:
    """
    Test :py:module:`~pyffs.czt`.
    """

    def test_czt_dft(self):
        N = M = 10
        for mod in AVAILABLE_MOD:
            x = mod.random.randn(N, 3) + 1j * mod.random.randn(N, 3)
            dft_x = mod.fft.fft(x, axis=0)
            czt_x = czt(x, A=1, W=mod.exp(-1j * 2 * mod.pi / N), M=M, axis=0)
            assert mod.allclose(dft_x, czt_x)

    def test_czt_idft(self):
        N = M = 10
        for mod in AVAILABLE_MOD:
            x = mod.random.randn(N) + 1j * mod.random.randn(N)
            idft_x = mod.fft.ifft(x)
            czt_x = czt(x, A=1, W=mod.exp(1j * 2 * mod.pi / N), M=M)
            assert mod.allclose(idft_x, czt_x / N)  # czt() does not do the scaling.

    def test_cztn(self):
        N = M = 10
        for mod in AVAILABLE_MOD:
            W = mod.exp(-1j * 2 * mod.pi / N)
            x = mod.random.randn(N, N, N) + 1j * mod.random.randn(N, N, N)  # extra dimension
            dft_x = mod.fft.fftn(x, axes=(1, 2))
            czt_x = cztn(x, A=[1, 1], W=[W, W], M=[M, M], axes=(1, 2))
            assert mod.allclose(dft_x, czt_x)

    def test_cztn_ref(self):
        N = M = 10
        for mod in AVAILABLE_MOD:
            W = mod.exp(-1j * 2 * mod.pi / N)
            x = mod.random.randn(N, N) + 1j * mod.random.randn(N, N)
            dft_x = mod.fft.fft2(x)
            czt_x = _cztn(x, A=[1, 1], W=[W, W], M=[M, M])
            assert mod.allclose(dft_x, czt_x)
