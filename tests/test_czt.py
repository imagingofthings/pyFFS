import numpy as np
from pyffs import czt, cztn
from pyffs.util import _verify_cztn_input


def cztn_comp(x, A, W, M, axes=None):
    axes, A, W = _verify_cztn_input(x, A, W, M, axes)

    # sequence of 1D FFS
    x_czt = x.copy()
    for d, ax in enumerate(axes):
        x_czt = czt(x_czt, A[d], W[d], M[d], axis=ax)

    return x_czt


def test_czt_dft():
    N = M = 10
    x = np.random.randn(N, 3) + 1j * np.random.randn(N, 3)
    dft_x = np.fft.fft(x, axis=0)
    czt_x = czt(x, A=1, W=np.exp(-1j * 2 * np.pi / N), M=M, axis=0)
    assert np.allclose(dft_x, czt_x)


def test_czt_idft():
    N = M = 10
    x = np.random.randn(N) + 1j * np.random.randn(N)
    idft_x = np.fft.ifft(x)
    czt_x = czt(x, A=1, W=np.exp(1j * 2 * np.pi / N), M=M)
    assert np.allclose(idft_x, czt_x / N)  # czt() does not do the scaling.


def test_cztn():
    N = M = 10
    W = np.exp(-1j * 2 * np.pi / N)
    x = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)  # extra dimension
    dft_x = np.fft.fftn(x, axes=(1, 2))
    czt_x = cztn(x, A=[1, 1], W=[W, W], M=[M, M], axes=(1, 2))
    assert np.allclose(dft_x, czt_x)


def test_cztn_comp():
    N = M = 10
    W = np.exp(-1j * 2 * np.pi / N)
    x = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    dft_x = np.fft.fft2(x)
    czt_x = cztn_comp(x, A=[1, 1], W=[W, W], M=[M, M])
    assert np.allclose(dft_x, czt_x)


if __name__ == "__main__":

    test_czt_dft()
    test_czt_idft()
    test_cztn()
    test_cztn_comp()
