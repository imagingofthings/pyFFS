import numpy as np
from pyffs import czt, czt2, cztn, cztn_comp


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


def test_czt2():
    N = M = 10
    W = np.exp(-1j * 2 * np.pi / N)
    Phi = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    dft_Phi = np.fft.fft2(Phi)
    czt_Phi = czt2(Phi, Ax=1, Ay=1, Wx=W, Wy=W, Mx=M, My=M)
    assert np.allclose(dft_Phi, czt_Phi)


def test_cztn():
    N = M = 10
    W = np.exp(-1j * 2 * np.pi / N)
    Phi = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)  # extra dimension
    dft_Phi = np.fft.fftn(Phi, axes=(1, 2))
    czt_Phi = cztn(Phi, A=[1, 1], W=[W, W], M=[M, M], axes=(1, 2))
    assert np.allclose(dft_Phi, czt_Phi)


def test_cztn_comp():
    N = M = 10
    W = np.exp(-1j * 2 * np.pi / N)
    Phi = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    dft_Phi = np.fft.fft2(Phi)
    czt_Phi = cztn_comp(Phi, A=[1, 1], W=[W, W], M=[M, M])
    assert np.allclose(dft_Phi, czt_Phi)


if __name__ == "__main__":

    test_czt_dft()
    test_czt_idft()
    test_czt2()
    test_cztn()
    test_cztn_comp()
