import math
import numpy as np
from pyffs.interp import fs_interp, fs_interp2


def dirichlet(x, T, T_c, N_FS):
    y = x - T_c

    n, d = np.zeros((2, len(x)))
    nan_mask = np.isclose(np.fmod(y, np.pi), 0)
    n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
    d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
    n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
    d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

    return n / d


def dirichlet_fs(N_FS, T, T_c):
    N = (N_FS - 1) // 2
    return np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N : N + 1])


def dirichlet_2D(sample_points, Tx, Ty, T_cx, T_cy, N_FSx, N_FSy):
    # compute along x and y, then combine
    x_vals = dirichlet(x=sample_points[0][:, 0], T=Tx, T_c=T_cx, N_FS=N_FSx)
    y_vals = dirichlet(x=sample_points[1][0, :], T=Ty, T_c=T_cy, N_FS=N_FSy)
    return np.outer(x_vals, y_vals)


def test_fs_interp():

    # parameters of signal
    T, T_c, N_FS = math.pi, math.e, 15

    # And the kernel's FS coefficients.
    diric_FS = dirichlet_fs(N_FS, T, T_c)

    # Generate interpolated signal
    a, b = T_c + (T / 2) * np.r_[-1, 1]
    M = 100  # We want lots of points.
    diric_sig = fs_interp(diric_FS, T, a, b, M)

    # Compare with theoretical result.
    t = a + (b - a) / (M - 1) * np.arange(M)
    diric_sig_exact = dirichlet(t, T, T_c, N_FS)
    assert np.allclose(diric_sig, diric_sig_exact)

    # Try real version
    diric_sig_real = fs_interp(diric_FS, T, a, b, M, real_x=True)
    assert np.allclose(diric_sig, diric_sig_real)


def test_fs_interp2():

    # parameters of signal
    T_x = np.pi
    T_y = np.pi
    N_FSx = 5
    N_FSy = 5
    T_cx = math.e
    T_cy = math.e

    # And the kernel's FS coefficients.
    diric_FS = np.outer(dirichlet_fs(N_FSx, T_x, T_cx), dirichlet_fs(N_FSy, T_y, T_cy))

    # Generate interpolated signal
    a_x, b_x = T_cx + (T_x / 2) * np.r_[-1, 1]
    a_y, b_y = T_cy + (T_y / 2) * np.r_[-1, 1]
    M_x = 6
    M_y = 6
    diric_sig = fs_interp2(diric_FS, T_x, T_y, a_x, a_y, b_x, b_y, M_x, M_y)

    # Compare with theoretical result.
    sample_points_x = a_x + (b_x - a_x) / (M_x - 1) * np.arange(M_x)
    sample_points_y = a_y + (b_y - a_y) / (M_y - 1) * np.arange(M_y)
    sample_points = [sample_points_x.reshape(M_x, 1), sample_points_y.reshape(1, M_y)]
    diric_sig_exact = dirichlet_2D(sample_points, T_x, T_y, T_cx, T_cy, N_FSx, N_FSy)
    assert np.allclose(diric_sig, diric_sig_exact)

    # # Try real version
    # diric_sig_real = fs_interp2(diric_FS, T_x, T_y, a_x, a_y, b_x, b_y, M_x, M_y, real_Phi=True)
    # assert np.allclose(diric_sig, diric_sig_real)


if __name__ == "__main__":

    test_fs_interp()
    test_fs_interp2()
