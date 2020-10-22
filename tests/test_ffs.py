import numpy as np
import math
from pyffs import ffs, ffs_sample, ffs2, ffs2_sample, iffs, iffs2


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
    x_vals = dirichlet(x=np.squeeze(sample_points[0]), T=Tx, T_c=T_cx, N_FS=N_FSx)
    y_vals = dirichlet(x=np.squeeze(sample_points[1]), T=Ty, T_c=T_cy, N_FS=N_FSy)
    return np.outer(x_vals, y_vals)


def test_ffs():
    T, T_c, N_FS = math.pi, math.e, 15
    N_samples = 16  # Any >= N_FS will do, but highly-composite best.

    # Sample the kernel and do the transform.
    sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
    diric_samples = dirichlet(sample_points, T, T_c, N_FS)
    diric_FS = ffs(diric_samples, T, T_c, N_FS)

    # Compare with theoretical result.
    assert np.allclose(diric_FS[:N_FS], dirichlet_fs(N_FS, T, T_c))

    # Inverse transform.
    diric_samples_recov = iffs(diric_FS, T, T_c, N_FS)

    # Compare with original samples.
    assert np.allclose(diric_samples, diric_samples_recov)


def test_ffs2():
    Tx = Ty = 1
    T_cx = T_cy = 0
    N_FSx = N_FSy = 3
    N_sx = 4
    N_sy = 3

    # Sample the kernel and do the transform.
    sample_points, _ = ffs2_sample(
        Tx=Tx,
        Ty=Ty,
        N_FSx=N_FSx,
        N_FSy=N_FSy,
        T_cx=T_cx,
        T_cy=T_cy,
        N_sx=N_sx,
        N_sy=N_sy,
    )
    diric_samples = dirichlet_2D(
        sample_points=sample_points,
        Tx=Tx,
        Ty=Ty,
        T_cx=T_cx,
        T_cy=T_cy,
        N_FSx=N_FSx,
        N_FSy=N_FSy,
    )
    diric_FS = ffs2(
        Phi=diric_samples,
        Tx=Tx,
        Ty=Ty,
        T_cx=T_cx,
        T_cy=T_cy,
        N_FSx=N_FSx,
        N_FSy=N_FSy,
    )

    # Compare with theoretical result.
    diric_FS_exact = np.outer(dirichlet_fs(N_FSx, Tx, T_cx), dirichlet_fs(N_FSy, Ty, T_cy))
    assert np.allclose(diric_FS[:N_FSx, :N_FSy], diric_FS_exact)

    # Inverse transform.
    diric_samples_recov = iffs2(
        Phi_FS=diric_FS,
        Tx=Tx,
        Ty=Ty,
        T_cx=T_cx,
        T_cy=T_cy,
        N_FSx=N_FSx,
        N_FSy=N_FSy,
    )

    # Compare with original samples.
    assert np.allclose(diric_samples, diric_samples_recov)


if __name__ == "__main__":

    test_ffs()
    test_ffs2()
