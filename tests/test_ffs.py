import math

import numpy as np

from pyffs import (
    ffs,
    ffs_sample,
    ffsn,
    ffsn_sample,
    iffs,
    iffsn,
)
from pyffs.func import dirichlet, dirichlet_fs, dirichlet_2D
from pyffs.util import _verify_ffsn_input


def ffsn_comp(Phi, T, T_c, N_FS, axes=None):
    axes, _ = _verify_ffsn_input(Phi, T, T_c, N_FS, axes)

    # sequence of 1D FFS
    Phi_FS = Phi.copy()
    for d, ax in enumerate(axes):
        Phi_FS = ffs(Phi_FS, T[d], T_c[d], N_FS[d], axis=ax)

    return Phi_FS


def iffsn_comp(Phi_FS, T, T_c, N_FS, axes=None):
    axes, _ = _verify_ffsn_input(Phi_FS, T, T_c, N_FS, axes)

    # sequence of 1D iFFS
    Phi = Phi_FS.copy()
    for d, ax in enumerate(axes):
        Phi = iffs(Phi, T[d], T_c[d], N_FS[d], axis=ax)

    return Phi


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


def test_ffsn_axes():
    T_x = T_y = 1
    T_cx = T_cy = 0
    N_FSx = N_FSy = 3
    N_sx, N_sy = 4, 3

    # Sample the kernel.
    sample_points, _ = ffsn_sample(
        T=[T_x, T_y], N_FS=[N_FSx, N_FSy], T_c=[T_cx, T_cy], N_s=[N_sx, N_sy]
    )
    diric_samples = dirichlet_2D(
        sample_points=sample_points, T=[T_x, T_y], T_c=[T_cx, T_cy], N_FS=[N_FSx, N_FSy]
    )

    # Add new dimension.
    diric_samples = diric_samples[:, np.newaxis, :]
    axes = (0, 2)

    # Perform transform.
    diric_FS = ffsn(
        Phi=diric_samples, T=[T_x, T_y], T_c=[T_cx, T_cy], N_FS=[N_FSx, N_FSy], axes=axes
    )

    # Compare with theoretical result.
    diric_FS_exact = np.outer(dirichlet_fs(N_FSx, T_x, T_cx), dirichlet_fs(N_FSy, T_y, T_cy))
    assert np.allclose(diric_FS[:N_FSx, 0, :N_FSy], diric_FS_exact)

    # Inverse transform.
    diric_samples_recov = iffsn(
        Phi_FS=diric_FS, T=[T_x, T_y], T_c=[T_cx, T_cy], N_FS=[N_FSx, N_FSy], axes=axes
    )

    # Compare with original samples.
    assert np.allclose(diric_samples, diric_samples_recov)


def test_ffsn_comp():
    T = [1, 1]
    T_c = [0, 0]
    N_FS = [3, 3]
    N_s = [4, 3]

    # Sample the kernel and do the transform.
    sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
    diric_samples = dirichlet_2D(sample_points=sample_points, T=T, T_c=T_c, N_FS=N_FS)
    diric_FS = ffsn_comp(Phi=diric_samples, T=T, N_FS=N_FS, T_c=T_c)

    # Compare with theoretical result.
    diric_FS_exact = np.outer(
        dirichlet_fs(N_FS[0], T[0], T_c[0]), dirichlet_fs(N_FS[1], T[1], T_c[1])
    )
    assert np.allclose(diric_FS[: N_FS[0], : N_FS[1]], diric_FS_exact)

    # Inverse transform.
    diric_samples_recov = iffsn_comp(Phi_FS=diric_FS, T=T, T_c=T_c, N_FS=N_FS)

    # Compare with original samples.
    assert np.allclose(diric_samples, diric_samples_recov)


def test_ffsn():
    T = [1, 1]
    T_c = [0, 0]
    N_FS = [3, 3]
    N_s = [4, 3]

    # Sample the kernel and do the transform.
    sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
    diric_samples = dirichlet_2D(sample_points=sample_points, T=T, T_c=T_c, N_FS=N_FS)
    diric_FS = ffsn(Phi=diric_samples, T=T, N_FS=N_FS, T_c=T_c)

    # Compare with theoretical result.
    diric_FS_exact = np.outer(
        dirichlet_fs(N_FS[0], T[0], T_c[0]), dirichlet_fs(N_FS[1], T[1], T_c[1])
    )
    assert np.allclose(diric_FS[: N_FS[0], : N_FS[1]], diric_FS_exact)

    # Inverse transform.
    diric_samples_recov = iffsn(Phi_FS=diric_FS, T=T, T_c=T_c, N_FS=N_FS)

    # Compare with original samples.
    assert np.allclose(diric_samples, diric_samples_recov)


if __name__ == "__main__":
    test_ffs()
    test_ffsn()
    test_ffsn_axes()
    test_ffsn_comp()
