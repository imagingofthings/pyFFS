import numpy as np
from numpy.testing import assert_array_equal
from pyffs import ffs_sample, ffs2_sample


def test_ffs_sample():
    sample_points, idx = ffs_sample(T=1, N_FS=5, T_c=np.pi, N_s=8)
    assert_array_equal(
        np.around(sample_points, 2),
        np.array([3.2, 3.33, 3.45, 3.58, 2.7, 2.83, 2.95, 3.08]),
    )
    assert_array_equal(
        idx,
        np.array([0, 1, 2, 3, -4, -3, -2, -1]),
    )


def test_ffs2_sample():
    Tx = Ty = 1
    N_FSx = N_FSy = 3
    T_cx = T_cy = 0
    N_sx = 4
    N_sy = 3
    sample_points, idx = ffs2_sample(
        Tx=Tx,
        Ty=Ty,
        N_FSx=N_FSx,
        N_FSy=N_FSy,
        T_cx=T_cx,
        T_cy=T_cy,
        N_sx=N_sx,
        N_sy=N_sy,
    )

    # check sample points
    assert sample_points[0].shape == (N_sx, 1)
    assert sample_points[1].shape == (1, N_sy)
    assert_array_equal(sample_points[0][:, 0], np.array([0.125, 0.375, -0.375, -0.125]))
    assert_array_equal(sample_points[1][0, :], np.array([0, 1 / 3, -1 / 3]))

    # check index values
    assert idx[0].shape == (N_sx, 1)
    assert idx[1].shape == (1, N_sy)
    assert_array_equal(idx[0][:, 0], np.array([0, 1, -2, -1]))
    assert_array_equal(idx[1][0, :], np.array([0, 1, -1]))


if __name__ == "__main__":

    test_ffs_sample()
    test_ffs2_sample()
