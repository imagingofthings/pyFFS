# #############################################################################
# test_util.py
# ============
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import numpy as np
from numpy.testing import assert_array_equal

from pyffs import ffs_sample, ffsn_sample
from pyffs.backend import AVAILABLE_MOD, get_array_module


class TestUtil:
    """
    Test :py:module:`~pyffs.util`.
    """

    def test_ffs_sample(self):
        for mod in AVAILABLE_MOD:
            sample_points, idx = ffs_sample(T=1, N_FS=5, T_c=np.pi, N_s=8, mod=mod)
            assert_array_equal(
                mod.around(sample_points, 2),
                mod.array([3.2, 3.33, 3.45, 3.58, 2.7, 2.83, 2.95, 3.08]),
            )
            assert_array_equal(
                idx,
                mod.array([4, 5, 6, 7, 0, 1, 2, 3]),
            )
            assert get_array_module(sample_points) == mod
            assert get_array_module(idx) == mod

    def test_ffsn_sample(self):
        N_s = [4, 3]
        for mod in AVAILABLE_MOD:
            sample_points, idx = ffsn_sample(T=[1, 1], N_FS=[3, 3], T_c=[0, 0], N_s=N_s, mod=mod)

            # check sample points
            assert sample_points[0].shape == (N_s[0], 1)
            assert sample_points[1].shape == (1, N_s[1])
            assert_array_equal(sample_points[0][:, 0], mod.array([0.125, 0.375, -0.375, -0.125]))
            assert_array_equal(sample_points[1][0, :], mod.array([0, 1 / 3, -1 / 3]))

            # check index values
            assert idx[0].shape == (N_s[0], 1)
            assert idx[1].shape == (1, N_s[1])
            assert_array_equal(idx[0][:, 0], mod.array([2, 3, 0, 1]))
            assert_array_equal(idx[1][0, :], mod.array([1, 2, 0]))

            assert get_array_module(sample_points) == mod
            assert get_array_module(idx) == mod

    def test_ffsn_sample_shape(self):
        D = 5
        T = np.ones(D)
        N_FS = np.arange(D) * 2 + 3
        T_c = np.zeros(D)
        N_s = N_FS

        for mod in AVAILABLE_MOD:

            sample_points, idx = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s, mod=mod)

            # check shape
            for d in range(D):
                sh = [1] * D
                sh[d] = N_s[d]
                assert list(sample_points[d].shape) == sh
                assert list(idx[d].shape) == sh

            assert get_array_module(sample_points) == mod
            assert get_array_module(idx) == mod
