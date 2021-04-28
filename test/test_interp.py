# #############################################################################
# test_interp.py
# ==============
# Author :
# Sepand KASHANI [kashani.sepand@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import math
from pyffs.func import dirichlet, dirichlet_fs, dirichlet_2D
from pyffs.interp import fs_interp, fs_interpn
from pyffs.backend import AVAILABLE_MOD


class TestInterp:
    """
    Test :py:module:`~pyffs.interp`.
    """

    def test_fs_interp(self):
        # parameters of signal
        T, T_c, N_FS = math.pi, math.e, 15

        for mod in AVAILABLE_MOD:

            # And the kernel's FS coefficients.
            diric_FS = dirichlet_fs(N_FS, T, T_c, mod=mod)

            # Generate interpolated signal
            a, b = T_c + (T / 2) * mod.r_[-1, 1]
            M = 100  # We want lots of points.
            diric_sig = fs_interp(diric_FS, T, a, b, M)

            # Compare with theoretical result.
            t = a + (b - a) / (M - 1) * mod.arange(M)
            diric_sig_exact = dirichlet(t, T, T_c, N_FS)
            assert mod.allclose(diric_sig, diric_sig_exact)

            # Try real version
            diric_sig_real = fs_interp(diric_FS, T, a, b, M, real_x=True)
            assert mod.allclose(diric_sig, diric_sig_real)

    def test_fs_interp2(self):
        for mod in AVAILABLE_MOD:
            # parameters of signal
            T_x = T_y = mod.pi
            N_FSx = N_FSy = 5
            T_cx = T_cy = math.e

            # And the kernel's FS coefficients.
            diric_FS = mod.outer(
                dirichlet_fs(N_FSx, T_x, T_cx, mod=mod), dirichlet_fs(N_FSy, T_y, T_cy, mod=mod)
            )

            # Generate interpolated signal
            a_x, b_x = T_cx + (T_x / 2) * mod.r_[-1, 1]
            a_y, b_y = T_cy + (T_y / 2) * mod.r_[-1, 1]
            M_x = M_y = 6
            diric_sig = fs_interpn(diric_FS, T=[T_x, T_y], a=[a_x, a_y], b=[b_x, b_y], M=[M_x, M_y])

            # Compare with theoretical result.
            sample_points_x = a_x + (b_x - a_x) / (M_x - 1) * mod.arange(M_x)
            sample_points_y = a_y + (b_y - a_y) / (M_y - 1) * mod.arange(M_y)
            sample_points = [sample_points_x.reshape(M_x, 1), sample_points_y.reshape(1, M_y)]
            diric_sig_exact = dirichlet_2D(
                sample_points, T=[T_x, T_y], T_c=[T_cx, T_cy], N_FS=[N_FSx, N_FSy]
            )
            assert mod.allclose(diric_sig, diric_sig_exact)

            # Try real version
            diric_sig_real = fs_interpn(
                diric_FS, T=[T_x, T_y], a=[a_x, a_y], b=[b_x, b_y], M=[M_x, M_y], real_x=True
            )
            assert mod.allclose(diric_sig, diric_sig_real)
