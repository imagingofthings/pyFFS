import math
import pathlib
import time

import click
import matplotlib.pyplot as plt
import numpy as np

import util
from pyffs.func import dirichlet_fs
from pyffs.interp import fs_interpn
from pyffs.backend import AVAILABLE_MOD, get_module_name


def naive_interp2d(diric_FS, T, a, b, M):

    # create sample points
    D = len(T)
    sample_points = []
    for d in range(D):
        sh = [1] * D
        sh[d] = M[d]
        sample_points.append(
            np.linspace(start=a[d], stop=b[d], num=M[d], endpoint=False).reshape(sh)
        )

    # initialize output
    x_vals = np.linspace(start=a[0], stop=b[0], num=M[0], endpoint=False)
    y_vals = np.linspace(start=a[1], stop=b[1], num=M[1], endpoint=False)
    output_shape = (len(x_vals), len(y_vals))
    vals = np.zeros(output_shape, dtype=complex)

    # loop to avoid creating potentially large matrices
    N_FSx, N_FSy = diric_FS.shape
    Kx = N_FSx // 2
    Ky = N_FSy // 2
    fsx_idx = np.arange(-Kx, Kx + 1)[:, np.newaxis]
    fsy_idx = np.arange(-Ky, Ky + 1)[np.newaxis, :]
    for i, _x_val in enumerate(x_vals):
        for j, _y_val in enumerate(y_vals):
            vals[i, j] = np.sum(
                diric_FS
                * np.exp(1j * 2 * np.pi * fsx_idx / T[0] * _x_val)
                * np.exp(1j * 2 * np.pi * fsy_idx / T[1] * _y_val)
            )
    return vals


@click.command()
@click.option("--n_trials", type=int, default=10)
def profile_fs_interp2(n_trials):
    print(f"\nCOMPARING FS_INTERP WITH {n_trials} TRIALS")

    # parameters of signal
    T, T_c, M = math.pi, math.e, 10  # M^2 number of samples
    N_FS_vals = [11, 31, 101, 301, 1001, 3001, 10001]  # N_FS^2 coefficients

    # sweep over number of interpolation points
    a, b = T_c + (T / 2) * np.r_[-1, 1]
    n_std = 0.5
    real_x = {"complex": False, "real": True}
    proc_time = dict()
    proc_time_std = dict()

    for N_FS in N_FS_vals:
        print("\nNumber of FS coefficients : {}".format(N_FS))
        proc_time[N_FS] = dict()
        proc_time_std[N_FS] = dict()

        # naive approach
        diric_FS = np.outer(dirichlet_fs(N_FS, T, T_c, mod=np), dirichlet_fs(N_FS, T, T_c, mod=np))
        _key = "naive"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            naive_interp2d(diric_FS, [T, T], [a, a], [b, b], [M, M])
            timings.append(time.time() - start_time)
        proc_time[N_FS][_key] = np.mean(timings)
        proc_time_std[N_FS][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[N_FS][_key]))

        # Loop through modules
        for mod in AVAILABLE_MOD:
            backend = get_module_name(mod)
            print("-- module : {}".format(backend))

            # compute FS coefficients
            diric_FS = mod.outer(
                dirichlet_fs(N_FS, T, T_c, mod=mod), dirichlet_fs(N_FS, T, T_c, mod=mod)
            )

            # Loop through functions
            for _f in real_x:
                _key = "{}_{}".format(_f, backend)
                timings = []
                for _ in range(n_trials):
                    start_time = time.time()
                    fs_interpn(diric_FS, T=[T, T], a=[a, a], b=[b, b], M=[M, M], real_x=real_x[_f])
                    timings.append(time.time() - start_time)
                proc_time[N_FS][_key] = np.mean(timings)
                proc_time_std[N_FS][_key] = np.std(timings)
                print("{} version : {} seconds".format(_f, proc_time[N_FS][_key]))

    # plot results
    fig, ax = plt.subplots()
    util.comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{M} samples per dimension, {n_trials} trials")
    ax.set_xlabel("Number of FS coefficients per dimension")
    fig.tight_layout()

    fname = pathlib.Path(__file__).resolve().parent / "profile_fs_interp_2D.png"
    fig.savefig(fname, dpi=300)


if __name__ == "__main__":
    profile_fs_interp2()
