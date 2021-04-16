import math
import pathlib
import time

import click
import matplotlib.pyplot as plt
import numpy as np

import util
from pyffs.func import dirichlet_fs
from pyffs.interp import fs_interp
from pyffs.backend import AVAILABLE_MOD, get_module_name


def naive_interp1d(diric_FS, T, a, b, M):

    sample_points = np.linspace(start=a, stop=b, num=M, endpoint=False)

    # loop as could be large matrix
    N_FS = len(diric_FS)
    K = N_FS // 2
    fs_idx = np.arange(-K, K + 1)
    vals = np.zeros(len(sample_points), dtype=complex)
    for i, _x_val in enumerate(sample_points):
        vals[i] = np.dot(diric_FS, np.exp(1j * 2 * np.pi / T * _x_val * fs_idx))
    return vals


@click.command()
@click.option("--n_trials", type=int, default=10)
def profile_fs_interp(n_trials):
    print(f"\nCOMPARING FS_INTERP WITH {n_trials} TRIALS")

    # parameters of signal
    T, T_c, M = math.pi, math.e, 1000
    N_FS_vals = [11, 31, 101, 301, 1001, 3001, 10001, 30001, 100001]

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

        # naive approach, apply synthesis formula
        diric_FS = dirichlet_fs(N_FS, T, T_c, mod=np)
        _key = "naive"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            naive_interp1d(diric_FS, T, a, b, M)
            timings.append(time.time() - start_time)
        proc_time[N_FS][_key] = np.mean(timings)
        proc_time_std[N_FS][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[N_FS][_key]))

        # Loop through modules
        for mod in AVAILABLE_MOD:
            backend = get_module_name(mod)
            print("-- module : {}".format(backend))

            # compute FS coefficients
            diric_FS = dirichlet_fs(N_FS, T, T_c, mod=mod)

            # Loop through functions
            for _f in real_x:
                _key = "{}_{}".format(_f, backend)
                timings = []
                for _ in range(n_trials):
                    start_time = time.time()
                    fs_interp(diric_FS, T, a, b, M, real_x=real_x[_f])
                    timings.append(time.time() - start_time)
                proc_time[N_FS][_key] = np.mean(timings)
                proc_time_std[N_FS][_key] = np.std(timings)
                print("{} version : {} seconds".format(_f, proc_time[N_FS][_key]))

    # plot results
    fig, ax = plt.subplots()
    util.comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{M} samples, {n_trials} trials")
    ax.set_xlabel("Number of FS coefficients")
    fig.tight_layout()

    fname = pathlib.Path(__file__).resolve().parent / "profile_fs_interp_1D.png"
    fig.savefig(fname, dpi=300)


if __name__ == "__main__":
    profile_fs_interp()
