import math
import pathlib
import time

import click
import matplotlib.pyplot as plt
import numpy as np

import util
from pyffs.func import dirichlet_fs
from pyffs.interp import fs_interpn


@click.command()
@click.option("--n_trials", type=int, default=50)
def profile_fs_interp2(n_trials):
    print(f"\nCOMPARING FS_INTERP WITH {n_trials} TRIALS")

    # parameters of signal
    T, T_c, M = math.pi, math.e, 10  # M^2 number of samples

    # sweep over number of interpolation points
    a, b = T_c + (T / 2) * np.r_[-1, 1]
    n_std = 1.0
    real_x = {"complex": False, "real": True}
    N_FS_vals = [11, 31, 101, 301, 1001, 3001]  # N_FS^2 coefficients
    proc_time = dict()
    proc_time_std = dict()
    for N_FS in N_FS_vals:
        print("\nNumber of FS coefficients : {}".format(N_FS))
        proc_time[N_FS] = dict()
        proc_time_std[N_FS] = dict()

        # compute FS coefficients
        diric_FS = np.outer(dirichlet_fs(N_FS, T, T_c), dirichlet_fs(N_FS, T, T_c))

        # Loop through functions
        for _f in real_x:
            timings = []
            for _ in range(n_trials):
                start_time = time.time()
                fs_interpn(diric_FS, T=[T, T], a=[a, a], b=[b, b], M=[M, M], real_x=real_x[_f])
                timings.append(time.time() - start_time)
            proc_time[N_FS][_f] = np.mean(timings)
            proc_time_std[N_FS][_f] = np.std(timings)
            print("{} version : {} seconds".format(_f, proc_time[N_FS][_f]))

    # plot results
    fig, ax = plt.subplots()
    util.comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{M} samples per dimension, {n_trials} trials")
    ax.set_xlabel("Number of FS coefficients per dimension")
    fig.tight_layout()

    fname = pathlib.Path(__file__).resolve().parent / "fs_interp_2D_real_speedup.png"
    fig.savefig(fname, dpi=300)


if __name__ == "__main__":
    profile_fs_interp2()
