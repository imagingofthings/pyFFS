import click
import math
import time
import numpy as np
import matplotlib.pyplot as plt
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
    real_Phi = {"complex": False, "real": True}
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
        for _f in real_Phi:
            timings = []
            for _ in range(n_trials):
                start_time = time.time()
                fs_interpn(diric_FS, T=[T, T], a=[a, a], b=[b, b], M=[M, M], real_Phi=real_Phi[_f])
                timings.append(time.time() - start_time)
            proc_time[N_FS][_f] = np.mean(timings)
            proc_time_std[N_FS][_f] = np.std(timings)
            print("{} version : {} seconds".format(_f, proc_time[N_FS][_f]))

    # plot results
    markers = ["o", "^", "v", "x", ">", "<", "D", "+"]
    plt.figure()
    for i, _f in enumerate(real_Phi):
        _proc_time = []
        _proc_time_std = []
        for N_FS in N_FS_vals:
            _proc_time.append(proc_time[N_FS][_f])
            _proc_time_std.append(proc_time_std[N_FS][_f])
        _proc_time = np.array(_proc_time)
        _proc_time_std = np.array(_proc_time_std)

        plt.loglog(N_FS_vals, _proc_time, label=_f, marker=markers[i])
        ax = plt.gca()
        ax.fill_between(
            N_FS_vals,
            (_proc_time - n_std * _proc_time_std),
            (_proc_time + n_std * _proc_time_std),
            alpha=0.2,
        )

    plt.legend()
    plt.title(f"{M} samples per dimension, {n_trials} trials")
    plt.xlabel("Number of FS coefficients per dimension")
    plt.ylabel("Processing time (s)")
    plt.grid()
    ax = plt.gca()
    ax.set_xticks(N_FS_vals)
    plt.tight_layout()
    plt.savefig("fs_interp_2D_real_speedup.png")


if __name__ == "__main__":
    profile_fs_interp2()
