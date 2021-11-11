import math
import pathlib as plib
import time
import click
import matplotlib.pyplot as plt
import numpy as np
from util import comparison_plot, plotting_setup, backend_to_label, naive_interp1d
from pyffs.func import dirichlet_fs
from pyffs.interp import fs_interp
from pyffs.backend import AVAILABLE_MOD, get_module_name


@click.command()
@click.option("--n_interp", type=int, default=1000)
@click.option("--n_trials", type=int, default=10)
@click.option("--direct", is_flag=True)
def profile_fs_interp(n_interp, n_trials, direct):
    print(f"\nCOMPARING FS_INTERP WITH {n_trials} TRIALS")

    fig_path = plotting_setup(linewidth=3, font_size=20)

    # parameters of signal
    T, T_c, M = math.pi, math.e, n_interp
    N_FS_vals = [11, 31, 101, 301, 1001, 3001, 10001, 30001, 100001]

    # sweep over number of interpolation points
    a, b = T_c + (T / 2) * np.r_[-1, 1]
    n_std = 0.5

    # real_x = {"complex": False, "real": True}
    real_x = {"complex": False}

    proc_time = dict()
    proc_time_std = dict()
    for N_FS in N_FS_vals:
        print("\nNumber of FS coefficients : {}".format(N_FS))
        proc_time[N_FS] = dict()
        proc_time_std[N_FS] = dict()

        # Loop through modules
        for mod in AVAILABLE_MOD:
            backend = backend_to_label[get_module_name(mod)]
            print("-- module : {}".format(backend))

            # compute FS coefficients
            diric_FS = dirichlet_fs(N_FS, T, T_c, mod=mod).astype("complex64")

            # Loop through functions
            for _f in real_x:
                if len(real_x.keys()) > 1:
                    _key = "{}_{}".format(_f, backend)
                else:
                    _key = backend
                timings = []
                fs_interp(
                    diric_FS, T, a, b, M, real_x=real_x[_f]
                )  # first one is a bit slow sometimes...
                for _ in range(n_trials):
                    start_time = time.time()
                    fs_interp(diric_FS, T, a, b, M, real_x=real_x[_f])
                    timings.append(time.time() - start_time)
                proc_time[N_FS][_key] = np.mean(timings)
                proc_time_std[N_FS][_key] = np.std(timings)
                print("{} version : {} seconds".format(_f, proc_time[N_FS][_key]))

        # naive approach, apply synthesis formula
        if direct:
            diric_FS = dirichlet_fs(N_FS, T, T_c, mod=np).astype("complex64")
            _key = "direct"
            timings = []
            for _ in range(n_trials):
                start_time = time.time()
                naive_interp1d(diric_FS, T, a, b, M)
                timings.append(time.time() - start_time)
            proc_time[N_FS][_key] = np.mean(timings)
            proc_time_std[N_FS][_key] = np.std(timings)
            print("-- {} : {} seconds".format(_key, proc_time[N_FS][_key]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{M} interpolation points")
    ax.set_xlabel("Number of FS coefficients")
    ax.set_xticks(np.array(N_FS_vals) - 1)
    if direct:
        ax.set_yticks([1e-3, 1e-1, 1e1])
    else:
        ax.set_yticks([1e-3, 1e-2, 1e-1])
    fig.tight_layout()
    fig.savefig(plib.Path(fig_path) / "profile_fs_interp1d_vary_NFS.png")

    plt.show()


if __name__ == "__main__":
    profile_fs_interp()
