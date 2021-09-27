import math
import pathlib as plib
import time
import click
import matplotlib.pyplot as plt
import numpy as np
from util import comparison_plot, plotting_setup, backend_to_label, naive_interp2d
from pyffs.func import dirichlet_fs
from pyffs.interp import fs_interpn
from pyffs.backend import AVAILABLE_MOD, get_module_name


@click.command()
@click.option("--n_interp", type=int, default=1000)
@click.option("--n_trials", type=int, default=10)
def profile_fs_interp2(n_interp, n_trials):
    print(f"\nCOMPARING FS_INTERP WITH {n_trials} TRIALS")
    fig_path = plotting_setup(linewidth=3, font_size=20)

    # parameters of signal
    M = n_interp
    T, T_c = math.pi, math.e  # M^2 number of samples
    N_FS_vals = [11, 31, 101, 301, 1001, 3001, 10001]  # N_FS^2 coefficients

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
            diric_FS = mod.outer(
                dirichlet_fs(N_FS, T, T_c, mod=mod), dirichlet_fs(N_FS, T, T_c, mod=mod)
            ).astype("complex64")

            # Loop through functions
            for _f in real_x:
                if len(real_x.keys()) > 1:
                    _key = "{}_{}".format(_f, backend)
                else:
                    _key = backend
                timings = []
                fs_interpn(
                    diric_FS, T=[T, T], a=[a, a], b=[b, b], M=[M, M], real_x=real_x[_f]
                )  # first time is a bit slow sometimes...
                for _ in range(n_trials):
                    start_time = time.time()
                    fs_interpn(diric_FS, T=[T, T], a=[a, a], b=[b, b], M=[M, M], real_x=real_x[_f])
                    timings.append(time.time() - start_time)
                proc_time[N_FS][_key] = np.mean(timings)
                proc_time_std[N_FS][_key] = np.std(timings)
                print("{} version : {} seconds".format(_f, proc_time[N_FS][_key]))

        # # naive approach - MUCH TOO SLOW!
        # diric_FS = np.outer(dirichlet_fs(N_FS, T, T_c, mod=np), dirichlet_fs(N_FS, T, T_c, mod=np)).astype("complex64")
        # _key = "direct"
        # timings = []
        # for _ in range(n_trials):
        #     start_time = time.time()
        #     naive_interp2d(diric_FS, [T, T], [a, a], [b, b], [M, M])
        #     timings.append(time.time() - start_time)
        # proc_time[N_FS][_key] = np.mean(timings)
        # proc_time_std[N_FS][_key] = np.std(timings)
        # print("-- {} : {} seconds".format(_key, proc_time[N_FS][_key]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{M} interpolations points per dimension")
    ax.set_xlabel("Number of FS coefficients per dimension")
    ax.set_xticks(np.array(N_FS_vals) - 1)
    fig.tight_layout()
    fig.savefig(plib.Path(fig_path) / "profile_fs_interp2d_vary_NFS.png")

    plt.show()


if __name__ == "__main__":
    profile_fs_interp2()
