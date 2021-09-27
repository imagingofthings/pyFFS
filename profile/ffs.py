import pathlib as plib
import time
import click
import matplotlib.pyplot as plt
import numpy as np
from util import comparison_plot, plotting_setup, backend_to_label
from pyffs import ffs_sample, ffs, next_fast_len
from pyffs.func import dirichlet
from pyffs.backend import AVAILABLE_MOD, get_module_name


@click.command()
@click.option("--n_trials", type=int, default=10)
def profile_ffsn(n_trials):
    print(f"\nCOMPARING FFSN APPROACHES WITH {n_trials} TRIALS")

    fig_path = plotting_setup(linewidth=3, font_size=20)

    T = 1
    T_c = 0
    N_FS_vals = [11, 31, 101, 301, 1001, 3001, 10001, 30001, 100001]

    n_std = 0.5

    proc_time = dict()
    proc_time_std = dict()

    for N_FS in N_FS_vals:
        print("\nN_FS : {}".format(N_FS))
        proc_time[N_FS] = dict()
        proc_time_std[N_FS] = dict()

        # Loop through modules
        for mod in AVAILABLE_MOD:

            backend = backend_to_label[get_module_name(mod)]

            # fastest FFT length depends on module
            N_s = next_fast_len(N_FS, mod=mod)
            print("-- module : {}, Length-{} FFT".format(backend, N_s))

            # Sample the kernel and do the transform.
            sample_points, _ = ffs_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s, mod=mod)
            diric_samples = dirichlet(x=sample_points, T=T, T_c=T_c, N_FS=N_FS)
            diric_samples = diric_samples.astype(
                "float32"
            )  # cast to float32, theoretically better for GPU
            ffs(x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)  # first one is a bit slow sometimes...
            timings = []
            for _ in range(n_trials):
                start_time = time.time()
                ffs(x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)
                timings.append(time.time() - start_time)
            proc_time[N_FS][backend] = np.mean(timings)
            proc_time_std[N_FS][backend] = np.std(timings)

            print("{} : {} seconds".format(backend, proc_time[N_FS][backend]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_xlabel("Number of FS coefficients")
    ax.set_xticks(np.array(N_FS_vals) - 1)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    fig.tight_layout()
    fig.savefig(plib.Path(fig_path) / "profile_ffs.png")

    plt.show()


if __name__ == "__main__":
    profile_ffsn()
