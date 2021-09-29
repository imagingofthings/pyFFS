import pathlib as plib
import time
import click
import matplotlib.pyplot as plt
import numpy as np
from util import comparison_plot, plotting_setup
from pyffs import ffsn_sample, convolve, iffs_shift
from pyffs.func import dirichlet_2D
from scipy.signal import convolve2d as convolve_scipy


@click.command()
@click.option("--n_trials", type=int, default=1)
def profile_ffsn(n_trials):
    fig_path = plotting_setup(linewidth=3, font_size=20)

    T = [1, 1]
    T_c = [0, 0]
    N_S_vals = np.logspace(np.log10(10), np.log10(300), num=10)

    n_std = 0.5

    proc_time = dict()
    proc_time_std = dict()

    for _N_S in N_S_vals:
        _N_S = int(_N_S // 2 * 2)
        print("\nN_S : {}".format(_N_S))
        N_s = [_N_S, _N_S]
        N_FS = [_N_S - 1, _N_S - 1]
        proc_time[_N_S] = dict()
        proc_time_std[_N_S] = dict()

        # Sample the kernel.
        sample_points, idx = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s, mod=np)
        diric_samples = dirichlet_2D(sample_points=sample_points, T=T, T_c=T_c, N_FS=N_FS)

        # pyFFS
        _key = "pyffs.convolve2d"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            convolve(f=diric_samples, h=diric_samples, T=T, T_c=T_c, N_FS=N_FS, reorder=True)
            timings.append(time.time() - start_time)
        proc_time[_N_S][_key] = np.mean(timings)
        proc_time_std[_N_S][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[_N_S][_key]))

        # SciPy
        diric_samples_ord = iffs_shift(diric_samples)
        _key = "scipy.signal.convolve2d (wrap)"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            convolve_scipy(
                diric_samples_ord, diric_samples_ord, mode="same", boundary="wrap"
            ) / N_s[0] / N_s[1]
            timings.append(time.time() - start_time)
        proc_time[_N_S][_key] = np.mean(timings)
        proc_time_std[_N_S][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[_N_S][_key]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_xlabel("Number of samples per dimension")
    ax.set_xticks([10, 30, 100, 300])
    fig.tight_layout()
    fig.savefig(plib.Path(fig_path) / "profile_convolve2d.png")

    plt.show()


if __name__ == "__main__":
    profile_ffsn()
