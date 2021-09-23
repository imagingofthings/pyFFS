import numpy as np
import pathlib as plib
from pyffs import ffs_sample, ffs, fs_interp
from pyffs.func import dirichlet
import matplotlib.pyplot as plt
import click
from scipy.signal import resample
from util import comparison_plot, plotting_setup
import time


@click.command()
@click.option("--n_samples", type=int, default=128)
@click.option("--n_trials", type=int, default=10)
@click.option("--n_interp", type=int, default=10000)
def profile_fs_interp(n_trials, n_samples, n_interp):
    fig_path = plotting_setup(linewidth=3, font_size=20)
    print(f"\nCOMPARING FFS AND FFT INTERP WITH {n_trials} TRIALS")
    n_std = 0.5

    M = n_interp
    width_vals = np.logspace(-3, 0, 10)

    T, T_c = 1, 0
    N_FS = n_samples - 1
    sample_points, _ = ffs_sample(T, N_FS, T_c, n_samples, mod=np)
    diric_samples = dirichlet(sample_points, T, T_c, N_FS)
    t_ord = np.sort(sample_points)
    diric_samples_ord = dirichlet(t_ord, T, T_c, N_FS)

    proc_time = dict()
    proc_time_std = dict()
    for val in width_vals:

        width = val * T
        start = T_c - width / 2
        stop = T_c + width / 2

        # interpolation points
        points = np.linspace(start, stop, M, endpoint=True)
        dt = points[1] - points[0]
        N_target = int(np.ceil(T / dt))
        if dt > sample_points[1] - sample_points[0]:
            print("Interpolation coarser than original sampling, skipping...")
            continue

        print("\nPeriod percentage : {}".format(val))
        proc_time[val] = dict()
        proc_time_std[val] = dict()

        # FFS
        _key = "pyffs.fs_interp"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            diric_FS = ffs(diric_samples, T, T_c, N_FS)[:N_FS]
            fs_interp(diric_FS, T=T, a=start, b=stop, M=M, real_x=False)
            timings.append(time.time() - start_time)
        proc_time[val][_key] = np.mean(timings)
        proc_time_std[val][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[val][_key]))

        # resample through zero padding
        _key = "scipy.signal.resample"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            resample(diric_samples_ord, N_target, t=np.sort(sample_points))

            timings.append(time.time() - start_time)
        proc_time[val][_key] = np.mean(timings)
        proc_time_std[val][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[val][_key]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.legend(loc="upper right")
    ax.set_title(f"{n_samples} samples, {M} interpolation points")
    ax.set_xlabel("Percentage of period")
    fig.tight_layout()
    fig.savefig(plib.Path(fig_path) / "bandlimited_interp1d_vary_width.png")

    plt.show()


if __name__ == "__main__":
    profile_fs_interp()
