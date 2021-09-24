import numpy as np
import pathlib as plib
from pyffs import ffs_sample, ffs, fs_interp
from pyffs.func import dirichlet
import matplotlib.pyplot as plt
import click
from scipy.signal import resample
from scipy.interpolate import interp1d
from util import comparison_plot, plotting_setup
import time


def sinc_interp(x, s, u):
    if len(x) != len(s):
        raise ValueError
    # Find the period
    T = s[1] - s[0]
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    return np.dot(x, np.sinc(sincM / T))


@click.command()
@click.option("--n_samples", type=int, default=128)
@click.option("--n_trials", type=int, default=10)
@click.option("--percent_period", type=float, default=0.1)
def profile_fs_interp(n_trials, n_samples, percent_period):
    fig_path = plotting_setup(linewidth=3, font_size=20)
    print(f"\nCOMPARING FFS AND FFT INTERP WITH {n_trials} TRIALS")
    n_std = 0.5
    M_vals = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]

    T, T_c = 1, 0
    N_FS = n_samples - 1
    sample_points, _ = ffs_sample(T, N_FS, T_c, n_samples, mod=np)
    diric_samples = dirichlet(sample_points, T, T_c, N_FS)
    t_ord = np.sort(sample_points)
    diric_samples_ord = dirichlet(t_ord, T, T_c, N_FS)

    width = percent_period * T
    start = T_c - width / 2
    stop = T_c + width / 2

    proc_time = dict()
    proc_time_std = dict()
    for num in M_vals:

        # interpolation points
        points = np.linspace(start, stop, num, endpoint=True)
        dt = points[1] - points[0]
        N_target = int(np.ceil(T / dt))
        if dt > sample_points[1] - sample_points[0]:
            print("Interpolation coarser than original sampling, skipping...")
            continue

        print("\nNumber of interpolation points : {}".format(num))
        proc_time[num] = dict()
        proc_time_std[num] = dict()

        # FFS
        _key = "pyffs.fs_interp"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            diric_FS = ffs(diric_samples, T, T_c, N_FS)[:N_FS]
            fs_interp(diric_FS, T=T, a=start, b=stop, M=num, real_x=False)
            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

        # FFT
        _key = "scipy.signal.resample"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            resample(diric_samples_ord, N_target, t=np.sort(sample_points))

            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

        # # linear
        # _key = "linear"
        # timings = []
        # for _ in range(n_trials):
        #     start_time = time.time()
        #     f = interp1d(x=sample_points, y=diric_samples, kind="linear")
        #     f(points)
        #     timings.append(time.time() - start_time)
        # proc_time[num][_key] = np.mean(timings)
        # proc_time_std[num][_key] = np.std(timings)
        # print("-- {} : {} seconds".format(_key, proc_time[num][_key]))
        #
        # # cubic
        # _key = "cubic"
        # timings = []
        # for _ in range(n_trials):
        #     start_time = time.time()
        #     f = interp1d(x=sample_points, y=diric_samples, kind="cubic")
        #     f(points)
        #     timings.append(time.time() - start_time)
        # proc_time[num][_key] = np.mean(timings)
        # proc_time_std[num][_key] = np.std(timings)
        # print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

        # # sinc -- VERY SLOW
        # _key = "sinc"
        # timings = []
        # for _ in range(n_trials):
        #     start_time = time.time()
        #     sinc_interp(diric_samples_ord, t_ord, points)
        #     timings.append(time.time() - start_time)
        # proc_time[num][_key] = np.mean(timings)
        # proc_time_std[num][_key] = np.std(timings)
        # print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{n_samples} samples, {percent_period*100}% of period")
    ax.set_xlabel("Number of interpolation points in section")
    fig.tight_layout()

    # # plot theoretical crossover
    # m = (stop - start) / T
    # dt = T / N_FS * (1 - m)
    # M_cross = int(m * T / dt)
    # if M_cross > 0:
    #     ax.axvline(x=M_cross, linestyle="--", color="g")
    fig.savefig(plib.Path(fig_path) / "bandlimited_interp1d_vary_M.png")

    plt.show()


if __name__ == "__main__":
    profile_fs_interp()
