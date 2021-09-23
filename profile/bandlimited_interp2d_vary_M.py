import numpy as np
import pathlib as plib
from pyffs import ffsn_sample, ffsn, fs_interpn
from pyffs.func import dirichlet_2D
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.signal import resample
import click
from util import comparison_plot, plotting_setup
import time


@click.command()
@click.option("--n_samples", type=int, default=32)
@click.option("--n_trials", type=int, default=10)
@click.option("--percent_region", type=float, default=0.04)
def profile_fs_interp(n_samples, n_trials, percent_region):
    fig_path = plotting_setup(linewidth=3, font_size=20)
    print(f"\nCOMPARING FFS AND FFT INTERP WITH {n_trials} TRIALS")
    n_std = 0.5

    M_vals = [10, 30, 100, 300, 1000, 3000]

    T = 2 * [1]
    T_c = 2 * [0]
    N_s = 2 * [n_samples]
    N_FS = [N_s[0] - 1, N_s[1] - 1]

    side_percent = np.sqrt(percent_region)
    width = side_percent * np.array(T)
    start = np.array(T_c) - width / 2
    stop = np.array(T_c) + width / 2

    sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s, mod=np)
    diric_samples = dirichlet_2D(sample_points, T, T_c, N_FS)
    diric_samples_ord = dirichlet_2D(
        [np.sort(sample_points[0], axis=0), np.sort(sample_points[1])], T, T_c, N_FS
    )

    proc_time = dict()
    proc_time_std = dict()
    for num in M_vals:

        # interpolation points
        points_x = np.linspace(start[0], stop[0], num, endpoint=True)
        dx = points_x[1] - points_x[0]
        points_y = np.linspace(start[1], stop[1], num, endpoint=True)
        dy = points_y[1] - points_y[0]

        x_vals = np.squeeze(sample_points[0])
        y_vals = np.squeeze(sample_points[1])
        if dx > x_vals[1] - x_vals[0] or dy > y_vals[1] - y_vals[0]:
            print("Interpolation coarser than original sampling, skipping...")
            continue

        print("\nNumber of interpolation points : {}x{}".format(num, num))
        proc_time[num] = dict()
        proc_time_std[num] = dict()

        # FFS
        _key = "pyffs.fs_interpn"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            diric_FS = ffsn(x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)[: N_FS[0], : N_FS[1]]
            fs_interpn(diric_FS, T=T, a=start, b=stop, M=[num, num], real_x=True)
            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

        # FFT
        # resample 2D
        _key = "scipy.signal.resample x2 "
        timings = []
        Nx_target = int(np.ceil(T[0] / dx))
        sample_points_x = np.squeeze(np.sort(sample_points[0], axis=0))
        Ny_target = int(np.ceil(T[1] / dy))
        sample_points_y = np.squeeze(np.sort(sample_points[1], axis=1))
        for _ in range(n_trials):
            start_time = time.time()
            vals_resample, _ = resample(diric_samples_ord, Nx_target, t=sample_points_x, axis=0)
            resample(vals_resample, Ny_target, t=sample_points_y, axis=1)
            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

        # # linear
        # _key = "linear"
        # timings = []
        # for _ in range(n_trials):
        #     start_time = time.time()
        #     f = interp2d(x=sample_points[0], y=sample_points[1], z=diric_samples, kind="linear")
        #     f(points_x, points_y)
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
        #     f = interp2d(x=sample_points[0], y=sample_points[1], z=diric_samples, kind="cubic")
        #     f(points_x, points_y)
        #     timings.append(time.time() - start_time)
        # proc_time[num][_key] = np.mean(timings)
        # proc_time_std[num][_key] = np.std(timings)
        # print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{N_s} samples, {percent_region*100}% of period")
    ax.set_xlabel("Number of interpolation points in section")
    fig.tight_layout()
    fig.savefig(plib.Path(fig_path) / "bandlimited_interp2d_vary_M.png")

    plt.show()


if __name__ == "__main__":
    profile_fs_interp()
