import numpy as np
import pathlib
from pyffs import ffsn_sample, ffsn, fs_interpn
from pyffs.func import dirichlet_2D
import matplotlib
import matplotlib.pyplot as plt
import click
import util
import time


font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)


def fft2_interpolate(samples, T, dx, dy, T_c):
    Nx_target = int(np.ceil(T[0] / dx))
    Ny_target = int(np.ceil(T[1] / dy))
    X = np.fft.fftshift(np.fft.fft2(samples))
    nx_pad = Nx_target - samples.shape[0]
    ny_pad = Ny_target - samples.shape[1]
    X_pad = np.pad(
        X,
        pad_width=((nx_pad // 2, nx_pad // 2), (ny_pad // 2, ny_pad // 2)),
        mode="constant",
        constant_values=0,
    )
    X_pad = np.fft.fftshift(X_pad)
    vals_fft = np.real(np.fft.ifft2(X_pad))
    x_fft = np.linspace(
        start=T_c[0] - T[0] / 2, stop=T_c[0] + T[0] / 2, num=vals_fft.shape[0], endpoint=True
    )
    y_fft = np.linspace(
        start=T_c[1] - T[1] / 2, stop=T_c[1] + T[1] / 2, num=vals_fft.shape[1], endpoint=True
    )

    return vals_fft, x_fft, y_fft


@click.command()
@click.option("--n_trials", type=int, default=10)
def profile_fs_interp(n_trials):
    print(f"\nCOMPARING FFS AND FFT INTERP WITH {n_trials} TRIALS")
    n_std = 0.5

    start = [-0.1, -0.1]
    stop = [0.1, 0.1]
    M_vals = [11, 31, 101, 301, 1001]

    T = [1, 1]
    T_c = [0, 0]
    N_FS = [121, 121]
    N_s = [128, 128]

    sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
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
        _key = "FFS"
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
        _key = "FFT"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            fft2_interpolate(diric_samples_ord, T, dx, dy, T_c)
            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

    # plot results
    fig, ax = plt.subplots()
    util.comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{N_FS} FS coefficients, {n_trials} trials")
    ax.set_xlabel("Number of interpolation points")
    fig.tight_layout()

    fname = pathlib.Path(__file__).resolve().parent / "profile_ffs_vs_fft_interp2d.png"
    fig.savefig(fname, dpi=300)

    plt.show()


if __name__ == "__main__":
    profile_fs_interp()
