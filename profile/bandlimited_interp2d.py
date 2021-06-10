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


def fft2_interpolate(dft, T, dx, dy):
    Nx_target = int(np.ceil(T[0] / dx))
    Ny_target = int(np.ceil(T[1] / dy))

    nx_pad = Nx_target - dft.shape[0]
    ny_pad = Ny_target - dft.shape[1]
    X_pad = np.pad(
        dft,
        pad_width=((nx_pad // 2, nx_pad // 2), (ny_pad // 2, ny_pad // 2)),
        mode="constant",
        constant_values=0,
    )
    X_pad = np.fft.fftshift(X_pad)
    np.real(np.fft.ifft2(X_pad))


@click.command()
@click.option("--n_trials", type=int, default=10)
def profile_fs_interp(n_trials):
    print(f"\nCOMPARING FFS AND FFT INTERP WITH {n_trials} TRIALS")
    n_std = 0.5

    start = 2 * [-0.1]
    stop = 2 * [0.1]
    M_vals = [100, 300, 1000]

    T = 2 * [1]
    T_c = 2 * [0]
    N_s = 2 * [256]
    N_FS = [N_s[0] - 1, N_s[1] - 1]

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
        diric_FS = ffsn(x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)[: N_FS[0], : N_FS[1]]
        for _ in range(n_trials):
            start_time = time.time()
            fs_interpn(diric_FS, T=T, a=start, b=stop, M=[num, num], real_x=True)
            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

        # FFT
        _key = "FFT"
        timings = []
        dft = np.fft.fftshift(np.fft.fft2(diric_samples_ord))
        for _ in range(n_trials):
            start_time = time.time()
            fft2_interpolate(dft, T, dx, dy)
            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

    # plot results
    fig, ax = plt.subplots()
    util.comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_title(f"{N_s} samples, {n_trials} trials")
    ax.set_xlabel("Number of interpolation points")
    fig.tight_layout()

    fname = pathlib.Path(__file__).resolve().parent / "profile_ffs_vs_fft_interp2d.png"
    fig.savefig(fname, dpi=300)

    plt.show()


if __name__ == "__main__":
    profile_fs_interp()
