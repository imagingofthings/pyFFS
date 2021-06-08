import numpy as np
import pathlib
from pyffs import ffs_sample, ffs, fs_interp
from pyffs.func import dirichlet
import matplotlib
import matplotlib.pyplot as plt
import click
import util
import time


font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)


def fft_interpolate(samples, T, dt, T_c):
    N_target = int(np.ceil(T / dt))
    X = np.fft.fftshift(np.fft.fft(samples))
    n_pad = N_target - len(samples)
    X_pad = np.pad(X, pad_width=(n_pad // 2, n_pad // 2), mode="constant", constant_values=0)
    X_pad = np.fft.fftshift(X_pad)
    vals_fft = np.real(np.fft.ifft(X_pad)) * len(X_pad) / len(samples)
    dt_fft = T / len(X_pad)
    # TODO : hack for aligning
    # shift = 4 * dt_fft
    shift = 0
    t_fft = np.linspace(
        start=T_c - T / 2 + shift, stop=T_c + T / 2 + shift, num=len(vals_fft), endpoint=True
    )

    return vals_fft, t_fft


@click.command()
@click.option("--n_trials", type=int, default=20)
def profile_fs_interp(n_trials):
    print(f"\nCOMPARING FFS AND FFT INTERP WITH {n_trials} TRIALS")
    n_std = 0.5

    start = -0.05
    stop = 0.05
    M_vals = [101, 301, 1001, 3001, 10001, 30001, 100001]

    T, T_c, N_FS, N_samples = 1, 0, 511, 512
    sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
    diric_samples = dirichlet(sample_points, T, T_c, N_FS)
    diric_samples_ord = dirichlet(np.sort(sample_points), T, T_c, N_FS)

    proc_time = dict()
    proc_time_std = dict()
    for num in M_vals:

        # interpolation points
        points = np.linspace(start, stop, num, endpoint=True)
        dt = points[1] - points[0]
        if dt > sample_points[1] - sample_points[0]:
            print("Interpolation coarser than original sampling, skipping...")
            continue

        print("\nNumber of interpolation points : {}".format(num))
        proc_time[num] = dict()
        proc_time_std[num] = dict()

        # FFS
        _key = "FFS"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            diric_FS = ffs(diric_samples, T, T_c, N_FS)[:N_FS]
            fs_interp(diric_FS, T=T, a=start, b=stop, M=num, real_x=True)
            timings.append(time.time() - start_time)
        proc_time[num][_key] = np.mean(timings)
        proc_time_std[num][_key] = np.std(timings)
        print("-- {} : {} seconds".format(_key, proc_time[num][_key]))

        # FFT
        _key = "FFT"
        timings = []
        for _ in range(n_trials):
            start_time = time.time()
            fft_interpolate(diric_samples_ord, T, dt, T_c)
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

    fname = pathlib.Path(__file__).resolve().parent / "profile_ffs_vs_fft_interp1d.png"
    fig.savefig(fname, dpi=300)

    plt.show()


if __name__ == "__main__":
    profile_fs_interp()
