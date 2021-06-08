"""
Interpolation example, FFS is equivalent to FFT, namely bandlimited interpolation.


"""

import numpy as np
from pyffs import ffs_sample, ffs, fs_interp
from pyffs.func import dirichlet
import matplotlib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os


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


start = -0.05
stop = 0.05
num = 2048

T, T_c, N_FS, N_samples = 1, 0, 121, 128
sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
diric_samples = dirichlet(sample_points, T, T_c, N_FS)
diric_FS = ffs(diric_samples, T, T_c, N_FS)[:N_FS]

# interpolation points
points = np.linspace(start, stop, num, endpoint=True)
dt = points[1] - points[0]

# interpolate FFS
vals_ffs = fs_interp(diric_FS, T=T, a=start, b=stop, M=num, real_x=True)

# interpolate with zero padding
diric_samples_ord = dirichlet(np.sort(sample_points), T, T_c, N_FS)
vals_fft, t_fft = fft_interpolate(diric_samples_ord, T, dt, T_c)

# interpolate with scipy
f_linear = interp1d(x=sample_points, y=diric_samples, kind="linear")
f_cubic = interp1d(x=sample_points, y=diric_samples, kind="cubic")
vals_linear = f_linear(points)
vals_cubic = f_cubic(points)

# plot
idx = np.logical_and(sample_points >= start, sample_points <= stop)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(points, vals_ffs, label="FFS")
ax.plot(t_fft, vals_fft, label="FFT")
ax.plot(points, vals_cubic, label="cubic")
ax.plot(points, vals_linear, label="linear")
ax.plot(points, dirichlet(points, T, T_c, N_FS), label="true")
# ax.scatter(sample_points[idx], diric_samples[idx], label="available")
ax.scatter(sample_points, diric_samples, label="available")
ax.set_xlabel("x [m]")
ax.set_xlim([start, stop])
plt.legend()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_1d.png"))

plt.show()
