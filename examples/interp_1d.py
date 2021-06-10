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


def fft_interpolate(samples, T, dt, T_c, offset=None):
    N_target = int(np.ceil(T / dt))
    X = np.fft.fftshift(np.fft.fft(samples))
    n_pad = N_target - len(samples)
    X_pad = np.pad(X, pad_width=(n_pad // 2, n_pad // 2), mode="constant", constant_values=0)

    # X_pad = np.fft.ifftshift(X_pad) * np.exp(- 1j * 2 * np.pi * np.fft.fftfreq(N_target, dt))
    X_pad = np.fft.ifftshift(X_pad)
    if offset is not None:
        X_pad *= np.exp(-1j * 2 * np.pi * np.fft.fftfreq(len(X_pad), dt) * offset)
    vals_fft = np.real(np.fft.ifft(X_pad)) * len(X_pad) / len(samples)
    t_fft = np.linspace(start=T_c - T / 2, stop=T_c + T / 2, num=len(vals_fft), endpoint=True)

    return vals_fft, t_fft


def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """

    if len(x) != len(s):
        raise ValueError
    # Find the period
    T = s[1] - s[0]
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    return np.dot(x, np.sinc(sincM / T))


start = -0.05
stop = 0.15
num = 128

T, T_c, N_samples = 1, 0, 64
N_FS = 51
sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
diric_samples = dirichlet(sample_points, T, T_c, N_FS)
diric_FS = ffs(diric_samples, T, T_c, N_FS)[:N_FS]

# interpolation points
points = np.linspace(start, stop, num, endpoint=True)
dt = points[1] - points[0]

# interpolate FFS
vals_ffs = fs_interp(diric_FS, T=T, a=start, b=stop, M=num, real_x=True)

# interpolate with zero padding
N_target = int(np.ceil(T / dt))
diric_samples_ord = dirichlet(np.sort(sample_points), T, T_c, N_FS)
offset = np.min(np.abs(sample_points))  # offset from `ffs_sample`
vals_fft, t_fft = fft_interpolate(diric_samples_ord, T, dt, T_c, offset=offset)

# sinc interpolation
t_interp, _ = ffs_sample(T, N_FS, T_c, N_target)
t_interp = np.sort(t_interp)
vals_sinc = sinc_interp(diric_samples_ord, np.sort(sample_points), t_interp)

# interpolate with scipy
f_linear = interp1d(x=sample_points, y=diric_samples, kind="linear")
f_cubic = interp1d(x=sample_points, y=diric_samples, kind="cubic")
vals_linear = f_linear(points)
vals_cubic = f_cubic(points)

# plot
linewidth = 3

# input
t_gt, _ = ffs_sample(T, N_FS, T_c, 4096)
t_gt = np.sort(t_gt)

idx_order = np.argsort(sample_points)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(t_gt, dirichlet(t_gt, T, T_c, N_FS), label="ground truth")
ax.scatter(sample_points[idx_order], diric_samples[idx_order], label="provided samples")
ax.set_xlabel("x [m]")
ax.set_xlim([T_c - T / 2, T_c + T / 2])
ax.axvline(
    x=start,
    c="k",
    linestyle="--",
    linewidth=linewidth,
)
ax.axvline(
    x=stop,
    c="k",
    linestyle="--",
    linewidth=linewidth,
)
plt.legend()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_1d_input.png"))


# -- zoomed in region
idx = np.logical_and(sample_points >= start, sample_points <= stop)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(points, vals_ffs, label="FFS")
ax.plot(t_fft, vals_fft, label="FFT")
# ax.plot(t_interp, vals_sinc, label="sinc")
ax.plot(points, vals_linear, label="linear")
ax.plot(points, vals_cubic, label="cubic")
ax.plot(t_interp, dirichlet(t_interp, T, T_c, N_FS), label="ground truth")
ax.scatter(sample_points, diric_samples, label="provided samples")
ax.set_xlabel("x [m]")
ax.set_xlim([start, stop])
plt.legend()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_1d.png"))

plt.show()
