"""
Interpolation example, FFS is equivalent to FFT, namely bandlimited interpolation.


"""

import numpy as np
from pyffs import ffs_sample, ffs, fs_interp
from pyffs.func import dirichlet
import matplotlib
from scipy.interpolate import interp1d
from scipy.signal import resample
import matplotlib.pyplot as plt
import os


font = {"family": "Times New Roman", "weight": "normal", "size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["lines.linewidth"] = 4
matplotlib.rcParams["lines.markersize"] = 10


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
    H = np.sinc(sincM / T)
    return np.dot(x, H)


start = 0.17
stop = 0.27
num = 128

T, T_c, N_samples = 1, 0, 64
T_c_diric = 0.2
N_FS = 51
sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
diric_samples = dirichlet(sample_points, T, T_c_diric, N_FS)
diric_FS = ffs(diric_samples, T, T_c, N_FS)[:N_FS]

# interpolation points
points = np.linspace(start, stop, num, endpoint=True)
dt = points[1] - points[0]

# interpolate FFS
vals_ffs = fs_interp(diric_FS, T=T, a=start, b=stop, M=num, real_x=True)

# interpolate with zero padding
N_target = int(np.ceil(T / dt))
diric_samples_ord = dirichlet(np.sort(sample_points), T, T_c_diric, N_FS)
resampled_x, resampled_t = resample(diric_samples_ord, N_target, t=np.sort(sample_points))

# sinc interpolation
sample_points_ord = np.sort(sample_points)
vals_sinc = sinc_interp(diric_samples_ord, sample_points_ord, points)

# interpolate with scipy
f_linear = interp1d(x=sample_points, y=diric_samples, kind="linear")
f_cubic = interp1d(x=sample_points, y=diric_samples, kind="cubic")
vals_linear = f_linear(points)
vals_cubic = f_cubic(points)

# plot

# input
t_gt, _ = ffs_sample(T, N_FS, T_c, 4096)
t_gt = np.sort(t_gt)

idx_order = np.argsort(sample_points)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(t_gt, dirichlet(t_gt, T, T_c_diric, N_FS), label="ground truth", alpha=0.7)
ax.scatter(sample_points[idx_order], diric_samples[idx_order], label="available samples")
ax.set_xlabel("x [m]")
ax.set_xlim([T_c - T / 2, T_c + T / 2])
ax.axvline(
    x=start,
    c="k",
    linestyle="--",
)
ax.axvline(
    x=stop,
    c="k",
    linestyle="--",
)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_1d_input.png"))


# -- zoomed in region
t_interp, _ = ffs_sample(T, N_FS, T_c, N_target)
t_interp = np.sort(t_interp)

idx = np.logical_and(sample_points >= start, sample_points <= stop)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(points, vals_ffs, label="pyffs.fs_interp", alpha=0.7)
# ax.plot(points, vals_sinc, label="sinc interp")
ax.plot(resampled_t, resampled_x, label="scipy.signal.resample", alpha=0.7)
ax.plot(t_interp, dirichlet(t_interp, T, T_c_diric, N_FS), label="ground truth", alpha=0.7)
ax.scatter(sample_points, diric_samples, label="available samples")
ax.set_xlabel("Time [s]")
ax.set_xlim([start, stop])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_1d.png"))

plt.show()
