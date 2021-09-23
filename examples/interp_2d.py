import numpy as np
from pyffs import ffsn_sample, ffsn, fs_interpn
from pyffs.func import dirichlet_2D
import matplotlib
from scipy.interpolate import interp2d
from scipy.signal import resample
import matplotlib.pyplot as plt
import os
import util

font = {"family": "Times New Roman", "weight": "normal", "size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["lines.linewidth"] = 4
matplotlib.rcParams["lines.markersize"] = 10


def fft2_interpolate(samples, T, dx, dy, T_c, offset=None):
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
    if offset is not None:
        phase_x = np.exp(-1j * 2 * np.pi * np.fft.fftfreq(X_pad.shape[0], dx) * offset[0])[
            :, np.newaxis
        ]
        phase_y = np.exp(-1j * 2 * np.pi * np.fft.fftfreq(X_pad.shape[1], dy) * offset[1])[
            np.newaxis, :
        ]
        X_pad *= phase_x * phase_y
    vals_fft = (
        np.real(np.fft.ifft2(X_pad))
        * X_pad.shape[0]
        * X_pad.shape[1]
        / samples.shape[0]
        / samples.shape[1]
    )
    x_fft = np.linspace(
        start=T_c[0] - T[0] / 2, stop=T_c[0] + T[0] / 2, num=vals_fft.shape[0], endpoint=True
    )
    y_fft = np.linspace(
        start=T_c[1] - T[1] / 2, stop=T_c[1] + T[1] / 2, num=vals_fft.shape[1], endpoint=True
    )

    return vals_fft, x_fft, y_fft


# signal parameters
T = 2 * [1]
T_c = 2 * [0]
T_c_diric = 2 * [0.3]
N_FS = 2 * [31]
N_s = 2 * [32]

# specify region to zoom into
start = [T_c_diric[0] - 0.05, T_c_diric[1] - 0.05]
stop = [T_c_diric[0] + 0.15, T_c_diric[1] + 0.15]
y_val = T_c_diric[1]  # cross-section plot (something within zoomed region)
num = 2 * [128]  # number of interpolation points

# sample function
sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
diric_samples = dirichlet_2D(sample_points, T, T_c_diric, N_FS)
diric_FS = ffsn(x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)[: N_FS[0], : N_FS[1]]

# interpolation points
points_x = np.linspace(start[0], stop[0], num[0], endpoint=True)
dx = points_x[1] - points_x[0]
points_y = np.linspace(start[1], stop[1], num[1], endpoint=True)
dy = points_y[1] - points_y[0]

# interpolate FFS
vals_ffs = fs_interpn(diric_FS, T=T, a=start, b=stop, M=num, real_x=True)

# interpolate with FFT and zero-padding
samples_ord = dirichlet_2D(
    [np.sort(sample_points[0], axis=0), np.sort(sample_points[1])], T, T_c_diric, N_FS
)
offset_x = np.min(np.abs(sample_points[0]))  # offset from `ffsn_sample`
offset_y = np.min(np.abs(sample_points[1]))  # offset from `ffsn_sample`
vals_fft, x_fft, y_fft = fft2_interpolate(samples_ord, T, dx, dy, T_c, offset=[offset_x, offset_y])


Nx_target = int(np.ceil(T[0] / dx))
sample_points_x = np.squeeze(np.sort(sample_points[0], axis=0))
Ny_target = int(np.ceil(T[1] / dy))
sample_points_y = np.squeeze(np.sort(sample_points[1], axis=1))
vals_resample, resampled_x = resample(samples_ord, Nx_target, t=sample_points_x, axis=0)
vals_resample, resampled_y = resample(vals_resample, Ny_target, t=sample_points_y, axis=1)


# interpolate with scipy
f_linear = interp2d(x=sample_points[0], y=sample_points[1], z=diric_samples.T, kind="linear")
vals_linear = f_linear(points_x, points_y).T
f_cubic = interp2d(x=sample_points[0], y=sample_points[1], z=diric_samples.T, kind="cubic")
vals_cubic = f_cubic(points_x, points_y).T

# ground truth
sample_points_interp = [points_x[:, np.newaxis], points_y[np.newaxis, :]]
vals_true = dirichlet_2D(sample_points_interp, T, T_c_diric, N_FS)


# -- plot cross section
idx_yc = np.argmin(np.abs(points_y - y_val))
idx_og = np.argmin(np.abs(np.squeeze(sample_points[1]) - y_val))
idx_fft = np.argmin(np.abs(y_fft - y_val))
idx_resample = np.argmin(np.abs(resampled_y - y_val))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(points_x, np.real(vals_ffs[:, idx_yc]), label="pyffs.fs_interp", alpha=0.7)
ax.plot(
    resampled_x, np.real(vals_resample[:, idx_resample]), label="scipy.signal.resample", alpha=0.7
)
ax.plot(points_x, np.real(vals_true[:, idx_yc]), label="ground truth", alpha=0.7)
ax.scatter(sample_points[0], diric_samples[:, idx_og], label="available samples")
ax.set_xlabel("x [m]")
ax.set_xlim([start[0], stop[0]])
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_output_slice.png")
)

# --- 2D plots
pcolormesh = True
x_shift = (sample_points[0][1, 0] - sample_points[0][0, 0]) / 2
y_shift = (sample_points[1][0, 1] - sample_points[1][0, 0]) / 2

# input
ax = util.plot2d(
    x_vals=np.sort(np.squeeze(sample_points[0])),
    y_vals=np.sort(np.squeeze(sample_points[1])),
    Z=np.real(np.fft.ifftshift(diric_samples)),
    pcolormesh=pcolormesh,
    colorbar=False,
)
# -- zoomed in region
zoom_color = "r"
ax.axvline(
    x=start[0] - x_shift * pcolormesh,
    ymin=0.5 + start[1] - y_shift * pcolormesh,
    ymax=0.5 + stop[1] - y_shift * pcolormesh,
    c=zoom_color,
    linestyle="--",
)
ax.axvline(
    x=stop[0] - x_shift * pcolormesh,
    ymin=0.5 + start[1] - y_shift * pcolormesh,
    ymax=0.5 + stop[1] - y_shift * pcolormesh,
    c=zoom_color,
    linestyle="--",
)
ax.axhline(
    y=start[1] - y_shift * pcolormesh,
    xmin=0.5 + start[0] - x_shift * pcolormesh,
    xmax=0.5 + stop[0] - x_shift * pcolormesh,
    c=zoom_color,
    linestyle="--",
)
ax.axhline(
    y=stop[1] - y_shift * pcolormesh,
    xmin=0.5 + start[0] - x_shift * pcolormesh,
    xmax=0.5 + stop[0] - x_shift * pcolormesh,
    c=zoom_color,
    linestyle="--",
)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_input.png"))

# FFS interp
ax = util.plot2d(
    x_vals=points_x,
    y_vals=points_y,
    Z=np.real(vals_ffs),
    pcolormesh=pcolormesh,
    colorbar=False,
)
# -- cross-section
ax.axhline(y=y_val, c="r", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_ffs.png"))

plt.show()
