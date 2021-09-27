import numpy as np
from pyffs import ffsn_sample, ffsn, fs_interpn
from pyffs.func import dirichlet_2D
from scipy.interpolate import interp2d
from scipy.signal import resample
import matplotlib.pyplot as plt
import pathlib as plib
from util import plotting_setup, plot2d


fig_path = plotting_setup()
ALPHA = 0.8

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
sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s, mod=np)
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
idx_resample = np.argmin(np.abs(resampled_y - y_val))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(points_x, np.real(vals_true[:, idx_yc]), label="ground truth", alpha=ALPHA, linestyle="-")
ax.plot(
    points_x, np.real(vals_ffs[:, idx_yc]), label="pyffs.fs_interpn", alpha=ALPHA, linestyle="--"
)
ax.plot(
    resampled_x,
    np.real(vals_resample[:, idx_resample]),
    label="scipy.signal.resample",
    alpha=ALPHA,
    linestyle="-.",
)
ax.scatter(sample_points[0], diric_samples[:, idx_og], label="available samples")
ax.set_xlabel("x [m]")
ax.set_xlim([start[0], stop[0]])
plt.legend()
plt.tight_layout()
fig.savefig(plib.Path(fig_path) / "interp_2d_output_slice.png")

# --- 2D plots
pcolormesh = True
x_shift = (sample_points[0][1, 0] - sample_points[0][0, 0]) / 2
y_shift = (sample_points[1][0, 1] - sample_points[1][0, 0]) / 2

# input
ax = plot2d(
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

# -- cross-section
fig = plt.gcf()
fig.tight_layout()
fig.savefig(plib.Path(fig_path) / "interp_2d_input.png")

# FFS interp
ax = plot2d(
    x_vals=points_x, y_vals=points_y, Z=np.real(vals_ffs), pcolormesh=pcolormesh, colorbar=False,
)
# -- cross-section
ax.axhline(y=y_val, c="r", linestyle="--")
fig = plt.gcf()
fig.tight_layout()
fig.savefig(plib.Path(fig_path) / "interp_2d_ffs.png")

plt.show()
