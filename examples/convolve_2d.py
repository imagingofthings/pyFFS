import numpy as np
from pyffs import ffsn_sample, iffs_shift
from pyffs.func import dirichlet_2D
from scipy.signal import convolve2d as convolve_scipy
from scipy.signal import fftconvolve
from pyffs.conv import convolve as convolve_fs
import matplotlib.pyplot as plt
import pathlib as plib
from util import plotting_setup, plot2d


fig_path = plotting_setup()
ALPHA = 0.9

T = [1, 1]
T_c = [0.1, 0.2]
T_c_diric = np.array([0.3, 0.3])
N_FS = [15, 9]
N_samples = [128, 128]
reorder = True  # pass ordered samples to `convolve_fs` -> need to reorder samples inside
y_val_plot = 2 * T_c_diric[1]

# input, which we convolve with itself
sample_points, idx = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_samples, mod=np)
if reorder:
    sample_points_x = np.sort(sample_points[0], axis=0)
    sample_points_y = np.sort(sample_points[1], axis=1)
    sample_points = [sample_points_x, sample_points_y]
    idx_x = np.arange(sample_points_x.shape[0]).astype(int)[:, np.newaxis]
    idx_y = np.arange(sample_points_y.shape[1]).astype(int)[np.newaxis, :]
    idx = [idx_x, idx_y]
diric_samples = dirichlet_2D(sample_points, T, T_c_diric, N_FS)

# FFS convolution
output_samples = convolve_fs(
    f=diric_samples, h=diric_samples, T=T, T_c=T_c, N_FS=N_FS, reorder=reorder
)

# classic convolution with FFT, scipy with linearized circular convolution
sample_points_x = np.squeeze(sample_points[0])
sample_points_y = np.squeeze(sample_points[1])
idx_x = np.squeeze(idx[0])
idx_y = np.squeeze(idx[1])
diric_samples_ord = iffs_shift(diric_samples) if not reorder else diric_samples

output_scipy = (
    convolve_scipy(diric_samples_ord, diric_samples_ord, mode="full", boundary="fill")
    / N_samples[0]
    / N_samples[1]
)
output_vals_x = np.linspace(
    2 * np.min(sample_points_x), 2 * np.max(sample_points_x), num=output_scipy.shape[0]
)
output_vals_y = np.linspace(
    2 * np.min(sample_points_y), 2 * np.max(sample_points_y), num=output_scipy.shape[1]
)

# scipy with circular boundary condition
output_scipy_wrap = (
    convolve_scipy(diric_samples_ord, diric_samples_ord, mode="full", boundary="wrap")
    / N_samples[0]
    / N_samples[1]
)

output_fftconvolve = (
    fftconvolve(diric_samples_ord, diric_samples_ord, mode="full") / N_samples[0] / N_samples[1]
)

# --- 1D plot,  2D cross section
diric_samples_true = dirichlet_2D(sample_points, T, 2 * T_c_diric, N_FS)

idx_ffs = np.argmin(np.abs(np.squeeze(sample_points[1]) - y_val_plot))
idx_fft = np.argmin(np.abs(output_vals_y - y_val_plot))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    sample_points[0][idx_x],
    np.real(diric_samples_true[idx_x, idx_ffs]),
    label="ground truth",
    alpha=ALPHA,
    linestyle="-",
)

ax.plot(
    sample_points[0][idx_x],
    np.real(output_samples[idx_x, idx_ffs]),
    label="pyffs.convolve2d",
    alpha=ALPHA,
    linestyle="--",
)
ax.plot(
    output_vals_x,
    np.real(output_scipy_wrap[:, idx_fft]),
    label="scipy.signal.convolve2d (wrap)",
    alpha=ALPHA,
    linestyle="dotted",
)
ax.plot(
    output_vals_x,
    np.real(output_fftconvolve[:, idx_fft]),
    label="scipy.signal.fftconvolve",
    alpha=ALPHA,
    linestyle="-.",
)
ax.set_xlabel("x [m]")
ax.set_xlim([np.min(sample_points[0]), np.max(sample_points[0])])
ax.legend()
fig.savefig(plib.Path(fig_path) / "convolve_2d_output_slice.png")

# --- 2D plots
pcolormesh = True

# input
ax = plot2d(
    x_vals=sample_points_x[idx_x],
    y_vals=sample_points_y[idx_y],
    Z=np.real(diric_samples_ord),
    pcolormesh=pcolormesh,
    colorbar=False,
)
fig = plt.gcf()
fig.tight_layout()
fig.savefig(plib.Path(fig_path) / "convolve_2d_input.png")

# output
ax = plot2d(
    x_vals=sample_points_x[idx_x],
    y_vals=sample_points_y[idx_y],
    Z=np.real(iffs_shift(output_samples)) if not reorder else np.real(output_samples),
    pcolormesh=pcolormesh,
    colorbar=False,
)
fig = plt.gcf()
fig.tight_layout()
fig.savefig(plib.Path(fig_path) / "convolve_2d_ffsconvolve.png")

# output
ax = plot2d(
    x_vals=output_vals_x,
    y_vals=output_vals_y,
    Z=np.real(output_scipy),
    pcolormesh=pcolormesh,
    colorbar=False,
)
ax.set_xlim([np.min(sample_points[0]), np.max(sample_points[0])])
ax.set_ylim([np.min(sample_points[1]), np.max(sample_points[1])])
fig = plt.gcf()
fig.tight_layout()
fig.savefig(plib.Path(fig_path) / "convolve_2d_fftconvolve.png")

plt.show()
