import numpy as np
from pyffs import ffsn_sample, ffs_shift
from pyffs.func import dirichlet_2D
from scipy.signal import convolve2d as convolve_scipy
from pyffs.conv import convolve2d as convolve_fs
import matplotlib
import matplotlib.pyplot as plt
import os

font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)

T = [1, 1]
T_c = [0.1, 0.2]
N_FS = [15, 9]
N_samples = [128, 128]
reorder = True  # pass ordered samples to `convolve_fs` -> need to reorder samples inside

# input, which we convolve with itself
sample_points, idx = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_samples)
if reorder:
    sample_points_x = np.sort(sample_points[0], axis=0)
    sample_points_y = np.sort(sample_points[1], axis=1)
    sample_points = [sample_points_x, sample_points_y]
    idx_x = np.arange(sample_points_x.shape[0]).astype(int)[:, np.newaxis]
    idx_y = np.arange(sample_points_y.shape[1]).astype(int)[np.newaxis, :]
    idx = [idx_x, idx_y]
diric_samples = dirichlet_2D(sample_points, T, T_c, N_FS)

# FFS convolution
output_samples = convolve_fs(
    f=diric_samples, h=diric_samples, T=T, T_c=T_c, N_FS=N_FS, reorder=reorder
)

# classic convolution with FFT (scipy)
sample_points_x = np.squeeze(sample_points[0])
sample_points_y = np.squeeze(sample_points[1])
idx_x = np.squeeze(idx[0])
idx_y = np.squeeze(idx[1])
diric_samples_ord = ffs_shift(diric_samples, idx)


output_fft = (
    convolve_scipy(diric_samples_ord, diric_samples_ord, mode="full") / N_samples[0] / N_samples[1]
)
output_vals_x = np.linspace(
    2 * np.min(sample_points_x), 2 * np.max(sample_points_x), num=output_fft.shape[0]
)
output_vals_y = np.linspace(
    2 * np.min(sample_points_y), 2 * np.max(sample_points_y), num=output_fft.shape[1]
)

# --- 1D plot,  2D cross section
idx_ffs = np.argmin(np.abs(np.squeeze(sample_points[1]) - 2 * T_c[1]))
idx_fft = np.argmin(np.abs(output_vals_y - 2 * T_c[1]))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(sample_points[0][idx_x], np.real(output_samples[idx_x, idx_ffs]), label="FFS")
ax.plot(output_vals_x, np.real(output_fft[:, idx_fft]), label="FFT")
ax.set_title("convolution of 2D bandlimited periodic functions, y={}".format(2 * T_c[1]))
ax.set_xlabel("x [m]")
plt.legend()
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_output_slice.png")
)

# --- 2D plots

# input
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X, Y = np.meshgrid(sample_points_x[idx_x], sample_points_y[idx_y])
cp = ax.contourf(X, Y, np.real(diric_samples_ord).T)
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("input")
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_input.png")
)

# output
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cp = ax.contourf(X, Y, np.real(ffs_shift(output_samples, idx)).T)
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("FFS convolve")
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_ffsconvolve.png")
)

# output
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X, Y = np.meshgrid(output_vals_x, output_vals_y)
cp = ax.contourf(X, Y, np.real(output_fft.T))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim([np.min(sample_points[0]), np.max(sample_points[0])])
ax.set_ylim([np.min(sample_points[1]), np.max(sample_points[1])])
ax.set_title("FFT convolve")
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_fftconvolve.png")
)

plt.show()
