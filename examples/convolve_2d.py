import numpy as np
from pyffs import ffsn_sample, ffsn_shift
from pyffs.func import dirichlet_2D
from scipy.signal import convolve2d as convolve_scipy
from scipy.signal import fftconvolve
from pyffs.conv import convolve as convolve_fs
import matplotlib.pyplot as plt
import os
import util  # plot plotting


T = [1, 1]
T_c = [0.1, 0.2]
T_c_diric = 2 * [0.3]
N_FS = [15, 9]
N_samples = [128, 128]
reorder = True  # pass ordered samples to `convolve_fs` -> need to reorder samples inside
y_val_plot = 2 * T_c_diric[1]

# input, which we convolve with itself
sample_points, idx = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_samples)
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
diric_samples_ord = ffsn_shift(diric_samples, idx)


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
idx_ffs = np.argmin(np.abs(np.squeeze(sample_points[1]) - y_val_plot))
idx_fft = np.argmin(np.abs(output_vals_y - y_val_plot))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    sample_points[0][idx_x],
    np.real(output_samples[idx_x, idx_ffs]),
    label="pyffs.convolve2d",
    alpha=0.7,
)
# ax.plot(output_vals_x, np.real(output_scipy[:, idx_fft]), label="scipy.signal.convolve2d", alpha=0.7)
ax.plot(
    output_vals_x,
    np.real(output_scipy_wrap[:, idx_fft]),
    label="scipy.signal.convolve2d (wrap)",
    alpha=0.7,
)
ax.plot(
    output_vals_x,
    np.real(output_fftconvolve[:, idx_fft]),
    label="scipy.signal.fftconvolve",
    alpha=0.7,
)
# ax.set_title("convolution of 2D bandlimited periodic functions, y={}".format(y_val_plot))
ax.set_xlabel("x [m]")
ax.set_xlim([np.min(sample_points[0]), np.max(sample_points[0])])
plt.legend()
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_output_slice.png")
)

# --- 2D plots
pcolormesh = True

# input
ax = util.plot2d(
    x_vals=sample_points_x[idx_x],
    y_vals=sample_points_y[idx_y],
    Z=np.real(diric_samples_ord),
    pcolormesh=pcolormesh,
    colorbar=False,
)
plt.tight_layout()
# ax.set_title("input")
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_input.png")
)

# output
ax = util.plot2d(
    x_vals=sample_points_x[idx_x],
    y_vals=sample_points_y[idx_y],
    Z=np.real(ffsn_shift(output_samples, idx)),
    pcolormesh=pcolormesh,
    colorbar=False,
)
plt.tight_layout()
# ax.set_title("FFS convolve")
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_ffsconvolve.png")
)

# # output
# ax = util.plot2d(
#     x_vals=output_vals_x,
#     y_vals=output_vals_y,
#     Z=np.real(output_scipy),
#     pcolormesh=pcolormesh,
#     colorbar=False,
# )
# ax.set_xlim([np.min(sample_points[0]), np.max(sample_points[0])])
# ax.set_ylim([np.min(sample_points[1]), np.max(sample_points[1])])
# plt.tight_layout()
# # ax.set_title("scipy.signal.convolve2d")
# plt.savefig(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_fftconvolve.png")
# )


# output
ax = util.plot2d(
    x_vals=output_vals_x,
    y_vals=output_vals_y,
    Z=np.real(output_scipy),
    pcolormesh=pcolormesh,
    colorbar=False,
)
ax.set_xlim([np.min(sample_points[0]), np.max(sample_points[0])])
ax.set_ylim([np.min(sample_points[1]), np.max(sample_points[1])])
plt.tight_layout()
# ax.set_title("scipy.signal.fftconvolve")
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_2d_fftconvolve.png")
)

plt.show()
