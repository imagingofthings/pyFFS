import numpy as np
from scipy.signal import convolve as convolve_scipy
from pyffs import ffs_sample
from pyffs.func import dirichlet
import matplotlib
import matplotlib.pyplot as plt
from pyffs.conv import convolve as convolve_fs
import os

font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)


T, T_c, N_FS, N_samples = 1, 0.2, 15, 512

# input, which we convolve with itself
sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
diric_samples = dirichlet(sample_points, T, T_c, N_FS)

# FFS convolution
output_samples = convolve_fs(f=diric_samples, h=diric_samples, T=T, T_c=T_c, N_FS=N_FS)

# classic convolution with FFT (scipy)
idx = np.argsort(sample_points)
diric_samples_ord = diric_samples[idx]
output_fft = (
    convolve_scipy(diric_samples_ord, diric_samples_ord, mode="full", method="fft") / N_samples
)
t_vals_full = np.linspace(2 * np.min(sample_points), 2 * np.max(sample_points), num=len(output_fft))

# plot
_, ax = plt.subplots(
    nrows=3, ncols=1, num="Convolve bandlimited, periodic signals", figsize=(10, 10)
)
ax[0].plot(sample_points[idx], diric_samples_ord)
ax[0].set_xlim([np.min(sample_points), np.max(sample_points)])
ax[0].set_ylabel("$f$")

ax[1].plot(sample_points[idx], diric_samples_ord)
ax[1].set_xlim([np.min(sample_points), np.max(sample_points)])
ax[1].set_ylabel("$h$")

ax[2].plot(sample_points[idx], np.real(output_samples[idx]), label="FFS conv", alpha=0.7)
ax[2].plot(t_vals_full, np.real(output_fft), label="FFT conv", alpha=0.7)
ax[2].set_xlim([np.min(sample_points), np.max(sample_points)])
ax[2].set_ylabel("$f \\ast h$")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_1d.png"))

plt.show()
