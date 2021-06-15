import numpy as np
from scipy.signal import convolve as convolve_scipy
from scipy.signal import fftconvolve
from pyffs import ffs_sample
from pyffs.func import dirichlet
import matplotlib.pyplot as plt
from pyffs.conv import convolve as convolve_fs
import os
import util  # for plotting


T, T_c, N_FS, N_samples = 1, 0.1, 15, 512
T_c_diric = 0.25
reorder = True  # pass ordered samples to `convolve_fs` -> need to reorder samples inside

# input, which we convolve with itself
sample_points, idx = ffs_sample(T, N_FS, T_c, N_samples)
if reorder:
    sample_points = np.sort(sample_points)
    idx = np.arange(len(sample_points)).astype(int)
diric_samples = dirichlet(sample_points, T, T_c_diric, N_FS)

# FFS convolution
output_samples = convolve_fs(
    f=diric_samples, h=diric_samples, T=T, T_c=T_c, N_FS=N_FS, reorder=reorder
)

# classic convolution with FFT (scipy)
output_fft = (
    convolve_scipy(diric_samples[idx], diric_samples[idx], mode="full", method="fft") / N_samples
)
output_fftconvolve = fftconvolve(diric_samples[idx], diric_samples[idx], mode="full") / N_samples
t_vals_full = np.linspace(2 * np.min(sample_points), 2 * np.max(sample_points), num=len(output_fft))

# plot
_, ax = plt.subplots(
    nrows=2, ncols=1, num="Convolve bandlimited, periodic signals", figsize=(10, 10)
)
ax[0].plot(sample_points[idx], diric_samples[idx])
ax[0].set_xlim([np.min(sample_points), np.max(sample_points)])
ax[0].set_ylabel("$f$")

ax[1].plot(sample_points[idx], np.real(output_samples[idx]), label="pyffs.convolve", alpha=0.7)
# ax[1].plot(t_vals_full, np.real(output_fft), label="scipy.signal.convolve", alpha=0.7)
ax[1].plot(t_vals_full, np.real(output_fftconvolve), label="scipy.signal.fftconvolve", alpha=0.7)
ax[1].set_xlim([np.min(sample_points), np.max(sample_points)])
ax[1].set_ylabel("$f \\ast f$")
ax[1].set_xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "convolve_1d.png"))

plt.show()
