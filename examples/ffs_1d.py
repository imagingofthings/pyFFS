import math
import numpy as np
from pyffs import ffs_sample, ffs
from pyffs.func import dirichlet
import matplotlib
import matplotlib.pyplot as plt
import os


font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)


T, T_c, N_FS = math.pi, math.e, 15
N_samples = 512
sample_points, _ = ffs_sample(T, N_FS, T_c, N_samples)
diric_samples = dirichlet(sample_points, T, T_c, N_FS)
diric_FS = ffs(diric_samples, T, T_c, N_FS)[:N_FS]

# plot time
idx = np.argsort(sample_points)
_, ax = plt.subplots(nrows=2, num="FS coefficients of Dirichlet", figsize=(10, 10))
ax[0].plot(sample_points[idx], diric_samples[idx])
ax[0].grid()
ax[0].set_title("Dirichlet kernel and FS coefficients")
ax[0].set_xlabel("Time [s]")

# plot frequency
N = N_FS // 2
fs_idx = np.arange(-N, N + 1)
ax[1].stem(fs_idx, np.abs(diric_FS))
ax[1].grid()
ax[1].set_xlabel("FS index")

plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "ffs_1d.png"))
plt.show()
