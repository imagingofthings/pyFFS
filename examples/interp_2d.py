import numpy as np
from pyffs import ffsn_sample, ffsn, fs_interpn
from pyffs.func import dirichlet_2D
import matplotlib
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import os

font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)


def fft2_interpolate(samples, T, dx, dy, T_c):
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


start = [-0.1, -0.1]
stop = [0.1, 0.1]
num = [128, 128]

T = [1, 1]
T_c = [0, 0]
N_FS = [61, 61]
N_s = [128, 128]

sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
diric_samples = dirichlet_2D(sample_points, T, T_c, N_FS)
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
    [np.sort(sample_points[0], axis=0), np.sort(sample_points[1])], T, T_c, N_FS
)
vals_fft, x_fft, y_fft = fft2_interpolate(samples_ord, T, dx, dy, T_c)

# interpolate with scipy
f_linear = interp2d(x=sample_points[0], y=sample_points[1], z=diric_samples.T, kind="linear")
vals_linear = f_linear(points_x, points_y).T
f_cubic = interp2d(x=sample_points[0], y=sample_points[1], z=diric_samples.T, kind="cubic")
vals_cubic = f_cubic(points_x, points_y).T

# ground truth
sample_points_interp = [points_x[:, np.newaxis], points_y[np.newaxis, :]]
vals_true = dirichlet_2D(sample_points_interp, T, T_c, N_FS)


# -- plot cross section
idx_yc = np.argmin(np.abs(points_y - T_c[1]))
idx_og = np.argmin(np.abs(np.squeeze(sample_points[1]) - T_c[1]))
idx_fft = np.argmin(np.abs(y_fft - T_c[1]))
# idx_x = np.squeeze(np.argsort(sample_points[0], axis=0))

#
# import pudb; pudb.set_trace()

# idx_fft = np.argmin(np.abs(points_y - T_c[1]))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(points_x, np.real(vals_ffs[:, idx_yc]), label="FFS")
ax.plot(points_x, np.real(vals_linear[:, idx_yc]), label="linear")
ax.plot(points_x, np.real(vals_cubic[:, idx_yc]), label="cubic")
ax.plot(points_x, np.real(vals_true[:, idx_yc]), label="true")
ax.plot(x_fft, np.real(vals_fft[:, idx_fft]), label="FFT")
ax.scatter(sample_points[0], diric_samples[:, idx_og], label="available")
ax.set_xlabel("x [m]")
ax.set_xlim([start[0], stop[0]])
plt.legend()
plt.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_output_slice.png")
)

# --- 2D plots

# input
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X, Y = np.meshgrid(np.fft.ifftshift(sample_points[0]), np.fft.ifftshift(sample_points[1]))
cp = ax.contourf(X, Y, np.real(np.fft.ifftshift(diric_samples).T))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("input")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_input.png"))

# FFS interp
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X, Y = np.meshgrid(points_x, points_y)
cp = ax.contourf(X, Y, np.real(vals_ffs.T))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("FS interp")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_ffs.png"))

# FFT interp
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X, Y = np.meshgrid(x_fft, y_fft)
cp = ax.contourf(X, Y, np.real(vals_fft.T))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("FFT interp")
ax.set_xlim([start[0], stop[0]])
ax.set_ylim([start[1], stop[1]])
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_fft.png"))

# cubic interp
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X, Y = np.meshgrid(points_x, points_y)
cp = ax.contourf(X, Y, np.real(vals_cubic.T))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("cubic interp")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_cubic.png"))

# ground truth values
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X, Y = np.meshgrid(points_x, points_y)
cp = ax.contourf(X, Y, np.real(vals_true.T))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("true vals")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "interp_2d_true.png"))

plt.show()
