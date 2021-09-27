import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib as plib


def plotting_setup(font_size=30, linewidth=4, markersize=10, fig_folder="figs"):
    font = {"family": "Times New Roman", "weight": "normal", "size": font_size}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["lines.linewidth"] = linewidth
    matplotlib.rcParams["lines.markersize"] = markersize

    fig_path = plib.Path(__file__).parent / fig_folder
    fig_path.mkdir(exist_ok=True)
    return fig_path


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


def plot2d(x_vals, y_vals, Z, pcolormesh=True, colorbar=True):

    if pcolormesh:
        # define corners of mesh
        dx = x_vals[1] - x_vals[0]
        x_vals -= dx / 2
        x_vals = np.append(x_vals, [x_vals[-1] + dx])

        dy = y_vals[1] - y_vals[0]
        y_vals -= dy / 2
        y_vals = np.append(y_vals, [y_vals[-1] + dy])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X, Y = np.meshgrid(x_vals, y_vals)
    if pcolormesh:
        cp = ax.pcolormesh(X, Y, Z.T)
    else:
        cp = ax.contourf(X, Y, Z.T)
    if colorbar:
        fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax
