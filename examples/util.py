import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {"family": "Times New Roman", "weight": "normal", "size": 25}
matplotlib.rc("font", **font)
matplotlib.rcParams["lines.linewidth"] = 3


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
    fig = plt.gcf()
    if colorbar:
        fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax
