import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pathlib as plib


backend_to_label = {"numpy": "CPU", "cupy": "GPU"}


def plotting_setup(font_size=30, linewidth=4, markersize=10, fig_folder="figs"):
    font = {"family": "Times New Roman", "weight": "normal", "size": font_size}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["lines.linewidth"] = linewidth
    matplotlib.rcParams["lines.markersize"] = markersize

    fig_path = plib.Path(__file__).parent / fig_folder
    fig_path.mkdir(exist_ok=True)
    return fig_path


def comparison_plot(proc_time, proc_time_std, n_std, ax=None):
    """
    Compare processing times of multiple approaches.

    Parameters
    ----------
    proc_time : dict[float, dict[str, float]]
        Average processing time for every combination of values and approaches. First set of keys
        specify the value being sweeped; next set of keys specify the different approaches.
    proc_time_std : dict[float, dict[str, float]]
        Standard deviation in processing time for every combination of values and approaches. First
        set of keys specify the  value being sweeped; next set of keys specify the different
        approaches.
    n_std : float
        Number of standard deviations to plot.
    ax : :py:class:`~matplotlib.axes.Axes`, optional
        `Axes` object to fill, default is to create one.

    Return
    ------
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    markers = ["o", "^", "v", "x", ">", "<", "D", "+"]

    if ax is None:
        _, ax = plt.subplots()

    x_vals = list(proc_time.keys())
    compare_vals = proc_time[x_vals[0]].keys()
    for i, _f in enumerate(compare_vals):
        _proc_time = []
        _proc_time_std = []
        for _val in x_vals:
            _proc_time.append(proc_time[_val][_f])
            _proc_time_std.append(proc_time_std[_val][_f])
        _proc_time = np.array(_proc_time)
        _proc_time_std = np.array(_proc_time_std)

        plt.loglog(x_vals, _proc_time, label=_f, marker=markers[i])
        ax.fill_between(
            x_vals,
            (_proc_time - n_std * _proc_time_std),
            (_proc_time + n_std * _proc_time_std),
            alpha=0.2,
        )
    ax.legend()
    ax.set_xticks(x_vals)
    ax.set_ylabel("Processing time (s)")
    ax.grid()

    return ax


def naive_interp1d(diric_FS, T, a, b, M):

    sample_points = np.linspace(start=a, stop=b, num=M, endpoint=False)

    # loop as could be large matrix
    N_FS = len(diric_FS)
    K = N_FS // 2
    fs_idx = np.arange(-K, K + 1)
    vals = np.zeros(len(sample_points), dtype=complex)
    for i, _x_val in enumerate(sample_points):
        vals[i] = np.dot(diric_FS, np.exp(1j * 2 * np.pi / T * _x_val * fs_idx))
    return vals


def naive_interp2d(diric_FS, T, a, b, M):

    # create sample points
    D = len(T)
    sample_points = []
    for d in range(D):
        sh = [1] * D
        sh[d] = M[d]
        sample_points.append(
            np.linspace(start=a[d], stop=b[d], num=M[d], endpoint=False).reshape(sh)
        )

    # initialize output
    x_vals = np.linspace(start=a[0], stop=b[0], num=M[0], endpoint=False)
    y_vals = np.linspace(start=a[1], stop=b[1], num=M[1], endpoint=False)
    output_shape = (len(x_vals), len(y_vals))
    vals = np.zeros(output_shape, dtype=complex)

    # loop to avoid creating potentially large matrices
    N_FSx, N_FSy = diric_FS.shape
    Kx = N_FSx // 2
    Ky = N_FSy // 2
    fsx_idx = np.arange(-Kx, Kx + 1)[:, np.newaxis]
    fsy_idx = np.arange(-Ky, Ky + 1)[np.newaxis, :]
    for i, _x_val in enumerate(x_vals):
        for j, _y_val in enumerate(y_vals):
            vals[i, j] = np.sum(
                diric_FS
                * np.exp(1j * 2 * np.pi * fsx_idx / T[0] * _x_val)
                * np.exp(1j * 2 * np.pi * fsy_idx / T[1] * _y_val)
            )
    return vals
