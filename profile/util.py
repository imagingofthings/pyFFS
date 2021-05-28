import matplotlib.pyplot as plt
import numpy as np


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
