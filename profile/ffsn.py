import pathlib
import time

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import next_fast_len

import util
from pyffs import ffsn_sample, ffsn, _ffsn
from pyffs.func import dirichlet_2D


@click.command()
@click.option("--n_trials", type=int, default=5)
def profile_ffsn(n_trials):
    print(f"\nCOMPARING FFSN APPROACHES WITH {n_trials} TRIALS")

    T = [1, 1]
    T_c = [0, 0]
    N_FS_vals = [101, 301, 1001, 3001, 10001]

    n_std = 1

    func = {"ffsn_fftn": ffsn, "ffsn_ref": _ffsn}

    proc_time = dict()
    proc_time_std = dict()
    for _N_FS in N_FS_vals:

        N_FS = [_N_FS, _N_FS]
        _N_s = next_fast_len(_N_FS)
        N_s = [_N_s, _N_s]

        print("\nN_FS : {}".format(_N_FS))
        print("N_s : {}".format(_N_s))
        proc_time[_N_FS] = dict()
        proc_time_std[_N_FS] = dict()

        # Sample the kernel and do the transform.
        sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s)
        diric_samples = dirichlet_2D(sample_points=sample_points, T=T, T_c=T_c, N_FS=N_FS)

        # Loop through functions
        for _f in func:
            timings = []
            for _ in range(n_trials):
                start_time = time.time()
                func[_f](x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)
                timings.append(time.time() - start_time)
            proc_time[_N_FS][_f] = np.mean(timings)
            proc_time_std[_N_FS][_f] = np.std(timings)

            print("{} : {} seconds".format(_f, proc_time[_N_FS][_f]))

    # plot results
    fig, ax = plt.subplots()
    util.comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_xlabel("Number of FS coefficients")
    fig.tight_layout()

    fname = pathlib.Path(__file__).resolve().parent / "ffsn_comparison.png"
    fig.savefig(fname, dpi=300)


if __name__ == "__main__":
    profile_ffsn()
