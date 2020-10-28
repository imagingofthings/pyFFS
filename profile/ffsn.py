import time
import numpy as np
from pyffs import ffsn_sample, ffsn, ffsn_comp
from pyffs.func import dirichlet_2D
import matplotlib.pyplot as plt
import click


@click.command()
@click.option("--n_trials", type=int, default=5)
def profile_ffsn(n_trials):
    print(f"\nCOMPARING FFSN APPROACHES WITH {n_trials} TRIALS")

    T = [1, 1]
    T_c = [0, 0]
    N_FS_vals = [101, 301, 1001, 3001]

    n_std = 1

    func = {"ffsn_fft": ffsn, "ffsn_comp": ffsn_comp}

    proc_time = dict()
    proc_time_std = dict()
    for _N_FS in N_FS_vals:

        N_FS = [_N_FS, _N_FS]
        N_s = [_N_FS, _N_FS]

        print("\nN_FS : {}".format(_N_FS))
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
                func[_f](Phi=diric_samples, T=T, N_FS=N_FS, T_c=T_c)
                timings.append(time.time() - start_time)
            proc_time[_N_FS][_f] = np.mean(timings)
            proc_time_std[_N_FS][_f] = np.std(timings)

            print("{} : {} seconds".format(_f, proc_time[_N_FS][_f]))

    # plot results
    markers = ["o", "^", "v", "x", ">", "<", "D", "+"]
    plt.figure()
    for i, _f in enumerate(func):
        _proc_time = []
        _proc_time_std = []
        for n_rays in N_FS_vals:
            _proc_time.append(proc_time[n_rays][_f])
            _proc_time_std.append(proc_time_std[n_rays][_f])
        _proc_time = np.array(_proc_time)
        _proc_time_std = np.array(_proc_time_std)

        plt.loglog(N_FS_vals, _proc_time, label=_f, marker=markers[i])
        ax = plt.gca()
        ax.fill_between(
            N_FS_vals,
            (_proc_time - n_std * _proc_time_std),
            (_proc_time + n_std * _proc_time_std),
            alpha=0.2,
        )

    plt.legend()
    plt.xlabel("Number of FS coefficients")
    plt.ylabel("Processing time (s)")
    plt.grid()
    plt.tight_layout()
    ax = plt.gca()
    ax.set_xticks(N_FS_vals)
    plt.savefig("ffsn_comparison.png")


if __name__ == "__main__":
    profile_ffsn()
