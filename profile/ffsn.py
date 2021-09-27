import pathlib as plib
import time
import click
import matplotlib.pyplot as plt
import numpy as np
from util import comparison_plot, plotting_setup, backend_to_label
from pyffs import ffsn_sample, ffsn, _ffsn, next_fast_len
from pyffs.func import dirichlet_2D
from pyffs.backend import AVAILABLE_MOD, get_module_name


@click.command()
@click.option("--n_trials", type=int, default=10)
def profile_ffsn(n_trials):
    print(f"\nCOMPARING FFSN APPROACHES WITH {n_trials} TRIALS")

    fig_path = plotting_setup(linewidth=3, font_size=20)

    T = [1, 1]
    T_c = [0, 0]
    N_FS_vals = [11, 31, 101, 301, 1001, 3001, 10001]

    n_std = 1

    # func = {"ffsn_fftn": ffsn, "ffsn_ref": _ffsn}
    func = {"ffsn_fftn": ffsn}

    proc_time = dict()
    proc_time_std = dict()

    for _N_FS in N_FS_vals:
        print("\nN_FS : {}".format(_N_FS))
        N_FS = [_N_FS, _N_FS]
        proc_time[_N_FS] = dict()
        proc_time_std[_N_FS] = dict()

        # Loop through modules
        for mod in AVAILABLE_MOD:

            backend = backend_to_label[get_module_name(mod)]

            # fastest FFT length depends on module
            _N_s = next_fast_len(_N_FS, mod=mod)
            N_s = [_N_s, _N_s]
            print("-- module : {}, Length-{} FFT".format(backend, N_s))

            for _f in func:

                # Sample the kernel and do the transform.
                sample_points, _ = ffsn_sample(T=T, N_FS=N_FS, T_c=T_c, N_s=N_s, mod=mod)
                diric_samples = dirichlet_2D(sample_points=sample_points, T=T, T_c=T_c, N_FS=N_FS)
                diric_samples = diric_samples.astype(
                    "float32"
                )  # cast to float32, theoretically better for GPU
                func[_f](
                    x=diric_samples, T=T, N_FS=N_FS, T_c=T_c
                )  # first one is a bit slow sometimes...
                if len(func.keys()) > 1:
                    _key = "{}_{}".format(_f, backend)
                else:
                    _key = backend
                timings = []
                for _ in range(n_trials):
                    start_time = time.time()
                    func[_f](x=diric_samples, T=T, N_FS=N_FS, T_c=T_c)
                    timings.append(time.time() - start_time)
                proc_time[_N_FS][_key] = np.mean(timings)
                proc_time_std[_N_FS][_key] = np.std(timings)

                print("{} : {} seconds".format(_f, proc_time[_N_FS][_key]))

    # plot results
    fig, ax = plt.subplots()
    comparison_plot(proc_time, proc_time_std, n_std, ax)
    ax.set_xlabel("Number of FS coefficients")
    ax.set_xticks(np.array(N_FS_vals) - 1)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    fig.tight_layout()
    fig.savefig(plib.Path(fig_path) / "profile_ffsn_2d.png")

    plt.show()


if __name__ == "__main__":
    profile_ffsn()
