import tqdm, argparse, os, pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

from dp_accounting.pld.common import DifferentialPrivacyParameters
from dp_accounting.pld.accountant import get_smallest_subsampled_gaussian_noise
from collections import defaultdict
from matplotlib.pyplot import cm

from plot_defaults import PLOT_PARAMS, PAPER_WIDTH, COLUMN_WIDTH


def wrap_defaultdict(instance, times):
    """Wrap an instance an arbitrary number of `times` to create nested defaultdict.

    source: https://stackoverflow.com/a/68397610
    """

    def _dd(x):
        return defaultdict(x.copy)

    dd = defaultdict(instance)
    for i in range(times - 1):
        dd = _dd(dd)

    return dd


def compute_and_store_sigmas(args):
    path_to_save_results = args.result_path
    target_epsilon = args.epsilon
    target_delta = args.delta
    num_comps = args.num_comps

    target_privacy_parameters = DifferentialPrivacyParameters(
        epsilon=target_epsilon, delta=target_delta
    )

    qs = np.logspace(-3, 0, 100)

    if os.path.exists(path_to_save_results):
        with open(path_to_save_results, "rb") as f:
            result_dict = pickle.load(f)
    else:
        result_dict = wrap_defaultdict(dict, 4)

    target_dict = result_dict[target_epsilon][target_delta][num_comps]

    for q in tqdm.tqdm(qs):
        if q not in target_dict.keys():
            sigma = get_smallest_subsampled_gaussian_noise(
                privacy_parameters=target_privacy_parameters,
                num_queries=num_comps,
                sensitivity=1.0,
                sampling_prob=q,
            )
            target_dict[q] = sigma

    result_dict[target_epsilon][target_delta][num_comps] = target_dict

    with open(path_to_save_results, "wb") as f:
        pickle.dump(result_dict, f)

    return target_dict


def plot_sigma_per_q(args):
    target_epsilons = list(map(float, args.list_of_epsilons))
    compositions = list(map(int, args.list_of_compositions))
    target_delta = args.delta

    if os.path.exists(args.result_path):
        with open(args.result_path, "rb") as f:
            result_dict = pickle.load(f)

    # plot
    plt.rcParams.update(PLOT_PARAMS)
    fig, axis = plt.subplots(
        ncols=len(target_epsilons),
        figsize=(PAPER_WIDTH, (6 / 16) * PAPER_WIDTH),
        sharex=True,
        sharey=True,
    )

    cmap = cm.rainbow(np.linspace(0, 1, len(compositions)))

    plt.setp(axis[0], ylabel="$\sigma_{\mathrm{eff}}$")
    plt.setp(axis[:], xlabel="$q$")
    pad = 5
    for i_eps, target_epsilon in enumerate(target_epsilons):
        ax = axis[i_eps]
        ax.set_title(f"$\epsilon={target_epsilon}$")

        for j_comp, num_comps in enumerate(compositions):
            data = result_dict[target_epsilon][target_delta][num_comps]
            data = pd.DataFrame.from_dict(data, orient="index").sort_index()
            if i_eps == 0:
                ax.plot(
                    data.index,
                    data.values.flatten() / data.index,
                    color=cmap[j_comp],
                    label=f"$T={num_comps}$",
                )
                ax.axhline(data.values[-1][0], ls="--", color=cmap[j_comp], label=f"$\sigma(q=1, T={num_comps})$")
            else:
                ax.plot(
                    data.index, data.values.flatten() / data.index, color=cmap[j_comp]
                )
                ax.axhline(data.values[-1][0], ls="--", color=cmap[j_comp])

        ax.set_yscale("log")
        ax.set_xscale("log")

    fig.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.22), ncols=len(compositions)
    )

    fig.savefig(
        args.figure_path + f"sigma_per_q_panel_delta{args.delta}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./results/")
    parser.add_argument("--delta", type=float, help="target delta")
    subparsers = parser.add_subparsers(required=True)

    #
    parser_compute = subparsers.add_parser("compute")
    parser_compute.add_argument("--epsilon", type=float, help="target epsilon")
    parser_compute.add_argument("--num_comps", type=int, help="number of compositions")
    parser_compute.set_defaults(func=compute_and_store_sigmas)

    #
    parser_plot = subparsers.add_parser("plot")
    parser_plot.add_argument(
        "-loe",
        "--list_of_epsilons",
        nargs="+",
        default=[],
        help="target epsilons to plot",
    )
    parser_plot.add_argument(
        "-loc",
        "--list_of_compositions",
        nargs="+",
        default=[],
        help="number of compositions to plot",
    )
    parser_plot.add_argument("--figure_path", type=str, default="./figures/")
    parser_plot.set_defaults(func=plot_sigma_per_q)

    args, _ = parser.parse_known_args()

    args.func(args)


if __name__ == "__main__":
    main()
