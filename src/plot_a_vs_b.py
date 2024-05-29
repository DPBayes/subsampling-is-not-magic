import tqdm, pickle, argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.special import erf, erfc

from plot_defaults import PLOT_PARAMS, PAPER_WIDTH, COLUMN_WIDTH

SQRT2 = np.sqrt(2.0)
inv_SQRT2 = 1.0 / SQRT2
DELTA_MULTIPLIER = 4.0


def find_no_subsampling_sigma(epsilon, delta):
    def error(proposed_sigma):
        delta_ = single_step_subsampled_delta(epsilon, 1.0, proposed_sigma)
        return (delta - delta_) ** 2

    from scipy.optimize import fsolve

    init_sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    sigma = fsolve(error, 0.5 * init_sigma)[0]

    return sigma, single_step_subsampled_delta(epsilon, 1.0, sigma)


def single_step_subsampled_delta(epsilon, q, sigma):
    common_arg_remove = sigma * np.log(np.exp(epsilon) - (1.0 - q)) - sigma * np.log(q)
    term0_remove = stats.norm(0.0, 1.0).sf(common_arg_remove - 0.5 / sigma)
    term1_remove = stats.norm(0.0, 1.0).sf(common_arg_remove + 0.5 / sigma)
    term2_remove = stats.norm(0.0, 1.0).sf(common_arg_remove + 0.5 / sigma)
    remove_only = (
        q * term0_remove + (1.0 - q) * term1_remove - np.exp(epsilon) * term2_remove
    )
    return remove_only


def derf(x):
    return 2.0 / np.sqrt(np.pi) * np.exp(-(x**2))


def dsigma_mathematica(epsilon, q, sigma):
    a = 1.0 / (2.0 * SQRT2 * sigma)
    b = sigma / SQRT2 * (np.log(np.exp(epsilon) - (1.0 - q)) - np.log(q))
    return sigma / (q * 2 * a * derf(a - b)) * (erf(a + b) + erf(a - b))


def a(epsilon, q, sigma):
    return 1 / (2.0 * np.sqrt(2.0)) / sigma


def b(epsilon, q, sigma):
    return sigma / np.sqrt(2.0) * np.log(1 + (np.exp(epsilon) - 1) / q)


def compute_diff(args):
    n_qs = 100
    n_sigmas = 10000
    n_epsilons = 1000
    target_delta = args.delta

    target_epsilons = np.logspace(
        np.log10(DELTA_MULTIPLIER * target_delta), np.log10(4.0), n_epsilons
    )

    largest_a_minus_b = np.zeros(len(target_epsilons))
    largest_a_minus_b_args = []

    qs = np.logspace(np.log10(DELTA_MULTIPLIER * target_delta), np.log10(0.9999), n_qs)
    qs_xpnd = qs_xpnd = np.repeat(qs.reshape(-1, 1), n_sigmas, axis=1)

    problematic_parameters = []

    i_eps = 0
    max_delta_difference = 0.0

    for target_epsilon in tqdm.tqdm(target_epsilons):
        # form grid of sigmas
        sigma1, delta1 = find_no_subsampling_sigma(target_epsilon, target_delta)
        while delta1 > target_delta:
            sigma1 = 1.00001 * sigma1
            delta1 = single_step_subsampled_delta(target_epsilon, 1.0, sigma1)
        sigmas = np.logspace(np.log10(qs[0] * sigma1), np.log10(sigma1), n_sigmas)
        sigmas_xpnd = np.repeat(sigmas.reshape(1, -1), n_qs, axis=0)

        # compute a, b and delta values
        a_array = a(target_epsilon, qs_xpnd, sigmas_xpnd)
        b_array = b(target_epsilon, qs_xpnd, sigmas_xpnd)
        delta_array = single_step_subsampled_delta(target_epsilon, qs_xpnd, sigmas_xpnd)

        a_gt_b = a_array > b_array
        delta_small_enough = delta_array <= target_delta

        problematic_indices = np.where(delta_small_enough * a_gt_b)
        if len(problematic_indices[0]):
            problematic_parameters.append(
                (
                    target_epsilon,
                    qs[problematic_indices[0]],
                    sigmas[problematic_indices[1]],
                )
            )

        largest_a_minus_b[i_eps] = (a_array - b_array)[delta_small_enough].max()
        largest_a_minus_b_indx = np.where(a_array - b_array == largest_a_minus_b[i_eps])
        largest_a_minus_b_args.append(
            (qs[largest_a_minus_b_indx[0][0]], sigmas[largest_a_minus_b_indx[1][0]])
        )

        i_eps += 1
        # compute the largest delta diff
        max_delta_difference = max(
            np.min(
                np.abs(delta_array * delta_small_enough - target_delta), axis=1
            ).max(),
            max_delta_difference,
        )
        if max_delta_difference == target_delta:
            assert 0

    # store results
    results = {
        "epsilons": target_epsilons,
        "ab_diffs": largest_a_minus_b,
        "max_delta_difference": max_delta_difference,
    }
    print(f"Max delta error: {results['max_delta_difference']}")

    with open(args.result_path + "a_minus_b_results.p", "wb") as f:
        pickle.dump(results, f)


def plot_diff(args):
    with open(args.result_path + "a_minus_b_results.p", "rb") as f:
        results = pickle.load(f)
    print(f"Max delta error: {results['max_delta_difference']}")

    plt.rcParams.update(PLOT_PARAMS)

    fig, ax = plt.subplots(figsize=2 * [COLUMN_WIDTH])

    plt.semilogx(results["epsilons"], results["ab_diffs"])
    plt.ylim(None, 0.0)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$a-b$")
    plt.savefig(
        args.figure_path + "a_minus_b_vs_eps.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    parser.add_argument("--result_path", type=str, default="./results/")

    #
    parser_compute = subparsers.add_parser("compute")
    parser_compute.add_argument(
        "--delta", type=float, help="target delta", default=1e-5
    )
    parser_compute.set_defaults(func=compute_diff)

    #
    parser_plot = subparsers.add_parser("plot")
    parser_plot.add_argument("--figure_path", type=str, default="./figures/")
    parser_plot.set_defaults(func=plot_diff)

    args, _ = parser.parse_known_args()

    args.func(args)


if __name__ == "__main__":
    main()
