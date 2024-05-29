#!/bin/bash

mkdir -p results
mkdir -p figures

# converence panel
for EPSILON in 0.125 0.25 0.5 1.0 2.0 4.0
do
    poetry run python plot_convergence.py --result_path="./results/results_from_google_dp_accountant.p" --delta=1e-5 "compute" --epsilon=$EPSILON --num_comps=1
    poetry run python plot_convergence.py --result_path="./results/results_from_google_dp_accountant.p" --delta=1e-5 "compute" --epsilon=$EPSILON --num_comps=10
    poetry run python plot_convergence.py --result_path="./results/results_from_google_dp_accountant.p" --delta=1e-5 "compute" --epsilon=$EPSILON --num_comps=100
    poetry run python plot_convergence.py --result_path="./results/results_from_google_dp_accountant.p" --delta=1e-5 "compute" --epsilon=$EPSILON --num_comps=1000
done

poetry run python plot_convergence.py --result_path="./results/results_from_google_dp_accountant.p" --delta=1e-5 "plot" -loe 0.125 0.25 0.5 1.0 2.0 4.0 -loc 1 10 100 1000

# a minus b

poetry run python plot_a_vs_b.py --result_path=./results/ compute
poetry run python plot_a_vs_b.py --result_path=./results/ plot --figure_path=./figures/
