"""

Variance Explained Comparison of Zero-Inflated Poisson Tensor Factorization


This script compares the variance explained by the Zero-Inflated Poisson Tensor Factorization (ZIPTF) model
with several baseline models on synthetic tensors. It measures the variance explained at different probabilities
of excess zeros and generates a line plot for visualization.

Usage:
python experiment_1a.py [-h] [-s SEED] [-t NUM_THREADS] [-nt NUM_TRIALS] [-ni MAX_NUM_ITERS]
                             [-n RUN_NAME] [-lr LEARNING_RATE] [-a ALPHA] [-b BETA] [-r RANK]
                             [-sh SHAPE] [-pz PROB_EXCESS_ZEROS]

Arguments:
  -h, --help            Show help message and exit.
  -s SEED, --seed SEED  Set the random seed. Default: 0.
  -t NUM_THREADS, --num-threads NUM_THREADS
                        Set the number of threads. Default: 8.
  -nt NUM_TRIALS, --num-trials NUM_TRIALS
                        Set the number of trials to run. Default: 20.
  -ni MAX_NUM_ITERS, --max-num-iters MAX_NUM_ITERS
                        Set the maximum number of iterations. Default: 1000.
  -n RUN_NAME, --run-name RUN_NAME
                        Set the file name prefix to save the results. Default: "exp_1a".
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Set the initial learning rate. Default: 0.01.
  -a ALPHA, --alpha ALPHA
                        Set the alpha parameter of the Gamma distribution for factors. Default: 3.0.
  -b BETA, --beta BETA  Set the beta parameter of the Gamma distribution for factors. Default: 0.3.
  -r RANK, --rank RANK  Set the rank of the tensor. Default: 9.
  -sh SHAPE [SHAPE ...], --shape SHAPE [SHAPE ...]
                        Set the shape of the tensor. Default: [10, 20, 300].
  -pz PROB_EXCESS_ZEROS, --prob-excess-zeros PROB_EXCESS_ZEROS
                        Set the probability of excess zeros. Default: [0.0, 0.2, 0.4, 0.6, 0.8].

Example:
python experiment_1a.py --seed 1 --num-threads 4 --num-trials 10 --max-num-iters 500
                             --run-name "exp_2" --learning-rate 0.001 --alpha 2.5 --beta 0.2
                             --rank 10 --shape 20 30 200 --prob-excess-zeros 0.1 0.3 0.5

The script will generate results and save them in a CSV file with the specified run name and
create a line plot in PDF format to visualize the variance explained by each model.

"""

import tensorly as tl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import pyro
import argparse
import os

from scBTF import BayesianCP
from tqdm import trange
from tensorly.decomposition import non_negative_parafac_hals

from utils import variance_explained, generate_zip_tensor


def btf_factorize(observed_tensor, true_tensor, rank, alpha, beta, model, num_steps, initial_lr):
    """
    """
    bayesianCP = BayesianCP(
        dims=observed_tensor.shape, rank=rank, init_alpha=alpha, init_beta=beta, model=model
    )
    bayesianCP.fit(observed_tensor, num_steps=num_steps, initial_lr=initial_lr)
    prs = bayesianCP.precis(observed_tensor)
    a, b, c = [prs[f'factor_{i}']['mean'] for i in range(3)]

    tensor_means = torch.einsum('ir,jr,kr->ijk', a, b, c)
    return variance_explained(true_tensor, tensor_means)


def main(args):
    torch.set_num_threads(args.num_threads)

    results_dir = './results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    np.random.seed(args.seed)
    seeds = np.random.randint(0, np.iinfo(np.int32).max, args.num_trials)

    res = pd.DataFrame(columns = ['NNCP-ALS', 'TG-BTF', 'GP-BTF', 'ZIPTF', 'P_Excess_Zeros'])

    for i in trange(len(seeds)):
        seed = seeds[i]

        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        np.random.seed(seed)

        for p_zero in args.prob_excess_zeros:
            true_tensor, observed_tensor, _ = generate_zip_tensor(args.shape, args.rank, args.alpha, args.beta, p_zero)

            # NNCP-ALS
            fact = non_negative_parafac_hals(
                tensor=observed_tensor.numpy(),
                rank=args.rank,
                n_iter_max=args.max_num_iters,
                init='random'
            )
            var_exp_cp = variance_explained(true_tensor, tl.cp_to_tensor(fact))

            # TRUNCATED GAUSSIAN
            var_exp_tg = btf_factorize(
                observed_tensor=observed_tensor,
                true_tensor=true_tensor,
                rank=args.rank,
                model='truncated_gaussian_auto',
                num_steps=args.max_num_iters,
                initial_lr=args.learning_rate,
                alpha=args.alpha,
                beta=args.beta
            )

            # GAMMA POISSON
            var_exp_gp = btf_factorize(
                observed_tensor=observed_tensor,
                true_tensor=true_tensor,
                rank=args.rank,
                model='gamma_poisson',
                num_steps=args.max_num_iters,
                initial_lr=args.learning_rate,
                alpha=args.alpha,
                beta=args.beta
            )

            # ZERO-INFLATED POISSON
            var_exp_zip = btf_factorize(
                observed_tensor=observed_tensor,
                true_tensor=true_tensor,
                rank=args.rank,
                model='zero_inflated_poisson',
                num_steps=args.max_num_iters,
                initial_lr=args.learning_rate,
                alpha=args.alpha,
                beta=args.beta
            )

            res.loc[len(res.index)] = [var_exp_cp, var_exp_tg, var_exp_gp, var_exp_zip, p_zero]

    res['P_Excess_Zeros'] = res['P_Excess_Zeros'].round(2)
    res = res.melt(id_vars=['P_Excess_Zeros'], var_name='Method', value_name='Explained Variance')
    res['Explained Variance'] = res['Explained Variance'].map(lambda x: x.item())
    res.to_csv(f"{results_dir}/{args.run_name}.csv")

    fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=100)
    sns.set_theme(style="ticks")
    sns.lineplot(
        data=res, x="P_Excess_Zeros", y="Explained Variance", hue="Method", style="Method",
        ax=ax,
    )
    plt.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize='small')
    ax.set_xlabel('Probability of Excess Zeros')
    fig.savefig(f"{results_dir}/{args.run_name}.pdf", dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="compares variance explained of ZIPTF with baselines"
    )

    parser.add_argument("-s", "--seed", default=0, type=int, help="rng seed")
    parser.add_argument("-t", "--num-threads", default=8, type=int, help="number of threads")
    parser.add_argument("-nt", "--num-trials", default=20, type=int, help="number of trials to run")
    parser.add_argument("-ni", "--max-num-iters", default=1000, type=int, help="maximum number of iterations")
    parser.add_argument("-n", "--run-name", default="exp_1a", type=str, help="file name prefix to save results")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("-a", "--alpha", default=3.0, type=float, help="alpha param of Gamma distribution for factors")
    parser.add_argument("-b", "--beta", default=0.3, type=float, help="beta param of Gamma distribution for factors")
    parser.add_argument("-r", "--rank", default=9, type=int, help="rank of tensor")
    parser.add_argument(
        "-sh", "--shape",
        default=[10, 20, 300], nargs='+', type=int,
        help="shape of tensor"
    )
    parser.add_argument(
        "-pz", "--prob-excess-zeros",
        default=np.arange(0., 0.9, 0.2), nargs='+', type=float,
        help="probability of excess zeros"
    )

    args = parser.parse_args()
    main(args)