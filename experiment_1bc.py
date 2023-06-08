"""

This script performs experiments to compare the accuracy and consistency of ZIPTF with C-ZIPTF. It runs each method on
The script generates boxplots and saves them as PDF files for visualization.

Usage:
python experiment_1bc.py [-h] [-s SEED] [-t NUM_THREADS] [-nt NUM_TRIALS] [-nr NUM_RESTARTS]
                                       [-pz PROB_EXCESS_ZEROS] [-ni MAX_NUM_ITERS] [-o RUN_NAME]
                                       [-lr LEARNING_RATE] [-a ALPHA] [-b BETA] [-r RANK] [-sh SHAPE [SHAPE ...]]

Arguments:
  -h, --help            Show help message and exit.
  -s SEED, --seed SEED  Set the random seed. Default: 0.
  -t NUM_THREADS, --num-threads NUM_THREADS
                        Set the number of threads. Default: 8.
  -nt NUM_TRIALS, --num-trials NUM_TRIALS
                        Set the number of trials to run. Default: 20.
  -nr NUM_RESTARTS, --num-restarts NUM_RESTARTS
                        Set the number of restarts to run for consensus aggregation. Default: 10.
  -pz PROB_EXCESS_ZEROS, --prob-excess-zeros PROB_EXCESS_ZEROS
                        Set the probability of excess zeros. Default: 0.6.
  -ni MAX_NUM_ITERS, --max-num-iters MAX_NUM_ITERS
                        Set the maximum number of iterations. Default: 1000.
  -o RUN_NAME, --run-name RUN_NAME
                        Set the file name prefix to save the results. Default: "exp_1bc".
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Set the initial learning rate. Default: 0.01.
  -a ALPHA, --alpha ALPHA
                        Set the alpha parameter of the Gamma distribution for factors. Default: 1.0.
  -b BETA, --beta BETA  Set the beta parameter of the Gamma distribution for factors. Default: 0.3.
  -r RANK, --rank RANK  Set the rank of the tensor. Default: 9.
  -sh SHAPE [SHAPE ...], --shape SHAPE [SHAPE ...]
                        Set the shape of the tensor. Default: [40, 20, 2000].

Example:
python experiment_1bc.py --seed 1 --num-threads 4 --num-trials 10 --num-restarts 5
                                       --prob-excess-zeros 0.8 --max-num-iters 500 --run-name "exp_2"
                                       --learning-rate 0.001 --alpha 2.0 --beta 0.5 --rank 10
                                       --shape 50 30 1000

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorly as tl

import argparse
import torch
import pyro

from tqdm import trange
from tensorly.cp_tensor import cp_normalize
from torch.distributions.gamma import Gamma
from tensorly.cp_tensor import CPTensor
from tensorly.metrics.factors import congruence_coefficient
from statannotations.Annotator import Annotator
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from scBTF import BayesianCP


def random_gamma(shape, alpha, beta=1.0):
    alpha = torch.ones(shape) * torch.tensor(alpha)
    beta = torch.ones(shape) * torch.tensor(beta)
    gamma_distribution = Gamma(alpha, beta)

    return gamma_distribution.sample()


def cosine_permute(ref_cp_tensor, tensors_to_permute):
    permuted_tensors = tensors_to_permute.cp_copy()
    ref_cp_tensor = cp_normalize(ref_cp_tensor)
    n_factors = len(ref_cp_tensor.factors)
    score, col = congruence_coefficient(
        ref_cp_tensor.factors, tensors_to_permute.factors
    )
    col = torch.tensor(col, dtype=torch.int64)
    for f in range(n_factors):
        permuted_tensors.factors[f] = permuted_tensors.factors[f][:, col]

    return score, col, permuted_tensors


def main(args):
    tl.set_backend('pytorch')

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    p_zero = args.prob_excess_zeros

    sim_repeats = args.num_trials
    n_restarts = args.num_restarts

    shape = args.shape
    rank = args.rank
    alpha = args.alpha
    beta = args.beta

    num_steps = args.max_num_iters

    A_factor_matrix = random_gamma(shape=(shape[0], rank), alpha=alpha, beta=beta)
    B_factor_matrix = random_gamma(shape=(shape[1], rank), alpha=alpha, beta=beta)
    C_factor_matrix = random_gamma(shape=(shape[2], rank), alpha=alpha, beta=beta)
    raw_tensor = torch.einsum('ir,jr,kr->ijk', A_factor_matrix, B_factor_matrix, C_factor_matrix)
    tensor = pyro.distributions.ZeroInflatedPoisson(rate=raw_tensor, gate=torch.tensor(p_zero)).sample()
    true = CPTensor((torch.ones(rank), [A_factor_matrix, B_factor_matrix, C_factor_matrix]))

    non_consensus = []
    for _ in trange(sim_repeats):
        bayesianCP = BayesianCP(dims=tensor.shape, rank=rank, init_alpha=70.0, model='zero_inflated_poisson')
        bayesianCP.fit(tensor, num_steps=num_steps)
        precis = bayesianCP.precis(tensor)
        non_consensus.append(CPTensor((torch.ones(rank), [precis[f'factor_{i}']['mean'] for i in range(3)])))

    factorization_sets = []
    for _ in trange(sim_repeats):
        factorization_set = []
        for i in range(n_restarts):
            bayesianCP = BayesianCP(dims=tensor.shape, rank=rank, init_alpha=70.0, model='zero_inflated_poisson')
            bayesianCP.fit(tensor, num_steps=num_steps)
            precis = bayesianCP.precis(tensor)
            factorization_set.append(CPTensor((torch.ones(rank), [precis[f'factor_{i}']['mean'] for i in range(3)])))
        factorization_sets.append(factorization_set)

    consensus = []
    for factorization_set in factorization_sets:
        data = np.column_stack([factorization.factors[2].numpy() for factorization in factorization_set]).T
        sums = data.T.sum(axis=0)
        sums[sums == 0] = 1
        data_normed = (data.T / sums).T * 1e5

        kmeans = KMeans(init="k-means++", n_clusters=rank, n_init=20, tol=1e-8)
        labels_ = make_pipeline(StandardScaler(), kmeans).fit(data_normed)[-1].labels_
        lof = LocalOutlierFactor(n_neighbors=15, contamination=0.002, metric='euclidean')
        outliers = make_pipeline(StandardScaler(), lof).fit_predict(data_normed)

        medians = np.stack(
            np.median(data_normed[(labels_ == group) & (outliers != -1), :], axis=0) for group in np.unique(labels_))
        bayesianCP = BayesianCP(dims=tensor.shape, rank=rank, init_alpha=70.0, model='zero_inflated_poisson_fixed',
                                fixed_mode=2, fixed_value=torch.from_numpy(medians.T).float())
        bayesianCP.fit(tensor, num_steps=num_steps)
        prs = bayesianCP.precis(tensor)
        a, b, c = [prs[f'factor_{i}']['mean'] for i in range(3)]
        consensus.append(CPTensor((torch.ones(rank), [a, b, c])))

    consensus_consistency = []
    for i in range(len(consensus)):
        for j in range(i):
            consensus_consistency.append(cosine_permute(consensus[i], consensus[j])[0])

    non_consensus_consistency = []
    for i in range(len(non_consensus)):
        for j in range(i):
            non_consensus_consistency.append(cosine_permute(non_consensus[i], non_consensus[j])[0])

    consistency = pd.DataFrame({'consensus': consensus_consistency, 'non consensus': non_consensus_consistency})
    consistency = consistency.melt(var_name='Method', value_name='Cosine Score')

    fig, ax = plt.subplots(1, 1, figsize=(2, 4))
    sns.boxplot(data=consistency, x='Method', y='Cosine Score', hue='Method', notch=False, width=0.5, showcaps=False,
                flierprops={"marker": "."}, dodge=False, ax=ax)
    plt.legend(bbox_to_anchor=(2.1, 1), frameon=False)

    annotator = Annotator(ax, data=consistency, pairs=[("consensus", "non consensus")], x='Method', y='Cosine Score')
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()
    ax.set_xticks([])
    ax.set_title(label='Internal Consistency', font={'size': 10})
    fig.savefig('results/consistency.pdf', dpi=300, bbox_inches="tight")

    cosine_perm = pd.DataFrame({
        'consensus': [cosine_permute(true, c)[0] for c in consensus],
        'non consensus': [cosine_permute(true, c)[0] for c in non_consensus]
    })
    cosine_perm = cosine_perm.melt(var_name='Method', value_name='Cosine Score')

    fig, ax = plt.subplots(1, 1, figsize=(2, 4))
    sns.boxplot(data=cosine_perm, x='Method', y='Cosine Score', hue='Method', notch=False, width=0.5, showcaps=False,
                flierprops={"marker": "."}, dodge=False, ax=ax)
    plt.legend(bbox_to_anchor=(2.1, 1), frameon=False)
    annotator = Annotator(ax, data=cosine_perm, pairs=[("consensus", "non consensus")], x='Method', y='Cosine Score')
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()
    ax.set_xticks([])
    ax.set_title(label='Versus Original Factors', font={'size': 10})
    fig.savefig('results/accuracy.pdf', dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="bayesian tensor factorization with zero inflated poisson model"
    )

    parser.add_argument("-s", "--seed", default=0, type=int, help="rng seed")
    parser.add_argument("-t", "--num-threads", default=8, type=int, help="number of threads")
    parser.add_argument("-nt", "--num-trials", default=20, type=int, help="number of trials to run")
    parser.add_argument("-nr", "--num-restarts", default=10, type=int,
                        help="number of restarts to run for consensus aggregation")
    parser.add_argument("-pz", "--prob-excess-zeros", default=0.6, type=float, help="probability of excess zeros")
    parser.add_argument("-ni", "--max-num-iters", default=1000, type=int, help="maximum number of iterations")
    parser.add_argument("-o", "--run-name", default="exp_1bc", type=str, help="file name prefix to save results")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("-a", "--alpha", default=1.0, type=float, help="alpha param of Gamma distribution for factors")
    parser.add_argument("-b", "--beta", default=0.3, type=float, help="beta param of Gamma distribution for factors")
    parser.add_argument("-r", "--rank", default=9, type=int, help="rank of tensor")
    parser.add_argument("-sh", "--shape", default=[40, 20, 2000], nargs='+', type=int, help="shape of tensor")

    args = parser.parse_args()
    main(args)