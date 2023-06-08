import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
import pyro
import torch
import argparse

from cnmf import cNMF
from scsim2 import SCsim_DiffScale
from collections import defaultdict
from sklearn.decomposition import non_negative_factorization
from scBTF import SingleCellTensor, SingleCellBTF
from utils import load_and_relabel, get_assignment, df_col_corr


def extract_matrix_and_tensor(simulation, nactivities, datadir, run_name):
    """ create normalized cell-by-gene matrix and the pseudobulk tensor"""
    adata = sc.AnnData(X=simulation.counts, obs=simulation.usage, var=simulation.spectra.T).copy()
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_counts=200)
    adata.raw = adata
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    countfn = '{d}/{n}_counts.h5ad'.format(d=datadir, n=run_name)
    sc.write(countfn, adata)

    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.leiden(adata, key_added="clusters", resolution=0.1)

    adata.obs['activity'] = np.argmax(adata.obs[[f'Activity_{i}' for i in range(1, nactivities + 1)]].values,
                                      axis=1) + 1
    adata.obs['donor'] = adata.obs['activity']
    adata.obs['activity'] = adata.obs['activity'].astype(str)
    adata.obs['donor'] = adata.obs['donor'] + (np.random.randint(0, 2, adata.obs.shape[0]) * args.nactivities)  # TODO
    adata.obs['donor'] = adata.obs['donor'].map(lambda x: f'D_{x}')

    sc_tensor = SingleCellTensor.from_anndata(
        adata, sample_label='donor', celltype_label='clusters', filter_gene_count=0,
        hgnc_approved_genes_only=False, normalize=True, variance_scale=False, sqrt_transform=False
    )
    sc_tensor.tensor = sc_tensor.tensor.round()
    return countfn, sc_tensor


def run_cnmf(simulation, rank, consensus_num_restarts, num_highvar_genes, counts_fn, datadir, seed, ld_thresh):
    cnmf_obj = cNMF(output_dir=datadir, name=f'{seed}')
    cnmf_obj.prepare(
        counts_fn=counts_fn, components=rank, n_iter=consensus_num_restarts, seed=seed,  # TODO
        num_highvar_genes=num_highvar_genes
    )
    cnmf_obj.factorize()
    cnmf_obj.combine()
    cnmf_obj.consensus(k=rank, density_threshold=ld_thresh, show_clustering=False, refit_usage=True)

    res = load_and_relabel(cnmf_obj, simulation.spectra, K=rank, ldthresh=ld_thresh)
    return res


def ziptf_gene_scores(sc_tensor, rank, max_num_iters):
    factorization_set = SingleCellBTF.factorize(
        sc_tensor=sc_tensor, rank=[rank], model='zero_inflated_poisson', n_restarts=1, num_steps=max_num_iters,
        init_alpha=sc_tensor.tensor.mean().item(), plot_var_explained=False
    )
    gene_scores = factorization_set.factorizations[rank][0].gene_factor['mean'].numpy()
    gene_scores = pd.DataFrame(gene_scores, index=factorization_set.sc_tensor.gene_list)
    return gene_scores


def nncp_als_gene_scores(sc_tensor, rank, max_num_iters):
    factorization_set = SingleCellBTF.factorize_hals(
        sc_tensor=sc_tensor, rank=[rank], n_restarts=1, num_steps=max_num_iters, plot_var_explained=False
    )
    gene_scores = factorization_set.factorizations[rank][0].gene_factor['mean'].numpy()
    gene_scores = pd.DataFrame(gene_scores, index=factorization_set.sc_tensor.gene_list)
    return gene_scores


def nmf_gene_scores(simulation, rank, seed):
    nmf_kwargs = {
        'beta_loss': 'kullback-leibler',  # frobenius
        'solver': 'mu',  # cd
        'max_iter': 1000,
        'init': 'random'
    }
    (_, gene_scores, _) = non_negative_factorization(
        simulation.counts, n_components=rank, random_state=seed, **nmf_kwargs)
    return pd.DataFrame(gene_scores, columns=simulation.spectra.columns).T


def cziptf_gene_scores(sc_tensor, rank, consensus_num_restarts, max_num_iters, contamination):
    factorization_set = SingleCellBTF.factorize(
        sc_tensor=sc_tensor, rank=rank, model='zero_inflated_poisson', n_restarts=consensus_num_restarts,
        num_steps=max_num_iters, init_alpha=sc_tensor.tensor.mean().item(), plot_var_explained=False
    )
    consensus = SingleCellBTF.get_consensus_factorization(
        factorization_set=factorization_set, rank=rank, model='zero_inflated_poisson_fixed',
        num_steps=max_num_iters, contamination=contamination, init_alpha=sc_tensor.tensor.mean().item(),
        fixed_mode_variance=100.
    )
    gene_scores = consensus.factorizations[rank][0].gene_factor['mean'].numpy()
    return pd.DataFrame(gene_scores, index=consensus.sc_tensor.gene_list)


def average_pearson_correlation(gene_score, true_gep_means, hvgs_all, stds):
    """ Calculate average pearson correlation between gene factors and GEPs after alignment"""
    hvgs = [g for g in hvgs_all if g in gene_score.index]
    true_gep_means_hvgs = true_gep_means.loc[:, hvgs]

    gene_scores_hvgs = gene_score.T.loc[:, hvgs]
    true_gep_means_hvgs_norm = true_gep_means_hvgs.div(stds.loc[hvgs], axis=1)

    gene_scores_hvgs_norm = gene_scores_hvgs.div(stds.loc[hvgs], axis=1)
    assignmap, _ = get_assignment(true_gep_means_hvgs_norm.T, gene_scores_hvgs_norm.T)

    gene_scores_hvgs_norm_relabeled = gene_scores_hvgs_norm.rename(index=assignmap)
    gene_scores_hvgs_norm_relabeled = gene_scores_hvgs_norm_relabeled.sort_index(axis=0)

    R = df_col_corr(true_gep_means_hvgs_norm.T, gene_scores_hvgs_norm_relabeled.T)
    return np.mean([R.loc[r, r] for r in R.index])


def main(args):
    torch.set_num_threads(args.num_threads)

    np.random.seed(args.seed)
    seeds = np.random.randint(0, np.iinfo(np.int32).max, args.num_trials)

    datadir = './data'
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    activity_deloc_apcs = []
    for activity_deloc in args.activity_delocs:
        for seed in seeds:
            torch.manual_seed(seed)
            pyro.set_rng_seed(seed)
            np.random.seed(seed)

            deprob = [.1] * args.nidentities + [.2] * args.nactivities
            deloc = [4] * args.nidentities + [activity_deloc] * args.nactivities

            density_threshold = 0.15
            simulation = SCsim_DiffScale(seed=seed, libloc=7.68)
            simulation.simulate(
                ncells=args.ncells, ngenes=args.ngenes, nidentities=args.nidentities, pct_doublets=.05,
                activity_pct=[.3] * args.nactivities, diffexploc=deloc, activity_min=[.2] * args.nactivities,
                activity_max=[.7] * args.nactivities, diffexpscale=1, diffexpprob=deprob, delta_range_libloc=0,
                activity_identities=[list(range(1, args.nidentities + 1))] * args.nactivities
            )
            simulation.geneparams.to_csv('{d}/{n}.geneparams.tsv'.format(d=datadir, n=args.run_name), sep='\t')

            countfn, sc_tensor = extract_matrix_and_tensor(
                simulation=simulation, nactivities=args.nactivities, run_name=args.run_name, datadir=datadir
            )

            rank = args.nidentities + args.nactivities
            cnmf_res = run_cnmf(
                simulation=simulation, rank=rank, consensus_num_restarts=args.consensus_num_restarts,
                num_highvar_genes=args.num_highvar_genes, counts_fn=countfn, datadir=datadir, seed=seed,
                ld_thresh=density_threshold
            )

            gene_scores = {
                'NMF': nmf_gene_scores(simulation, rank, seed),
                'CNMF': cnmf_res['obj'].load_results(K=rank, density_threshold=density_threshold)[2],
                'NNCP_ALS': nncp_als_gene_scores(sc_tensor, rank, args.max_num_iters),
                'ZIPTF': ziptf_gene_scores(sc_tensor, rank, args.max_num_iters),
                'CZIPTF': cziptf_gene_scores(sc_tensor, rank, args.consensus_num_restarts, args.max_num_iters,
                                             args.contamination)
            }

            apcs = defaultdict(list)
            for method, gene_score in gene_scores.items():
                apc = average_pearson_correlation(
                    gene_score=gene_score, true_gep_means=simulation.spectra, hvgs_all=cnmf_res['hvgs'],
                    stds=cnmf_res['stds']
                )
                apcs[method].append(apc)
            apcs = pd.DataFrame(apcs).melt(var_name='Method', value_name='APC')
            apcs['Mean DE logFC'] = str(activity_deloc)
            activity_deloc_apcs.append(apcs)

    activity_deloc_apcs = pd.concat(activity_deloc_apcs, axis=0)
    activity_deloc_apcs['deloc_method'] = activity_deloc_apcs['Mean DE logFC'] + "_" + activity_deloc_apcs['Method']
    order = ['NMF', 'CNMF', 'CP_HALS', 'ZIP_BTF', 'C_ZIP_BTF']
    activity_deloc_apcs['order'] = activity_deloc_apcs.Method.map(dict(zip(order, range(len(order)))))
    activity_deloc_apcs = activity_deloc_apcs.sort_values(by=['Mean DE logFC', 'order'])
    activity_deloc_apcs.to_csv(f'results/{args.run_name}_apcs.csv')

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
    ax = sns.stripplot(
        data=activity_deloc_apcs, x='deloc_method', y='APC', hue='Method', marker='X', size=5,
        jitter=.1, dodge=False, palette='Dark2', ax=ax
    )
    plt.legend(
        bbox_to_anchor=(1, 1), frameon=False, fontsize='x-small', markerscale=0.75, handletextpad=-0.1,
        title='Method', title_fontsize='small'
    )
    ymin = activity_deloc_apcs.APC.min() - 0.05
    ax.set_ylim(ymin, 1.03)
    ax.set_xlim(-0.5, len(activity_deloc_apcs.deloc_method.unique()) - 0.05)
    ax.set_ylabel('Average Pearson Correlation')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_xticks([])

    for x in range(4, len(activity_deloc_apcs.deloc_method.unique()) - 1, 5):
        ax.plot([x + 0.5, x + 0.5], [ymin, 1.03], 'k', lw=1.5)
    for x in range(0, len(activity_deloc_apcs.deloc_method.unique()) - 1):
        ax.plot([x + 0.5, x + 0.5], [ymin, 1.03], 'k:', lw=0.05)
    plt.show()
    fig.savefig(f"results/{args.run_name}_pearson_correlation.pdf", dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare CZIPTF with baseline matrix and tensor factorization methods on simulated scRNA-Seq data"
    )

    parser.add_argument('-n', '--run-name', type=str, default='exp2', help='Run name')
    parser.add_argument('-s', '--seed', type=int, default=54, help='Random seed')
    parser.add_argument('-c', '--ncells', type=int, default=10000, help='Number of cells')
    parser.add_argument('-g', '--ngenes', type=int, default=5000, help='Number of genes')
    parser.add_argument('-de', '--activity-delocs', default=[0.25, 0.5, 0.75], nargs='+', type=float,
                        help='Activity DE loc')
    parser.add_argument('-nid', '--nidentities', type=int, default=5, help='Number of identity programs')
    parser.add_argument('-na', '--nactivities', type=int, default=3, help='Number of activity programs')
    parser.add_argument('-cm', '--contamination', type=float, default=0.0001, help='Contamination')
    parser.add_argument('-hvg', '--num-highvar-genes', type=int, default=1000, help='Number of highly variable genes')
    parser.add_argument('-nr', '--consensus-num-restarts', type=int, default=10, help='Number of consensus restarts')
    parser.add_argument('-nit', '--max-num-iters', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-nt', '--num-threads', type=int, default=8, help='Number of threads')
    parser.add_argument("-ntr", "--num-trials", default=10, type=int, help="number of trials to run")

    args = parser.parse_args()
    main(args)