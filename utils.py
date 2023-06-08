import torch
import pyro

import numpy as np
import pandas as pd

from torch.distributions.gamma import Gamma
from tensorly.metrics.factors import congruence_coefficient
from tensorly.cp_tensor import cp_normalize


def random_gamma(shape, alpha, beta):
    """
    Helper for sampling from a gamma distribution
    """
    return Gamma(torch.ones(shape) * alpha,  torch.ones(shape) * beta).sample()


def variance_explained(original, reconstructed):
    """
    Calculates the variance explained by the tensor factorization:
    variance_explained = 1 - || original - reconstructed ||^2 / || original ||^2
    """
    if type(original) != torch.Tensor:
        original = torch.tensor(original)
    if type(reconstructed) != torch.Tensor:
        reconstructed = torch.tensor(reconstructed)
    return 1 - (torch.norm(original.float() - reconstructed.float()) ** 2 / torch.norm(original.float()) ** 2)


def generate_zip_tensor(shape, rank, alpha, beta, p_zero):
    """
    Generates a tensor with a known rank R factorization. Factors generated from a Gamma distribution.
    Observed tensor sampled from a ZIP distribution.
    """
    A_factor_matrix = random_gamma(shape=(shape[0], rank), alpha=alpha, beta=beta)
    B_factor_matrix = random_gamma(shape=(shape[1], rank), alpha=alpha, beta=beta)
    C_factor_matrix = random_gamma(shape=(shape[2], rank), alpha=alpha, beta=beta)

    true_tensor = torch.einsum('ir,jr,kr->ijk', A_factor_matrix, B_factor_matrix, C_factor_matrix)
    observed_tensor = pyro.distributions.ZeroInflatedPoisson(rate=true_tensor, gate=torch.tensor(p_zero)).sample()
    return true_tensor, observed_tensor, [A_factor_matrix, B_factor_matrix, C_factor_matrix]


def cosine_permute(ref_cp_tensor, tensors_to_permute):
    """
    """
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


def get_assignment(true_usages, inferred_usages):
    """
    Adapted from https://codeocean.com/capsule/6314882/tree/v1

    [23] D. Kotliar, A. Veres, M. A. Nagy, S. Tabrizi, E. Hodis, D.A. Melton, and P.C. Sabeti, Identifying
    gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq,
    363 eLife (2019), 8, e43803, https://doi.org/10.7554/eLife.43803

    Get a mapping to reorder columns of inferred usage matrix
    to best match the columns of true usage matrix to allow easier comparison
    """
    R = {}
    mapping = {}
    avail = set(true_usages.columns)  # Set of programs to match
    toassign = set(inferred_usages.columns)  # Set of programs to assign

    for i in inferred_usages.columns:
        R[i] = true_usages.corrwith(inferred_usages[i])

    R = pd.DataFrame.from_dict(R)
    Runst = R.unstack().reset_index()
    Runst.columns = ['learned', 'truth', 'corr']
    Runst = Runst.sort_values(by='corr', ascending=False)

    for i in Runst.index:
        if (Runst.loc[i, 'truth'] in avail) and (Runst.loc[i, 'learned'] in toassign):
            mapping[Runst.loc[i, 'learned']] = Runst.loc[i, 'truth']
            avail = avail - {Runst.loc[i, 'truth']}
            toassign = toassign - {Runst.loc[i, 'learned']}

        if len(toassign) == 0:
            break

    return mapping, R


def df_col_corr(X, Y):
    """
    Adapted from https://codeocean.com/capsule/6314882/tree/v1

    [23] D. Kotliar, A. Veres, M. A. Nagy, S. Tabrizi, E. Hodis, D.A. Melton, and P.C. Sabeti, Identifying
    gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq,
    363 eLife (2019), 8, e43803, https://doi.org/10.7554/eLife.43803

    Calculate pairwise correlation between columns of X and Y
    """
    X_norm = X.subtract(X.mean(axis=0), axis=1)
    X_norm = X_norm.divide(X_norm.std(axis=0), axis=1)
    Y_norm = Y.subtract(Y.mean(axis=0), axis=1)
    Y_norm = Y_norm.divide(Y_norm.std(axis=0), axis=1)

    X_mask = np.ma.array(X_norm.values, mask=np.isnan(X_norm.values))
    Y_mask = np.ma.array(Y_norm.values, mask=np.isnan(Y_norm.values))

    return pd.DataFrame(np.ma.dot(X_mask.T, Y_mask) / (X.shape[0] - 1), index=X.columns, columns=Y.columns)


def load_df_from_npz(filename):
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj


def load_and_relabel(cobj, true_gep_means, K=11, ldthresh=0.07):
    """
    Adapted from https://codeocean.com/capsule/6314882/tree/v1

    [23] D. Kotliar, A. Veres, M. A. Nagy, S. Tabrizi, E. Hodis, D.A. Melton, and P.C. Sabeti, Identifying
    gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq,
    363 eLife (2019), 8, e43803, https://doi.org/10.7554/eLife.43803

    Load results from a cNMF run
    """
    hvgs = open(cobj.paths['nmf_genes_list']).read().split('\n')
    (usage, gene_scores_Z, gene_scores, top_genes) = cobj.load_results(K=K, density_threshold=ldthresh)
    gene_scores = gene_scores.T
    gene_scores_Z = gene_scores_Z.T
    tpm_stats = load_df_from_npz(cobj.paths['tpm_stats'])

    stds = tpm_stats['__std']
    true_gep_means_hvgs = true_gep_means.loc[:, hvgs]

    gene_scores_hvgs = gene_scores.loc[:, hvgs]
    true_gep_means_hvgs_norm = true_gep_means_hvgs.div(stds.loc[hvgs], axis=1)

    gene_scores_hvgs_norm = gene_scores_hvgs.div(stds.loc[hvgs], axis=1)
    assignmap, R = get_assignment(true_gep_means_hvgs_norm.T, gene_scores_hvgs_norm.T)

    gene_scores_relabeled = gene_scores.rename(index=assignmap)
    gene_scores_relabeled = gene_scores_relabeled.sort_index(axis=0)

    gene_scores_Z_relabeled = gene_scores_Z.rename(index=assignmap)
    gene_scores_Z_relabeled = gene_scores_Z_relabeled.sort_index(axis=0)

    usage.columns = [int(x) for x in usage.columns]
    usage_relabeled = usage.rename(columns=assignmap)
    usage_relabeled = usage_relabeled.sort_index(axis=1)

    return {
        'stds': stds, 'hvgs': hvgs, 'usage': usage_relabeled, 'tpm': gene_scores_relabeled,
        'Z': gene_scores_Z_relabeled, 'obj': cobj
    }


def load_and_relabel(cobj, true_gep_means, K=11, ldthresh=0.07):
    res = {}
    hvgs = open(cobj.paths['nmf_genes_list']).read().split('\n')
    (usage, gene_scores_Z, gene_scores, top_genes) = cobj.load_results(K=K, density_threshold=ldthresh)
    gene_scores = gene_scores.T
    gene_scores_Z = gene_scores_Z.T
    tpm_stats = load_df_from_npz(cobj.paths['tpm_stats'])

    stds = tpm_stats['__std']
    true_gep_means_hvgs = true_gep_means.loc[:, hvgs]

    gene_scores_hvgs = gene_scores.loc[:, hvgs]
    true_gep_means_hvgs_norm = true_gep_means_hvgs.div(stds.loc[hvgs], axis=1)

    gene_scores_hvgs_norm = gene_scores_hvgs.div(stds.loc[hvgs], axis=1)
    assignmap, R = get_assignment(true_gep_means_hvgs_norm.T,
                                  gene_scores_hvgs_norm.T)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    gene_scores_hvgs_norm_relabeled = gene_scores_hvgs_norm.rename(index=assignmap)
    # gene_scores_hvgs_norm_relabeled.index = [int(x.replace('True_', '')) for x in gene_scores_hvgs_norm_relabeled.index]
    gene_scores_hvgs_norm_relabeled = gene_scores_hvgs_norm_relabeled.sort_index(axis=0)

    R = df_col_corr(true_gep_means_hvgs_norm.T, gene_scores_hvgs_norm_relabeled.T)
    ax = sns.heatmap(R.sort_index(axis=1).sort_index(axis=0))
    ax.set_title('Spectra correlation')

    gene_scores_relabeled = gene_scores.rename(index=assignmap)
    gene_scores_relabeled = gene_scores_relabeled.sort_index(axis=0)

    gene_scores_Z_relabeled = gene_scores_Z.rename(index=assignmap)
    gene_scores_Z_relabeled = gene_scores_Z_relabeled.sort_index(axis=0)

    usage.columns = [int(x) for x in usage.columns]
    usage_relabeled = usage.rename(columns=assignmap)
    # usage_relabeled.columns = [int(x.replace('True_', '')) for x in usage_relabeled.columns]
    usage_relabeled = usage_relabeled.sort_index(axis=1)
    # usage_relabeled.columns = ['cNMF_%d' % x for x in usage_relabeled.columns]

    res = {'stds': stds, 'hvgs': hvgs, 'usage': usage_relabeled, 'tpm': gene_scores_relabeled,
           'Z': gene_scores_Z_relabeled, 'obj': cobj}
    return (res)