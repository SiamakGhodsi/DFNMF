import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import torch
import math
from models.nmf_helpers import compute_RS

def IGDC(a: np.ndarray, b:np.ndarray, s: str):
    igdc = 0
    if a.ndim == 1:
        igdc = math.dist(a, b)
    else:
        a = a.transpose()
        if (s=='avg'):
            cumul = 0
            for j in a:
                cumul+= math.dist(j,b)
            igdc = cumul/np.shape(a)[0]
        elif(s=='min'):
            dis = []
            for j in a:
                dis.append(math.dist(j,b))
            igdc = min(dis)
    return igdc

def get_membership_matrix(indices: np.ndarray) -> np.ndarray:
    """
    Convert a 1D array of membership indices into a one-hot membership matrix.
    This version avoids np.eye allocation if needed.
    :param indices: (num_nodes,) array of cluster or group labels.
    :return: (num_nodes, num_indices) one-hot membership matrix.
    """
    n = indices.shape[0]
    num_indices = int(np.max(indices)) + 1
    membership = np.zeros((n, num_indices), dtype=np.float64)
    membership[np.arange(n), indices.astype(int)] = 1
    return membership


def compute_group_balance(clusters: np.ndarray, groups: np.ndarray, normalize: bool = False) -> (np.ndarray, float):
    """
    Compute per-cluster and average balance.
    The balance of a cluster is defined as (min(counts)/max(counts)) [possibly scaled by a normalization factor].
    :param clusters: (n,) predicted cluster labels.
    :param groups: (n,) protected group labels.
    :param normalize: if True, scale the ratio by (min(group_size)/max(group_size)).
    :return: balances (per cluster), average balance.
    """
    # Build one-hot membership matrices.
    cluster_memberships = get_membership_matrix(clusters)
    group_memberships = get_membership_matrix(groups)

    # counts: each row corresponds to a cluster and each column to a group.
    counts = cluster_memberships.T @ group_memberships  # shape: (num_clusters, num_groups)
    # Avoid division by zero by adding a small epsilon.
    eps = 1e-6
    # For each cluster, the minimum ratio is min(counts[c, :]) / (max(counts[c, :]) + eps).
    balances = (np.min(counts, axis=1) / (np.max(counts, axis=1) + eps))
    if normalize:
        # Normalization factor computed from overall group sizes.
        group_sizes = group_memberships.sum(axis=0)
        norm_factor = np.min(group_sizes) / (np.max(group_sizes) + eps)
    else:
        norm_factor = 1.0
    balances = balances * norm_factor
    return balances, balances.mean()


def compute_individual_balance(clusters: np.ndarray, groups: np.ndarray, normalize: bool = False) -> (np.ndarray, float):
    """
    Compute per-node balance based on the fairness graph computed from protected groups.
    For each node i, the balance is defined as:
          balance[i] = (min(counts[i, :]) / (max(counts[i, :]) + eps))
    where counts = fair_mat @ get_membership_matrix(clusters)
          and fair_mat is computed from groups using compute_RS.

    :param clusters: (n,) predicted cluster labels.
    :param groups: (n,) protected group membership labels.
    :param normalize: if True, scale by (min(cluster_size) / (max(cluster_size)+eps)).
    :return: a tuple containing:
             - balances: (n,) array with the balance for each individual,
             - average balance: a float, the mean balance over all individuals.
    """
    # Compute the fairness graph from groups.
    fair_mat = compute_RS(groups)  # fair_mat has shape (n, n)

    # Get the one-hot membership matrix for the predicted clusters.
    cluster_memberships = get_membership_matrix(clusters)  # shape: (n, num_clusters)

    # Compute the "counts" matrix: for each node, how much membership it gets
    # weighted by the fairness graph.
    counts = fair_mat @ cluster_memberships  # shape: (n, num_clusters)

    eps = 1e-6
    # Compute the balance per node: minimum value in each row divided by maximum value in that row.
    balances = np.min(counts, axis=1) / (np.max(counts, axis=1) + eps)

    if normalize:
        # Optionally, normalize using overall cluster sizes.
        cluster_sizes = cluster_memberships.sum(axis=0)
        norm_factor = np.min(cluster_sizes) / (np.max(cluster_sizes) + eps)
    else:
        norm_factor = 1.0

    balances = balances * norm_factor

    # Return the per-node balances and the mean balance.
    return balances, balances.mean()


def reflow_clusters0(clusters: np.ndarray) -> np.ndarray:
    """
    Remap cluster labels to contiguous integers.
    :param clusters: (n,) cluster labels.
    :return: (n,) reflowed labels.
    """
    _, reflow = np.unique(clusters, return_inverse=True)
    return reflow
    
def reflow_clusters(y):
    """
    Reflow clusters using torch's one_hot.
    :param y: array-like cluster labels.
    :return: tensor of reflowed labels.
    """
    if isinstance(y, torch.Tensor):
        y = y.clone().detach().to(torch.long)
    else:
        y = torch.tensor(y, dtype=torch.long)

    unique_vals = torch.unique(y)
    # Build a cost matrix implicitly: here we re-map so that unique labels are assigned in sorted order.
    # This is equivalent to using return_inverse from np.unique.
    mapping = {old.item(): new for new, old in enumerate(torch.sort(unique_vals).values)}
    reflow = torch.tensor([mapping[val.item()] for val in y])
    return reflow


def align_clusters(true_clusters: np.ndarray, pred_clusters: np.ndarray) -> (int, np.ndarray, np.ndarray):
    """
    Align predicted clusters to true clusters using the Hungarian algorithm.
    :param true_clusters: (n,) ground truth labels.
    :param pred_clusters: (n,) predicted labels.
    :return: number of mismatches, reflowed true clusters, aligned predicted clusters.
    """
    # Reflow clusters to contiguous labels.
    reflow_true = reflow_clusters0(true_clusters)
    reflow_pred = reflow_clusters0(pred_clusters)
    num_clusters = int(np.max(reflow_true)) + 1

    # Build cost matrix: cost[i, j] = number of mismatches if predicted cluster j is mapped to true cluster i.
    cost_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int32)
    for i in range(num_clusters):
        for j in range(num_clusters):
            # Count number of nodes with true label i and predicted label j.
            cost_matrix[i, j] = np.sum((reflow_true == i) & (reflow_pred != j))
    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Build aligned predicted clusters
    aligned_pred = np.zeros_like(reflow_pred)
    for i, j in zip(row_ind, col_ind):
        aligned_pred[reflow_pred == j] = i
    num_mistakes = np.sum(aligned_pred != reflow_true)
    return num_mistakes, reflow_true, aligned_pred

## ------------------------------------------- Clustering metrics ----------------------------------------

def lab2com0(y: np.ndarray):
    """
    Convert label vector into list of sets (each set containing indices for that label).
    """
    return [set(np.where(y == label)[0]) for label in np.unique(y)]

def lab2com(y):
    """
    Convert torch tensor of labels into list of sets.
    """
    unique_vals = torch.unique(y)
    return [set(torch.where(y == val)[0].cpu().numpy()) for val in unique_vals]
