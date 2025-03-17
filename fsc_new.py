from utils import *
import igraph as ig
from evaluations_optmzd import *
import torch
import numpy as np
import scipy.io as sio

from scipy.linalg import null_space, eigh, sqrtm
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.cluster import KMeans

from data import *
import networkx as nx
import pandas as pd
import scipy as sp

import time

def kmeans_clustering(features: np.ndarray, num_clusters: int, normalize_rows: bool = False) -> np.ndarray:
    """
    Run k-means clustering on the rows of the feature matrix. Optionally normalize each row before clustering.

    :param features: (n, d) feature matrix.
    :param num_clusters: number of clusters.
    :param normalize_rows: if True, normalize each row.
    :return: cluster assignments (n,)
    """
    if normalize_rows:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features = features / norms
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=500, random_state=42)
    return kmeans.fit_predict(features)


def compute_top_eigen(mat: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the eigenvectors corresponding to the k smallest eigenvalues of a symmetric matrix.

    :param mat: (n, n) symmetric matrix.
    :param k: number of eigenvectors.
    :return: (n, k) eigenvector matrix.
    """
    assert mat.shape[0] >= k, 'Insufficient number of eigenvectors'
    eigenvalues, eigenvectors = eigh(mat, subset_by_index=[0, k - 1])
    return eigenvectors


def compute_laplacian(adj_mat: np.ndarray, normalize_laplacian: bool = False) -> (np.ndarray, np.ndarray):
    """
    Compute the graph Laplacian and return (L, D) where D is the degree matrix.
    If normalize_laplacian is True, returns the normalized Laplacian.

    :param adj_mat: (n, n) adjacency matrix.
    :param normalize_laplacian: if True, compute normalized Laplacian.
    :return: (L, D)
    """
    degree_vec = np.sum(adj_mat, axis=1)
    D = np.diag(degree_vec)
    L = D - adj_mat
    if normalize_laplacian:
        with np.errstate(divide='ignore'):
            inv_sqrt = 1.0 / np.sqrt(degree_vec)
        inv_sqrt[np.isinf(inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt
    return L, D


# ---------- Main Clustering Functions ----------

def normal_sc(adj_mat: np.ndarray, num_clusters: int, normalize_laplacian: bool = False,
              normalize_evec: bool = False) -> np.ndarray:
    """
    Standard Spectral Clustering.

    :param adj_mat: (n, n) adjacency matrix.
    :param num_clusters: number of clusters.
    :param normalize_laplacian: if True, use normalized Laplacian.
    :param normalize_evec: if True, normalize the rows of the eigenvector matrix before k-means.
    :return: cluster assignments (n,)
    """
    L, _ = compute_laplacian(adj_mat, normalize_laplacian)
    eigenvectors = compute_top_eigen(L, num_clusters)
    clusters = kmeans_clustering(eigenvectors, num_clusters, normalize_evec)
    return clusters


def group_fair_sc(adj_mat: np.ndarray, fair_mat: np.ndarray, num_clusters: int,
                  normalize_laplacian: bool = False, normalize_evec: bool = False) -> np.ndarray:
    """
    Fair Spectral Clustering using nullspace projection (FairSC).
    This method follows the FairSC (2019) approach by computing the nullspace basis of fair_mat.T.
    Steps:
        1. Compute L = D - W and D from the adjacency matrix.
        2. Compute Z = null_space(fair_mat.T). (fair_mat should be of shape (n, h-1) and centered.)
        3. Compute Q = sqrtm(Zᵀ * D * Z) and Q_inv = inv(Q + 1e-6*I).
        4. Compute the scaled nullspace: Z̃ = Z * Q_inv.
        5. Form the matrix M = Z̃ᵀ * L * Z̃ (and symmetrize it).
        6. Compute the top k eigenvectors Y of M.
        7. Obtain features H = Z̃ * Y and cluster H via k-means.

    :param adj_mat: (n, n) adjacency matrix (float64)
    :param fair_mat: (n, h-1) fairness matrix (float64), typically computed as (indicator - (group_size/n))
    :param num_clusters: number of clusters
    :param normalize_evec: if True, normalize rows of the eigenvector matrix before k-means.
    :return: cluster assignments (n,)
    """
    n = adj_mat.shape[0]
    # Step 1: Compute L and D
    L, D = compute_laplacian(adj_mat, normalize_laplacian=False)

    # Step 2: Compute Z = null_space(fair_mat.T)
    Z = null_space(fair_mat.T)
    if Z.shape[1] < num_clusters:
        raise ValueError("Nullspace of fairness matrix is too small for the given number of clusters.")

    # Step 3: Compute Q = sqrtm(Z.T @ D @ Z)
    Q = np.real(sqrtm(Z.T @ D @ Z))
    # To avoid numerical issues, add a small identity term.
    Q_inv = np.linalg.inv(Q + 1e-6 * np.eye(Q.shape[0]))

    # Step 4: Scale Z: Z̃ = Z @ Q_inv
    Z_tilde = Z @ Q_inv

    # Step 5: Form M = Z̃.T @ L @ Z̃ and symmetrize
    M = Z_tilde.T @ L @ Z_tilde
    M = (M + M.T) / 2.0

    # Step 6: Compute top k eigenvectors of M.
    Y = compute_top_eigen(M, num_clusters)

    # Step 7: Compute features H = Z̃ @ Y and cluster using k-means.
    H = Z_tilde @ Y
    clusters = kmeans_clustering(H, num_clusters, normalize_rows=normalize_evec)
    return clusters

def scalable_fair_sc(adj_mat: np.ndarray, fair_mat: np.ndarray, num_clusters: int,
                     tol_eig: float = 1e-8) -> np.ndarray:
    """
    Scalable Fair Spectral Clustering using an implicit projection (deflation) approach.

    This function avoids expensive dense matrix operations by constructing a LinearOperator that
    applies the projection onto the orthogonal complement of the fairness constraints.

    :param adj_mat: (n, n) adjacency matrix.
    :param fair_mat: (n, h-1) fairness (group membership) matrix.
    :param num_clusters: number of clusters.
    :param tol_eig: tolerance for the eigensolver.
    :return: cluster assignments (n,)
    """
    n = adj_mat.shape[0]
    m = fair_mat.shape[1]  # m = h - 1

    # Compute unnormalized Laplacian and degree matrix.
    L, _ = compute_laplacian(adj_mat, normalize_laplacian=False)
    degree_vec = np.sum(adj_mat, axis=1)

    # Compute D^{-1/2} as a diagonal matrix.
    with np.errstate(divide='ignore'):
        sqrt_d = np.sqrt(degree_vec)
        inv_sqrt_d = 1.0 / sqrt_d
    inv_sqrt_d[np.isinf(inv_sqrt_d)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt_d)

    # Normalized Laplacian: Ln = D^{-1/2} * L * D^{-1/2}
    Ln = D_inv_sqrt @ L @ D_inv_sqrt
    Ln = (Ln + Ln.T) / 2.0

    # Compute C = D^{-1/2} * fair_mat
    C = D_inv_sqrt @ fair_mat
    # Compute an orthonormal basis U2 for the range of C using QR decomposition.
    U2, _ = np.linalg.qr(C, mode='reduced')

    # Define the projection function: projects any vector onto the orthogonal complement of range(C)
    def project(v):
        return v - U2 @ (U2.T @ v)

    # Define the action of the projected operator: P * (Ln * (P(v)))
    def matvec(v):
        proj_v = project(v)
        w = Ln @ proj_v
        return project(w)

    A_linop = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    # Compute (num_clusters + m) eigenpairs; the first m should be nearly zero (fairness constraints).
    r = num_clusters + m
    eigvals, eigvecs = eigsh(A_linop, k=r, which='SA', tol=tol_eig, maxiter=1000)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Discard the first m eigenvectors.
    X_selected = eigvecs[:, m:m + num_clusters]

    # Unproject to obtain features: H = D^{-1/2} \ X_selected.
    # Since D^{-1/2} is diagonal, this is equivalent to element-wise division.
    H = X_selected / sqrt_d[:, np.newaxis]
    clusters = kmeans_clustering(H, num_clusters, normalize_rows=False)
    return clusters

def Afun_operator(Ln: np.ndarray, C: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """
    Implements the MATLAB Afun operator of sFSC 2023:

    y1 = (C' * C) \ (C' * b);
    y2 = b - C * y1;
    y3 = Ln * y2;
    y4 = (C' * C) \ (C' * y3);
    Apb = y3 - C * y4 - sigma * y2 + sigma * b;

    :param Ln: normalized Laplacian.
    :param C: matrix computed as inv(sqrt(D)) * fair_mat.
    :param b: vector on which to operate.
    :param sigma: scaling constant, set as norm(Ln, 1).
    :return: result of the operator.
    """

    CtC = C.T @ C
    y1 = np.linalg.solve(CtC, C.T @ b)
    y2 = b - C @ y1
    y3 = Ln @ y2
    y4 = np.linalg.solve(CtC, C.T @ y3)
    return y3 - C @ y4 - sigma * y2 + sigma * b

def scalable_fair_sc_deflation(adj_mat: np.ndarray, fair_mat: np.ndarray, num_clusters: int, tol_eig: float = 1e-8) -> np.ndarray:
    """
    Scalable Fair Spectral Clustering using a deflation-based approach that mimics the MATLAB implementation
    (alg3.m with Afun.m). This function:
    1. Computes the Laplacian L and its square root from the degree matrix D.
    2. Forms the normalized Laplacian Ln = inv(sqrtD)Linv(sqrtD).
    3. Computes C = inv(sqrtD) * fair_mat.
    4. Defines an operator Afun(Ln, C, ·, sigma) where sigma = norm(Ln,1).
    5. Uses an iterative eigensolver to compute the k smallest eigenvectors.
    6. Computes H = inv(sqrtD)*X and runs k-means on H.

    :param adj_mat: (n, n) adjacency matrix (as float64).
    :param fair_mat: (n, h-1) fairness (group membership) matrix (as float64).
    :param num_clusters: number of clusters.
    :param tol_eig: tolerance for the eigensolver.
    :return: cluster assignments (n,)
    """

    n = adj_mat.shape[0]
    # Compute Laplacian and degree matrix D.
    L, D = compute_laplacian(adj_mat, normalize_laplacian=False)
    degree_vec = np.diag(D)

    # Compute sqrtD = sqrtm(D). Since D is diagonal, this is simply sqrt(degree_vec).
    sqrt_d = np.sqrt(degree_vec)
    sqrtD = np.diag(sqrt_d)

    # Compute Ln = inv(sqrtD) * L * inv(sqrtD)
    inv_sqrtD = np.diag(1.0 / sqrt_d)
    Ln = inv_sqrtD @ L @ inv_sqrtD
    Ln = (Ln + Ln.T) / 2.0  # enforce symmetry

    # Set C = inv(sqrtD) * fair_mat, i.e. divide each row of fair_mat by sqrt_d.
    C = fair_mat / sqrt_d[:, np.newaxis]

    sigma = np.linalg.norm(Ln, 1)

    # Define a LinearOperator for the Afun operator.
    def matvec(b):
        return Afun_operator(Ln, C, b, sigma)

    A_linop = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    # Solve for the num_clusters eigenpairs using eigsh.
    # (We use 'SA' for smallest algebraic eigenvalues; adjust ncv if needed.)
    eigenvals, X = eigsh(A_linop, k=num_clusters, which='SA', tol=tol_eig, maxiter=1000, ncv=4*num_clusters)

    # Compute H = inv(sqrtD) * X, i.e. divide each row of X by sqrt_d.
    H = X / sqrt_d[:, np.newaxis]

    # Run k-means clustering on H.
    clusters = kmeans_clustering(H, num_clusters, normalize_rows=False)
    return clusters

# ---------- Example Usage ----------

if __name__ == "__main__":
    start = time.perf_counter()
    A, F = load_facebook()
    A, F = load_friendship()

    G1 = nx.from_numpy_array(A, create_using=nx.Graph, parallel_edges=False)
    # Create an igraph graph from the adjacency matrix A.
    # Here we assume A is a NumPy array. The (A > 0).tolist() converts it to a boolean adjacency matrix.
    g_ig = ig.Graph.Adjacency((A > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    F_gen = compute_F(F)

    standard_groups = reflow_clusters(F)
    groups = np.transpose(standard_groups)

    predfsc = group_fair_sc(A, F_gen, 5)
    pred_FSC = reflow_clusters(predfsc)
    # calculate balance
    balances_FSC, Bfsc = compute_group_balance(pred_FSC.numpy(), groups.numpy())
    # convert predicted labels to networkx clusters
    coms_FSC = lab2com(pred_FSC)

    predsfsc = scalable_fair_sc_deflation(A, F_gen, 3)
    pred_sFSC = reflow_clusters(predsfsc)
    # calculate balance
    balances_sFSC, Bsfsc = compute_group_balance(pred_sFSC.numpy(), groups.numpy())
    # convert predicted labels to networkx clusters
    coms_sFSC = lab2com(pred_sFSC)
    # compute modularity

    end = time.perf_counter()

    # compute modularity
    Qfsc = nx.community.modularity(G1, coms_FSC)
    Qsfsc = nx.community.modularity(G1, coms_sFSC)

    # Convert them to lists (if they are tensors or NumPy arrays).
    membership_FSC = pred_FSC.numpy().tolist() if hasattr(pred_FSC, "numpy") else pred_FSC.tolist()
    membership_sFSC = pred_sFSC.numpy().tolist() if hasattr(pred_sFSC, "numpy") else pred_sFSC.tolist()

    # compute modularity
    Qfsc2 = g_ig.modularity(membership_FSC)
    Qsfsc2 = g_ig.modularity(membership_sFSC)

    print("ok: FSC is Q={}, B={}".format(Qfsc,Bfsc))
    print("ok: sFSC is Q={}, B={}".format(Qsfsc, Bsfsc))

    print("igraph mod comparison to NX: FSC Q_NX={}, Q_IG={}".format(Qfsc,Qfsc2))
    print("igraph mod comparison to NX: sFSC Q_NX={}, Q_IG={}".format(Qsfsc,Qsfsc2))

    start1 = time.perf_counter()
    Qfsc = nx.community.modularity(G1, coms_FSC)
    end1 = time.perf_counter()

    start2 = time.perf_counter()
    Qfsc2 = g_ig.modularity(membership_FSC)
    end2 = time.perf_counter()

    print("igraph modularity computation took {:.8f} seconds".format(end2 - start2))
    print("NetworkX modularity computation took {:.8f} seconds".format(end1 - start1))

    print("Overall runtime with evaluations functions took {:.4f} seconds".format(end - start))