from numpy import ndarray

from utils import *
from evaluations import *
import torch
import numpy as np
import scipy.io as sio

from scipy.linalg import null_space, eigh, sqrtm
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.cluster import KMeans

#from data import *
#from versions.submitted_code.data import load_facebook
import networkx as nx
import pandas as pd
import scipy as sp

def kmeans_clustering(features: np.ndarray, num_clusters: int, normalize_rows: bool = False) -> np.ndarray:
    """
    Run k-means clustering on the rows of the feature matrix.
    Optionally normalize each row before clustering.

    :param features: (n, d) feature matrix.
    :param num_clusters: number of clusters.
    :param normalize_rows: if True, normalize each row before clustering.
    :return: cluster assignments (n,)
    """
    if normalize_rows:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
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

    :param adj_mat: (n, n) adjacency matrix.
    :param fair_mat: (n, h-1) fairness (group membership) matrix.
    :param num_clusters: number of clusters.
    :param normalize_laplacian: if True, use normalized Laplacian adjustments.
    :param normalize_evec: if True, normalize eigenvector rows before clustering.
    :return: cluster assignments (n,)
    """
    # Compute nullspace basis of fair_mat.T
    Z = null_space(fair_mat.T)
    if Z.shape[1] < num_clusters:
        raise ValueError("The null space of the fairness matrix is too small for the given number of clusters.")

    # Compute Laplacian (using unnormalized Laplacian here)
    L, D = compute_laplacian(adj_mat, normalize_laplacian=False)

    if normalize_laplacian:
        # If desired, adjust via Q = sqrtm(Z.T * D * Z)
        Q = np.real(sqrtm(Z.T @ D @ Z))
        Q_inv = np.linalg.inv(Q + 1e-6 * np.eye(Q.shape[0]))
        Z = Z @ Q_inv

    LL = Z.T @ L @ Z
    LL = (LL + LL.T) / 2.0  # enforce symmetry

    Y = compute_top_eigen(LL, num_clusters)
    features = Z @ Y
    clusters = kmeans_clustering(features, num_clusters, normalize_evec)
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


# ---------- Example Usage ----------

if __name__ == "__main__":

    path2 = r"\\tib.tibub.de\DFS0\Home\ghodsis\Documents\My Files\Projects\Projects\FairNMF\code\data\School\Facebook-known-pairs_data_2013.csv"
    path1 = r"\\tib.tibub.de\DFS0\Home\ghodsis\Documents\My Files\Projects\Projects\FairNMF\code\data\School\metadata_2013.txt"

    df = pd.read_csv(path1, delimiter='\t', header=None)
    df = df.drop(columns=[1])
    df = df.replace(['F', 'M', 'Unknown'], [1, 2, 0])
    df = df.set_index(0)

    print(df.loc[34].values[0])

    E = np.genfromtxt(path2 , delimiter=' ').astype(np.int32)
    e = E.shape[0]

    N = np.unique(E[:, :-1])
    n = N.shape[0]

    A = np.zeros([n, n]).astype(np.int32)
    F = np.zeros(n).astype(np.int32)

    for i in range(e):
        l1 = int(np.where(N == E[i, 0])[0])
        l2 = int(np.where(N == E[i, 1])[0])

        F[l1] = df.loc[E[i, 0]].values[0]
        F[l2] = df.loc[E[i, 1]].values[0]

        A[l1, l2] = E[i, 2]
        A[l2, l1] = E[i, 2]

    sm = np.array([5])
    inter = np.setdiff1d(np.arange(n), sm)

    A = A[inter, :]
    A = A[:, inter]
    F = F[inter]
    G1 = nx.from_numpy_array(A, create_using=nx.Graph, parallel_edges=False)
    F_gen = compute_F(F)

    standard_groups = reflow_clusters(F)
    groups = np.transpose(standard_groups)

    predfsc = group_fair_sc(A, F_gen, 3)
    pred_FSC = reflow_clusters(predfsc)
    # calculate balance
    balances_FSC, Bfsc = compute_group_balance(pred_FSC.numpy(), groups.numpy())
    # convert predicted labels to networkx clusters
    coms_FSC = lab2com(pred_FSC)
    # compute modularity
    Qfsc = nx.community.modularity(G1, coms_FSC)

    predsfsc = scalable_fair_sc(A, F_gen, 3)
    pred_sFSC = reflow_clusters(predsfsc)
    # calculate balance
    balances_sFSC, Bsfsc = compute_group_balance(pred_sFSC.numpy(), groups.numpy())
    # convert predicted labels to networkx clusters
    coms_sFSC = lab2com(pred_sFSC)
    # compute modularity

    Qsfsc = nx.community.modularity(G1, coms_sFSC)

    print("ok: FSC is Q={}, B={}".format(Qfsc,Bfsc))
    print("ok: sFSC is Q={}, B={}".format(Qsfsc, Bsfsc))

    """
    np.random.seed(42)
    n = 100
    # Generate a random symmetric affinity (adjacency) matrix.
    W = np.random.rand(n, n)
    W = (W + W.T) / 2

    # Create a sample fairness matrix:
    # For example, if there are 2 groups then fair_mat is (n x 1).
    groups = np.random.randint(0, 2, size=n)
    fair_mat = groups.reshape(-1, 1).astype(float)
    k = 3  # number of clusters

    # Run standard spectral clustering.
    clusters_normal = normal_sc(W, k, normalize_laplacian=False, normalize_evec=False)
    print("Standard Spectral Clustering clusters:")
    print(clusters_normal)

    # Run fair spectral clustering (FairSC approach).
    clusters_fair = group_fair_sc(W, fair_mat, k, normalize_laplacian=False, normalize_evec=False)
    print("\nFair Spectral Clustering clusters:")
    print(clusters_fair)

    # Run scalable fair spectral clustering.
    clusters_scalable = scalable_fair_sc(W, fair_mat, k, tol_eig=1e-8)
    print("\nScalable Fair Spectral Clustering clusters:")
    print(clusters_scalable)
    """