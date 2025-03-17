import torch
import numpy as np
from utils.utils import *
from models.sc_helpers import *
from models.nmf_helpers import *

def ifnmtf(adj_mat: np.ndarray, k: int, lam: float, groups: np.ndarray, eps: float, iter: int):
    """
    :param adj_mat: (n, n) is the adjacency matrix: symmetric matrix containing node connections i.e. the edges of the graph
    :param k: the number of clusters >= 2 that the algorithm is supposed to discover
    :param lam: fir iFairNMTF used to tune the lambda fairness regularizer
    :param groups: (n, ) the sensitive attribute containing group memberships from the data (for iFairNMTF)
    :param eps: epsillon adequecy of the system
    :param iter: number of iterations
    :return predNMF: (n, k) maximum membership, indicating belonging to only one cluster with the highest score
    :return W: (k, k) cluster-cluster interaction matrix, which contains fuzzy degress of cluster connectivity
    :return err: (iter, ) model loss as per iteration
    """
    eps = torch.tensor(eps)
    n = np.shape(adj_mat)[0]
    A = torch.tensor(adj_mat, dtype=torch.float)
    H = torch.rand(n, k, dtype=torch.float)  # the membership degree matrix (H) initialization

    # Co-cluster factor initialization by eigen_values of the Adjacency matrix
    W = torch.rand(k, k, dtype=torch.float)
    # W = svd_init(A, k)

    if lam == 0:
        # For vanilla NMTF, define dummy zero matrices.
        L = torch.zeros(n, n, dtype=torch.float)
        Ln = torch.zeros(n, n, dtype=torch.float)
        Lp = torch.zeros(n, n, dtype=torch.float)
    else:
        # For fairness, compute the Laplacian of the fairness contrastive matrix.
        L, Ln, Lp = joint_Laplacian(groups) # Lp positive and Ln negative parts

    err = torch.zeros(iter)
    for t in range(iter):
        Hn = (A.T @ H @ W + A @ H @ W.T + lam * (Ln @ H))
        Hd = H @ W.T @ H.T @ H @ W + H @ W @ H.T @ H @ W.T + lam * ((Lp) @ H)#(RTR @ H)
        H = H * (Hn / torch.maximum(Hd, eps)) ** 0.25

        Wn = H.T @ A @ H
        Wd = H.T @ H @ W @ H.T @ H
        W = W * (Wn / torch.maximum(Wd, eps))

        err[t] = torch.norm(A - H @ W @ H.T) ** 2 + lam * torch.trace(H.T @ L @ H)

    # transform fuzzy memberships (overlaps) to strict (disjoint)
    predNMF = torch.argmax(H, dim=1)

    return predNMF, W, err

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
    clusters = kmeans(eigenvectors, num_clusters, normalize_evec)
    return clusters

def ind_fair_sc(adj_mat: np.ndarray, groups: np.ndarray, k: int,
                normalize_laplacian: bool = False, normalize_evec: bool = False) -> np.ndarray:
    """
    :param adj_mat: (num_nodes, num_nodes) Adjacency matrix of the observed graph
    :param groups: (num_nodes,) An array indicating sensitive group membership
    :param k: Number of clusters to discover
    :param normalize_laplacian: Whether to use normalized Laplacian or not
    :param normalize_evec: Whether to normalize the rows of eigenvector matrix before running k-means
    :return clusters: (num_nodes,) The cluster assignment for each node
    """
    # Compute the constraint matrix
    R = compute_RD(groups)         # Representation_graph adjacency mat: A (n x n) graph specifying
                                   # which node can represent which other nodes
    Z = null_space(R)              # null_space_basis
    assert Z.shape[1] >= k, 'Rank of Z, the constraint matrix is too high'

    # Compute the Laplacian
    L, D = compute_laplacian(adj_mat, normalize_laplacian=False)
    if normalize_laplacian:
        Q = np.real(sqrtm(Z.T @ D @ Z))
        Q_inv = np.linalg.inv(1e-6 * np.eye(Q.shape[0]) + Q)
    else:
        # Step 3: Compute Q = sqrtm(Z.T @ D @ Z)
        Q = np.real(sqrtm(Z.T @ D @ Z))
        # To avoid numerical issues, add a small identity term.
        Q_inv = np.linalg.inv(Q + 1e-6 * np.eye(Q.shape[0]))

    # Step 4: Scale Z: Z̃ = Z @ Q_inv
    Z_tilde = Z @ Q_inv

    # Step 5: Form M = Z̃.T @ L @ Z̃ and symmetrize
    LL = Z_tilde.T @ L @ Z_tilde
    LL = (LL + LL.T) / 2.0

    # Compute eigenvectors
    Y = compute_top_eigen(LL, k)
    YY = np.matmul(Z, Y)

    # Run k-means
    clusters = kmeans(YY, k, normalize_evec)
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
    clusters = kmeans(H, num_clusters, normalize_rows=normalize_evec)
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
    clusters = kmeans(H, num_clusters, normalize_rows=False)
    return clusters

#-------------------------------------------- Last update sFSC ----------------------------------------------

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
    clusters = kmeans(H, num_clusters, normalize_rows=False)
    return clusters

