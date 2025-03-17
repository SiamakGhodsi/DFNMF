import numpy as np
from scipy.linalg import null_space, eigh, sqrtm
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.cluster import KMeans


def kmeans(features: np.ndarray, num_clusters: int, normalize_rows: bool = False) -> np.ndarray:
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

def compute_RD(sensitive):
    """
    :param sensitive: (num_nodes,) Vector indicating protected group memberships
    :return R: nxn Representation graph constraint
    """
    n = len(sensitive)
    # counting number of protected groups
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)

    group_one_hot = np.eye(h)[sensitive, :]
    R = 1- np.matmul(group_one_hot, group_one_hot.T)
    #diag = np.eye((n))

    #R = similarity_matrix - diag
    R_normal = R/R.sum(axis=1, keepdims=True)
    return R_normal


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



