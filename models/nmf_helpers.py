import numpy as np
import torch

def compute_RS(sensitive):
    """
    :param sensitive: (num_nodes,) Vector indicating protected group memberships
    :return R: nxn Representation graph constraint
    """
    n = len(sensitive)
    # counting number of protected groups
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)

    group_one_hot = np.eye(h)[sensitive, :]
    similarity_matrix = np.matmul(group_one_hot, group_one_hot.T)
    diag = np.eye((n))

    R = similarity_matrix - diag
    R_normal = R/R.sum(axis=1, keepdims=True)
    return R_normal


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

def joint_Laplacian(groups):
    """
    :param groups: (num_nodes,) Vector indicating protected group memberships
    :return L: nxn Representation graph constraint
    """

    RS = compute_RS(groups)
    RD = compute_RD(groups)

    R1D = torch.tensor(RD, dtype=torch.float)
    R1S = torch.tensor(RS, dtype=torch.float)
    R = R1D - R1S
    L = torch.diag(torch.sum(R, dim=1)) - R
    Lp = (torch.abs(L) + L) / 2
    Ln = (torch.abs(L) - L) / 2
    return L, Ln, Lp