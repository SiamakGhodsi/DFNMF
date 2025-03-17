
from utils import *
from evaluations import *
import torch
import numpy as np
import scipy.io as sio

eps = torch.tensor(0.000001)

#path = "data/LastFM/"
#A = sio.loadmat(path + 'LastFM.mat')['A'].toarray()
#F0 = sio.loadmat(path + 'LastFM.mat')['F'].flatten()

path = r"\\tib.tibub.de\DFS0\Home\ghodsis\Documents\My Files\Projects\Projects\FairNMF\code\data\LastFM\LastFM.mat"

A = sio.loadmat(path)['A'].toarray()
F0 = sio.loadmat(path)['F'].flatten()
F = np.unique(F0, return_inverse=True)[1]

k = 5
lambdas = [1, 10]

Adj = torch.tensor(A, dtype=torch.float)
standard_groups = reflow_clusters(F)
groups = np.transpose(standard_groups)
F_country = compute_F(groups)
F1 = torch.tensor(F_country, dtype=torch.float)
FF = F1 @ F1.T

n = A.shape[0]

for idx, lam in enumerate(lambdas):
    iter = 100
    # Fair_NMF
    H = torch.rand(n, k, dtype=torch.float)  # the membership degree matrix (H) initialization
    FFp = (torch.abs(FF) + FF) / 2
    FFn = (torch.abs(FF) - FF) / 2

    err_lfm = torch.zeros((2, iter))
    for t in range(iter):
        Hu = Adj @ H + (lam / 2) * (FFn @ H)
        Hd = H @ H.T @ H + (lam / 2) * (FFp @ H)
        H = H * (Hu / torch.maximum(Hd, eps)) ** 0.25
        err_lfm[idx, t] = torch.norm(Adj - H @ H.T) ** 2 + lam * torch.norm(F1.T @ H) ** 2


print("Successfully finished")

