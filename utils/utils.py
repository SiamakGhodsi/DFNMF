import torch
import numpy as np
import os
from networkx.generators import random_regular_graph
from networkx import to_numpy_array
import pickle
import copy
import scipy as sp


def nullspace(At, rcond=None):
    """A = At.numpy()
    ns = sp.linalg.null_space(A, rcond=None)"""

    # return the nullspace of matrix F to constitute Z
    ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)
    vht=vht.T
    Mt, Nt = ut.shape[0], vht.shape[1]
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    numt= torch.sum(st > tolt, dtype=int)
    ns = vht[numt:,:].T.conj()
    return ns

# original balance of "Chierichetti et. al. 2017" for k-means (data/feature clustering)
def balance_calculation(data, centers, mapping):
	"""
	Checks fairness for each of the clusters defined by k-centers.
	Returns balance using the total and class counts.
	
	Args:
		data (list)
		centers (list)
		mapping (list) : tuples of the form (data, center)
		
	Returns:
		fair (dict) : key=center, value=(sum of 1's corresponding to fairness variable, number of points in center)
	"""
	fair = dict([(i, [0, 0]) for i in centers])
	for i in mapping:
		fair[i[1]][1] += 1
		if data[i[0]][0] == 1: # MARITAL
			fair[i[1]][0] += 1

	curr_b = []
	capacity = []
	for i in list(fair.keys()):
		p = fair[i][0]
		q = fair[i][1] - fair[i][0]
		if p == 0 or q == 0:
			balance = 0
		else:
			balance = min(float(p/q), float(q/p))
		curr_b.append(balance)

		### Print data
		#print(i, ",", p, ",", q,",",balance)
		capacity.append(p+q)

	return min(curr_b), capacity

def compute_F(sensitive):
    """
    :param groups: (num_nodes,) Vector indicating protected group memberships
    :return F: Group_fairness constraint matrix as in Kleindesnner
    """
    n = len(sensitive)
    # converting sensitive to a vector with entries in [h] and building F
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)
    #sens_unique = reshape(sens_unique, [1, h]);

    sensitiveNEW = copy.deepcopy(sensitive)

    temp = 0;
    for i in sens_unique:
        ind = np.where(np.isin(sensitive, i))
        sensitiveNEW[ind] = temp
        temp = temp + 1;

    F = np.zeros((n, h - 1));

    for ell in range(h - 1):
        temp = np.where(np.isin(sensitiveNEW, ell))
        F[temp[0], ell] = 1;
        groupSize = len(temp[0]);
        F[:, ell] = F[:, ell] - groupSize/n;

    return F

