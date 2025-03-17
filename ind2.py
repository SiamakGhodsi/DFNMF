import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from utils import *
from evaluations import *
from algorithms import *
from competitors import *
from data import *
import os
import time
import networkx as nx

eps = torch.tensor(0.000001)

## GPU or CPU
dtype = data_type()

# -------------------------------------------------- grid-search --------------------------------------------------
k = list(range(2, 10))
lambdas = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 100]
#lambdas = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3 ,5, 7.5, 10, 15, 20, 30,
#           40, 50, 75, 100, 150, 200, 300, 400, 500]

names = ["Diaries", "Facebook", "Friendship", "DrugNET_ethnicity", "NBA", "LastFM"]
for name in names:
    if (name == "Diaries"):
        A, F = load_diaries()
    elif (name == "Facebook"):
        A, F = load_facebook()
    elif (name == "Friendship"):
        A, F = load_friendship()
    elif (name == "DrugNET_ethnicity"):
        A, F = load_drugnet()
    elif (name == "NBA"):
        A, F = load_nba()
    elif (name == "LastFM"):
        A, F = load_lfm()

    n = A.shape[0]
    Adj = torch.tensor(A, dtype=torch.float)
    standard_groups = reflow_clusters(F.ravel())
    groups = np.transpose(standard_groups)
    L, Ln, Lp = joint_Laplacian(groups)

    G1 = nx.from_numpy_array(A, create_using=nx.Graph, parallel_edges=False)

    #DrugNET comparisons
    df = pd.DataFrame(columns=['network', 'attr', 'method', 'nodes (n)', 'clusters (k)', 'groups (h)',
                'avg_rho', 'min_rho', 'ind_rhos', 'avg_alpha', 'min_alpha', 'Cluster_alphas', 'clusters'])
    for num_c in k:

        #num_repeats = 3
        num_repeats = 1
        rho_min_iter_ifsc, rho_avg_iter_ifsc, rho_iter_ifsc, rho_min_iter_fsc, rho_avg_iter_fsc, rho_iter_fsc, \
        rho_min_iter_sc, rho_avg_iter_sc, rho_iter_sc = ([] for ii in range(9))

        alpha_min_iter_ifsc, alpha_avg_iter_ifsc, alpha_c_iter_ifsc, alpha_min_iter_fsc, alpha_avg_iter_fsc,\
        alpha_c_iter_fsc, alpha_min_iter_sc, alpha_avg_iter_sc, alpha_c_iter_sc = ([] for ii in range(9))

        for param in range(num_repeats):
            # Individual Fair_SC
            predifsc = ind_fair_sc(A, groups, num_c)
            pred_IFSC = reflow_clusters(predifsc)
            # calculate individual balance
            b_i_IFSC, b_avg_ind_IFSC, b_min_ind_IFSC = compute_individual_balance(pred_IFSC.numpy(), groups.numpy())
            rho_min_iter_ifsc.append(b_min_ind_IFSC);
            rho_avg_iter_ifsc.append(b_avg_ind_IFSC);
            rho_iter_ifsc.append(b_i_IFSC)
            # calculate group balance
            b_clust_IFSC, b_avg_IFSC, b_min_IFSC = compute_group_balance(pred_IFSC.numpy(), groups.numpy())
            alpha_c_iter_ifsc.append(b_clust_IFSC);
            alpha_avg_iter_ifsc.append(b_avg_IFSC);
            alpha_min_iter_ifsc.append(b_min_IFSC);

            # Fair_SC
            predfsc = group_fair_sc(A, groups, num_c)
            pred_FSC = reflow_clusters(predfsc)
            # calculate individual balance
            b_i_FSC, b_avg_ind_FSC, b_min_ind_FSC = compute_individual_balance(pred_FSC.numpy(), groups.numpy())
            rho_min_iter_fsc.append(b_min_ind_FSC);
            rho_avg_iter_fsc.append(b_avg_ind_FSC);
            rho_iter_fsc.append(b_i_FSC)
            # calculate group balance
            b_clust_FSC, b_avg_FSC, b_min_FSC = compute_group_balance(pred_FSC.numpy(), groups.numpy())
            alpha_c_iter_fsc.append(b_clust_FSC);
            alpha_avg_iter_fsc.append(b_avg_FSC);
            alpha_min_iter_fsc.append(b_min_FSC);

            # SC
            predsc = normal_sc(A, num_c)
            pred_SC = reflow_clusters(predsc)
            # calculate individual balance
            b_i_SC, b_avg_ind_SC, b_min_ind_SC = compute_individual_balance(pred_SC.numpy(), groups.numpy())
            rho_min_iter_sc.append(b_min_ind_SC);
            rho_avg_iter_sc.append(b_avg_ind_SC);
            rho_iter_sc.append(b_i_FSC)
            # calculate group balance
            b_clust_SC, b_avg_SC, b_min_SC = compute_group_balance(pred_SC.numpy(), groups.numpy())
            alpha_c_iter_sc.append(b_clust_SC);
            alpha_avg_iter_sc.append(b_avg_SC);
            alpha_min_iter_sc.append(b_min_SC);

        rho_min_ifsc = sum(rho_min_iter_ifsc) / num_repeats
        rho_avg_ifsc = sum(rho_avg_iter_ifsc) / num_repeats
        alpha_min_ifsc = sum(alpha_min_iter_ifsc) / num_repeats
        alpha_avg_ifsc = sum(alpha_avg_iter_ifsc) / num_repeats

        rho_min_fsc = sum(rho_min_iter_fsc) / num_repeats
        rho_avg_fsc = sum(rho_avg_iter_fsc) / num_repeats
        alpha_min_fsc = sum(alpha_min_iter_fsc) / num_repeats
        alpha_avg_fsc = sum(alpha_avg_iter_fsc) / num_repeats

        rho_min_sc = sum(rho_min_iter_sc) / num_repeats
        rho_avg_sc = sum(rho_avg_iter_sc) / num_repeats
        alpha_min_sc = sum(alpha_min_iter_sc) / num_repeats
        alpha_avg_sc = sum(alpha_avg_iter_sc) / num_repeats

        rho_i_ifsc = b_i_IFSC; rho_i_fsc = b_i_FSC  ; rho_i_sc = b_i_SC
        alpha_c_ifsc = alpha_c_iter_ifsc; alpha_c_fsc = alpha_c_iter_fsc; alpha_c_sc =alpha_c_iter_sc

        # row of results to be appended to df
        col1 = [name for i in range(3)]
        col2 = ["Gender" for i in range(3)]
        col3 = ["ifair_sc", "fair_sc", "vanilla_sc"]
        col4 = [n for i in range(3)]
        col5 = [num_c for i in range(3)]
        col6 = [len(np.unique(groups)) for i in range(3)]
        col7 = [rho_avg_ifsc, rho_avg_fsc, rho_avg_sc]
        col8 = [rho_min_ifsc, rho_min_fsc, rho_min_sc]
        col9 = [rho_i_ifsc.tolist(), rho_i_fsc.tolist(), rho_i_sc.tolist()]
        col10= [alpha_avg_ifsc, alpha_avg_fsc, alpha_avg_sc]
        col11= [alpha_min_ifsc, alpha_min_fsc, alpha_min_sc]
        col12= [alpha_c_ifsc, alpha_c_fsc, alpha_c_sc]
        col13= [pred_IFSC.tolist(), pred_FSC.tolist(), pred_SC.tolist()]
        results = dict()
        results = {'network': col1, 'attr': col2, 'method': col3, 'nodes (n)': col4, 'clusters (k)': col5,
                   'groups (h)': col6, 'avg_rho': col7, 'min_rho': col8, 'ind_rhos': col9,
                   'avg_alpha': col10, 'min_alpha': col11, 'Cluster_alphas': col12, 'clusters': col13}
        temp = pd.DataFrame((results))
        df = pd.concat([df,temp], ignore_index=True)
        print(temp)

    print(df)
    df.to_csv(name +'2.csv', index=False)