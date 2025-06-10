import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import scipy.sparse as sparse
import copy



# ------------------------------------------------ Pokec --------------------------------------------------
# Define input and output paths
path1 = "/nfs/home/ghodsis/Projects/FairNMF/code/data/Pokec/"
path_out = path1 + "pre_processed/z/"

# -------------------------------------------------------------------
# 1. Load node data from region_job.csv
#    Expected columns: 
#       - Column 0: Node ID
#       - Column 4: Raw age
#       - Column 5: Class label
# -------------------------------------------------------------------
data = np.genfromtxt(path1 + 'region_job2.csv', delimiter=',', skip_header=1, dtype=np.int32)
node_ids    = data[:, 0]
raw_ages    = data[:, 4]
class_labels = data[:, 5]

# -------------------------------------------------------------------
# 2. Bin ages into four groups:
#       Group 1: Age <= 18
#       Group 2: Age 19 to 25
#       Group 3: Age 26 to 35
#       Group 4: Age >= 36
# -------------------------------------------------------------------
binned_age = np.empty_like(raw_ages)
binned_age[raw_ages <= 18] = 1
binned_age[(raw_ages >= 19) & (raw_ages <= 25)] = 2
binned_age[(raw_ages >= 26) & (raw_ages <= 35)] = 3
binned_age[raw_ages >= 36] = 4

# Create dictionaries for quick lookup of age and label by node ID
age_dict   = {node: age for node, age in zip(node_ids, binned_age)}
label_dict = {node: lab for node, lab in zip(node_ids, class_labels)}

# -------------------------------------------------------------------
# 3. Load relationship data from region_job_relationship.txt
#    This file contains tab-separated pairs of node IDs.
# -------------------------------------------------------------------
E = np.genfromtxt(path1 + 'region_job_relationship2.txt', delimiter='\t', dtype=np.int32)
# Get the unique node IDs present in the edge list
N = np.unique(E)
n = N.shape[0]

# Build a mapping from node ID to index (for constructing matrices)
node2idx = {node: idx for idx, node in enumerate(N)}

# -------------------------------------------------------------------
# 4. Initialize the adjacency matrix (A) and feature/label arrays.
# -------------------------------------------------------------------
A = np.zeros((n, n), dtype=np.int32)  # Adjacency matrix
F = np.zeros(n, dtype=np.int32)         # Protected attribute (binned age)
labels = np.zeros(n, dtype=np.int32)    # Class labels

# -------------------------------------------------------------------
# 5. Process each edge to:
#       - Update the adjacency matrix (undirected graph)
#       - Assign the binned age and label for each node
# -------------------------------------------------------------------
for edge in E:
    node1, node2 = edge
    idx1 = node2idx[node1]
    idx2 = node2idx[node2]
    
    # Set the undirected edge in the adjacency matrix
    A[idx1, idx2] = 1
    A[idx2, idx1] = 1
    
    # Assign protected attribute and label based on our dictionaries.
    # If a node appears multiple times, the values should be consistent.
    F[idx1] = age_dict[node1]
    F[idx2] = age_dict[node2]
    
    labels[idx1] = label_dict[node1]
    labels[idx2] = label_dict[node2]

# -------------------------------------------------------------------
# 6. Save the processed data:
#       - Convert the dense adjacency matrix to a sparse format and save it.
#       - Save the dense CSVs for the graph, features, and labels.
# -------------------------------------------------------------------
A_sparse = sparse.csc_matrix(A)
sparse.save_npz(path_out + "sparse_Pokec_v2_graph.npz", A_sparse)
np.savetxt(path_out + "Pokec_v2_graph.csv", A, fmt='%d', delimiter=',')
np.savetxt(path_out + "Pokec_v2_feature.csv", F, fmt='%d', delimiter=',')
np.savetxt(path_out + "Pokec_v2_label.csv", labels, fmt='%d', delimiter=',')

print("Pokec v2 pre-processing complete.")

all_in_one = np.ones(F.shape[0])
uniqe_vals, count = np.unique(F, return_counts=True)
Pokec_balance = min(count)/max(count)

print("Dataset balance = ", Pokec_balance)

balance_file = path_out + "Pokec_balance.txt"
with open(balance_file, "w") as f:
    f.write("Dataset balance (min count / max count): {:.4f}\n".format(Pokec_balance))
    f.write("Protected attribute group counts:\n")
    for group, cnt in zip(uniqe_vals, count):
        f.write("  Group {}: {}\n".format(group, cnt))

print("Balance info saved to:", balance_file)

# A_sparse = sparse.load_npz(path1+"pre_prpcessed/sparse_Pokec_graph_region_A.npz")
# A = A_sparse.toarray()
