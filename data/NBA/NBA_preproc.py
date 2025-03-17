import numpy as np
from scipy.sparse import save_npz

# Define paths
path1 = "C:/Users/ghodsis/Desktop/FairNMF/code/data/NBA/"
path_out = path1 + "pre_processed/n/"

# -------------------------------
# 1. Load attribute and label data from nba.csv.
#    att: columns [node_id, protected_attribute] from column 0 and 37.
#    lab: columns [node_id, class_label] from column 0 and 1.
# -------------------------------
data = np.genfromtxt(path1 + 'nba.csv', delimiter=',', skip_header=1, dtype=np.int64)
att = data[:, [0, 37]]
lab = data[:, [0, 1]]

# Build dictionaries for quick lookup.
att_dict = {row[0]: row[1] for row in att}
lab_dict = {row[0]: row[1] for row in lab}

# -------------------------------
# 2. Load relationship data from nba_relationship.txt.
# -------------------------------
E = np.genfromtxt(path1 + 'nba_relationship.txt', delimiter='\t', dtype=np.int64)
# Unique node IDs (as in the relationships)
N = np.unique(E)
n = N.shape[0]
# Create a mapping from node ID to index.
node2idx = {node: idx for idx, node in enumerate(N)}

# -------------------------------
# 3. Build the adjacency matrix A.
#    Here we vectorize the edge processing.
# -------------------------------
A = np.zeros((n, n), dtype=np.int64)
indices1 = np.array([node2idx[node] for node in E[:, 0]])
indices2 = np.array([node2idx[node] for node in E[:, 1]])
A[indices1, indices2] = 1
A[indices2, indices1] = 1

# -------------------------------
# 4. Construct the feature vector F and label vector L.
#    Iterate over the unique nodes only once.
# -------------------------------
F = np.zeros(n, dtype=np.int64)
L = np.zeros(n, dtype=np.int64)
for i, node in enumerate(N):
    F[i] = att_dict.get(node, 0)  # default to 0 if missing
    L[i] = lab_dict.get(node, 0)

# Save the graph and features.
np.savetxt(path1 + "NBAgraph.csv", A, fmt='%d', delimiter=',')
np.savetxt(path1 + "NBAfeature.csv", F, fmt='%d', delimiter=',')
np.savetxt(path1 + "NBAlabel.csv", L, fmt='%d', delimiter=',')

# -------------------------------
# 5. Binarize the class labels.
#    For example, set the majority class to 1 and all others to 0.
# -------------------------------
unique_labels, counts = np.unique(L, return_counts=True)
majority_label = unique_labels[np.argmax(counts)]
binary_labels = np.where(L == majority_label, 1, 0)
np.savetxt(path1 + "NBAlabel_binary.csv", binary_labels, fmt='%d', delimiter=',')

print("NBA pre-processing complete.")
print("Majority label is:", majority_label)