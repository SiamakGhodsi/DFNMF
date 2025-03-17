import numpy as np
import matplotlib.pyplot as plt

# Define file paths (adjust the path if necessary)
path1 = "/nfs/home/ghodsis/Projects/FairNMF/code/data/Pokec/"
path_out = path1 + "pre_processed/n/"
feature_file = path_out + "Pokecfeature_reg1.csv"
label_file   = path_out + "Pokeclabel_reg1.csv"

# -------------------------------------------------------------------
# 1. Load the feature (protected attribute groups) and label data
# -------------------------------------------------------------------
features = np.loadtxt(feature_file, delimiter=',', dtype=np.int32)
labels   = np.loadtxt(label_file, delimiter=',', dtype=np.int32)

# -------------------------------------------------------------------
# 2. Compute statistics for the protected attribute groups (features)
# -------------------------------------------------------------------
unique_groups, counts_groups = np.unique(features, return_counts=True)
# Balance: ratio of the smallest group count to the largest group count
balance = np.min(counts_groups) / np.max(counts_groups)

print("Protected Attribute Groups (Age groups):")
for group, count in zip(unique_groups, counts_groups):
    print(f"  Group {group}: {count} samples")
print(f"Dataset balance (min count / max count) for protected attribute: {balance:.4f}")

# -------------------------------------------------------------------
# 3. Compute statistics for the class labels
# -------------------------------------------------------------------
unique_labels, counts_labels = np.unique(labels, return_counts=True)
print("\nClass Labels:")
for lab, count in zip(unique_labels, counts_labels):
    print(f"  Label {lab}: {count} samples")

# -------------------------------------------------------------------
# 4. Visualize the distributions using histograms
# -------------------------------------------------------------------
plt.figure(figsize=(14, 6))

# Histogram for Protected Attribute Groups
plt.subplot(1, 2, 1)
plt.bar(unique_groups, counts_groups, color='skyblue', edgecolor='black')
plt.xlabel("Protected Attribute Group (Age)")
plt.ylabel("Number of Samples")
plt.title("Distribution of Age Groups")

# Histogram for Class Labels
plt.subplot(1, 2, 2)
plt.bar(unique_labels, counts_labels, color='salmon', edgecolor='black')
plt.xlabel("Class Label")
plt.ylabel("Number of Samples")
plt.title("Distribution of Class Labels")

plt.tight_layout()
# Save the figure to a file
plt.savefig(path_out + "Pokec_group_label_distribution.png")
plt.show()