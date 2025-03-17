import numpy as np

# Define the input directory (adjust the path as needed)
path = "C:/Users/ghodsis/Desktop/FairNMF/code/data/Pokec/pre_processed/z/"

# -------------------------------------------------------------------
# 1. Load the saved feature and label files.
#    - Pokecfeature_reg1.csv contains the binned age groups.
#    - Pokeclabel_reg1.csv contains the class labels.
# -------------------------------------------------------------------
labels = np.loadtxt(path + "Pokec_label_reg2.csv", delimiter=",", dtype=int)

unique_labels, counts = np.unique(labels, return_counts=True)
majority_label = unique_labels[np.argmax(counts)]
print("Majority label is:", majority_label)

# Binarize the labels: majority becomes 1, others become 0.
binary_labels = np.where(labels == majority_label, 1, 0)

# Save the binarized labels to a CSV file.
np.savetxt(path + "Pokec_binary_label_reg2.csv", binary_labels, fmt='%d', delimiter=',')

print("Binarized class labels saved to:", path + "Pokec_binary_label_reg2.csvv")