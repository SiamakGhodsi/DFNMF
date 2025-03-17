import numpy as np
import matplotlib.pyplot as plt

# Define the NBA preprocessed data directory.
# (Adjust this path to where your NBA files are saved.)
path_nba = "C:/Users/ghodsis/Desktop/FairNMF/code/data/NBA/"

# -------------------------------
# 1. Load the saved files:
#    - NBAfeature.csv contains the protected attribute (features).
#    - NBAlabel.csv contains the original class labels.
#    - NBAlabel_binary.csv contains the binarized class labels.
# -------------------------------
features = np.loadtxt(path_nba + "NBAfeature.csv", delimiter=",", dtype=int)
labels   = np.loadtxt(path_nba + "NBAlabel.csv", delimiter=",", dtype=int)
binary_labels = np.loadtxt(path_nba + "NBAlabel_binary.csv", delimiter=",", dtype=int)

# -------------------------------
# 2. Compute unique values and counts.
# -------------------------------
# For protected attribute groups (features)
unique_groups, group_counts = np.unique(features, return_counts=True)

# For original labels
unique_labels, label_counts = np.unique(labels, return_counts=True)

# For binary labels
unique_binary, binary_counts = np.unique(binary_labels, return_counts=True)

# -------------------------------
# 3. Visualizations for NBA Protected Attribute Groups
# -------------------------------

# Vertical Bar Chart for Protected Attribute Groups
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(unique_groups, group_counts, color='skyblue', edgecolor='black')
ax.set_xlabel("NBA Protected Attribute Group", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of NBA Protected Attribute Groups", fontsize=14)
for i, cnt in enumerate(group_counts):
    ax.text(unique_groups[i], cnt, str(cnt), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(path_nba + "NBA_Features_Bar.svg", format='svg')
plt.savefig(path_nba + "NBA_Features_Bar.pdf", format='pdf')
plt.close()

# Pie Chart for Protected Attribute Groups
fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_groups)))
ax.pie(group_counts, labels=[f"Group {g}" for g in unique_groups], autopct='%1.1f%%',
       startangle=90, colors=colors, textprops={'fontsize': 10})
ax.axis('equal')
ax.set_title("NBA Protected Attribute Groups Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(path_nba + "NBA_Features_Pie.svg", format='svg')
plt.savefig(path_nba + "NBA_Features_Pie.pdf", format='pdf')
plt.close()

# Horizontal Bar Chart for Protected Attribute Groups (with annotations)
fig, ax = plt.subplots(figsize=(8,6))
y_pos = np.arange(len(unique_groups))
ax.barh(y_pos, group_counts, color='lightgreen', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Group {g}" for g in unique_groups], fontsize=12)
ax.set_xlabel("Count", fontsize=12)
ax.set_title("NBA Protected Attribute Groups Distribution (Horizontal)", fontsize=14)
for i, cnt in enumerate(group_counts):
    ax.text(cnt, i, f' {cnt}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(path_nba + "NBA_Features_HorizontalBar.svg", format='svg')
plt.savefig(path_nba + "NBA_Features_HorizontalBar.pdf", format='pdf')
plt.close()

# -------------------------------
# 4. Visualizations for NBA Original Class Labels
# -------------------------------

# Vertical Bar Chart for Original Labels
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(unique_labels, label_counts, color='salmon', edgecolor='black')
ax.set_xlabel("NBA Original Class Labels", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of NBA Original Class Labels", fontsize=14)
for i, cnt in enumerate(label_counts):
    ax.text(unique_labels[i], cnt, str(cnt), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(path_nba + "NBA_OriginalLabels_Bar.svg", format='svg')
plt.savefig(path_nba + "NBA_OriginalLabels_Bar.pdf", format='pdf')
plt.close()

# Pie Chart for Original Labels
fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.Pastel2(np.linspace(0, 1, len(unique_labels)))
ax.pie(label_counts, labels=[f"Label {l}" for l in unique_labels], autopct='%1.1f%%',
       startangle=90, colors=colors, textprops={'fontsize': 10})
ax.axis('equal')
ax.set_title("NBA Original Class Labels Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(path_nba + "NBA_OriginalLabels_Pie.svg", format='svg')
plt.savefig(path_nba + "NBA_OriginalLabels_Pie.pdf", format='pdf')
plt.close()

# Horizontal Bar Chart for Original Labels (with annotations)
fig, ax = plt.subplots(figsize=(8,6))
y_pos = np.arange(len(unique_labels))
ax.barh(y_pos, label_counts, color='violet', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Label {l}" for l in unique_labels], fontsize=12)
ax.set_xlabel("Count", fontsize=12)
ax.set_title("NBA Original Class Labels Distribution (Horizontal)", fontsize=14)
for i, cnt in enumerate(label_counts):
    ax.text(cnt, i, f' {cnt}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(path_nba + "NBA_OriginalLabels_HorizontalBar.svg", format='svg')
plt.savefig(path_nba + "NBA_OriginalLabels_HorizontalBar.pdf", format='pdf')
plt.close()

# -------------------------------
# 5. Visualizations for NBA Binary Class Labels
# -------------------------------

# Vertical Bar Chart for Binary Labels
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(unique_binary, binary_counts, color='salmon', edgecolor='black')
ax.set_xlabel("NBA Binary Class Labels", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of NBA Binary Class Labels", fontsize=14)
for i, cnt in enumerate(binary_counts):
    ax.text(unique_binary[i], cnt, str(cnt), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(path_nba + "NBA_BinaryLabels_Bar.svg", format='svg')
plt.savefig(path_nba + "NBA_BinaryLabels_Bar.pdf", format='pdf')
plt.close()

# Pie Chart for Binary Labels
fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.Pastel2(np.linspace(0, 1, len(unique_binary)))
ax.pie(binary_counts, labels=[f"Label {l}" for l in unique_binary], autopct='%1.1f%%',
       startangle=90, colors=colors, textprops={'fontsize': 10})
ax.axis('equal')
ax.set_title("NBA Binary Class Labels Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(path_nba + "NBA_BinaryLabels_Pie.svg", format='svg')
plt.savefig(path_nba + "NBA_BinaryLabels_Pie.pdf", format='pdf')
plt.close()

# Horizontal Bar Chart for Binary Labels (with annotations)
fig, ax = plt.subplots(figsize=(8,6))
y_pos = np.arange(len(unique_binary))
ax.barh(y_pos, binary_counts, color='violet', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Label {l}" for l in unique_binary], fontsize=12)
ax.set_xlabel("Count", fontsize=12)
ax.set_title("NBA Binary Class Labels Distribution (Horizontal)", fontsize=14)
for i, cnt in enumerate(binary_counts):
    ax.text(cnt, i, f' {cnt}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(path_nba + "NBA_BinaryLabels_HorizontalBar.svg", format='svg')
plt.savefig(path_nba + "NBA_BinaryLabels_HorizontalBar.pdf", format='pdf')
plt.close()

print("Static vectorized charts for NBA features, original labels, and binary labels have been saved (SVG and PDF).")
