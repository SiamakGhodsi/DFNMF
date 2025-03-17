import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyecharts.charts import Bar, Pie
from pyecharts import options as opts

# Define the input directory (adjust the path as needed)
path_in = "data/Pokec/pre_processed/"
path1 = "/nfs/home/ghodsis/Projects/FairNMF/code/data/Pokec/pre_processed/n/"

# -------------------------------------------------------------------
# 1. Load the saved feature and label files.
#    - Pokecfeature_reg1.csv contains the binned age groups.
#    - Pokeclabel_reg1.csv contains the class labels.
# -------------------------------------------------------------------
features = np.loadtxt(path1 + "Pokecfeature_reg1.csv", delimiter=",", dtype=int)
labels   = np.loadtxt(path1 + "Pokeclabel_reg1.csv", delimiter=",", dtype=int)

# Compute unique values and counts for Age Groups and Class Labels.
unique_groups, group_counts = np.unique(features, return_counts=True)
unique_labels, label_counts = np.unique(labels, return_counts=True)

# -------------------------
# 2. Visualizations for Age Groups
# -------------------------

# Vertical Bar Chart for Age Groups
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(unique_groups, group_counts, color='skyblue', edgecolor='black')
ax.set_xlabel("Age Group", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Age Groups", fontsize=14)
for i, cnt in enumerate(group_counts):
    ax.text(unique_groups[i], cnt, str(cnt), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(path1 + "AgeGroups_Bar.svg", format='svg')
plt.savefig(path1 + "AgeGroups_Bar.pdf", format='pdf')
plt.close()

# Pie Chart for Age Groups
fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_groups)))
ax.pie(group_counts, labels=[f"Group {g}" for g in unique_groups], autopct='%1.1f%%',
       startangle=90, colors=colors, textprops={'fontsize': 10})
ax.axis('equal')
ax.set_title("Age Groups Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(path1 + "AgeGroups_Pie.svg", format='svg')
plt.savefig(path1 + "AgeGroups_Pie.pdf", format='pdf')
plt.close()

# Horizontal Bar Chart for Age Groups (with annotations)
fig, ax = plt.subplots(figsize=(8,6))
y_pos = np.arange(len(unique_groups))
ax.barh(y_pos, group_counts, color='lightgreen', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Group {g}" for g in unique_groups], fontsize=12)
ax.set_xlabel("Count", fontsize=12)
ax.set_title("Distribution of Age Groups (Horizontal)", fontsize=14)
for i, cnt in enumerate(group_counts):
    ax.text(cnt, i, f' {cnt}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(path1 + "AgeGroups_HorizontalBar.svg", format='svg')
plt.savefig(path1 + "AgeGroups_HorizontalBar.pdf", format='pdf')
plt.close()

# -------------------------
# 3. Visualizations for Class Labels
# -------------------------

# Vertical Bar Chart for Class Labels
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(unique_labels, label_counts, color='salmon', edgecolor='black')
ax.set_xlabel("Class Label", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Class Labels", fontsize=14)
for i, cnt in enumerate(label_counts):
    ax.text(unique_labels[i], cnt, str(cnt), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(path1 + "ClassLabels_Bar.svg", format='svg')
plt.savefig(path1 + "ClassLabels_Bar.pdf", format='pdf')
plt.close()

# Pie Chart for Class Labels
fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.Pastel2(np.linspace(0, 1, len(unique_labels)))
ax.pie(label_counts, labels=[f"Label {l}" for l in unique_labels], autopct='%1.1f%%',
       startangle=90, colors=colors, textprops={'fontsize': 10})
ax.axis('equal')
ax.set_title("Class Labels Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(path1 + "ClassLabels_Pie.svg", format='svg')
plt.savefig(path1 + "ClassLabels_Pie.pdf", format='pdf')
plt.close()

# Horizontal Bar Chart for Class Labels (with annotations)
fig, ax = plt.subplots(figsize=(8,6))
y_pos = np.arange(len(unique_labels))
ax.barh(y_pos, label_counts, color='violet', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Label {l}" for l in unique_labels], fontsize=12)
ax.set_xlabel("Count", fontsize=12)
ax.set_title("Distribution of Class Labels (Horizontal)", fontsize=14)
for i, cnt in enumerate(label_counts):
    ax.text(cnt, i, f' {cnt}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(path1 + "ClassLabels_HorizontalBar.svg", format='svg')
plt.savefig(path1 + "ClassLabels_HorizontalBar.pdf", format='pdf')
plt.close()

print("Static vectorized charts (SVG and PDF) for Age Groups and Class Labels have been saved.")