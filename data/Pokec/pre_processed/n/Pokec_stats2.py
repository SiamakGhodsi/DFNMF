import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyecharts.charts import Bar, Pie
from pyecharts import options as opts

# Define the input directory (adjust the path as needed)
path_in = "data/Pokec/pre_processed/"
path1 = "/nfs/home/ghodsis/Projects/FairNMF/code/data/Pokec/pre_processed/n/"

# -------------------------------------------------------------------
# 1. Load the feature and label files.
#    - 'Pokecfeature_reg1.csv' contains the binned Age groups.
#    - 'Pokeclabel_reg1.csv' contains the class labels.
# -------------------------------------------------------------------
features = np.loadtxt(path1 + "Pokecfeature_reg1.csv", delimiter=",", dtype=int)
labels = np.loadtxt(path1 + "Pokeclabel_reg1.csv", delimiter=",", dtype=int)

# -------------------------------------------------------------------
# 2. Compute statistics:
#    - Unique groups and counts for both the protected attribute and labels.
# -------------------------------------------------------------------
unique_groups, group_counts = np.unique(features, return_counts=True)
unique_labels, label_counts = np.unique(labels, return_counts=True)

print("Age Groups Distribution:")
for group, cnt in zip(unique_groups, group_counts):
    print("  Group {}: {}".format(group, cnt))

print("\nClass Labels Distribution:")
for lab, cnt in zip(unique_labels, label_counts):
    print("  Label {}: {}".format(lab, cnt))

# -------------------------------------------------------------------
# 3. Create Matplotlib visualizations (Histograms/Bar Charts)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Bar chart for Age Groups (Protected Attribute)
plt.subplot(1, 2, 1)
plt.bar(unique_groups, group_counts, color='skyblue', edgecolor='black')
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.title("Distribution of Age Groups")

# Bar chart for Class Labels
plt.subplot(1, 2, 2)
plt.bar(unique_labels, label_counts, color='salmon', edgecolor='black')
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.title("Distribution of Class Labels")

plt.tight_layout()
plt.savefig(path1 + "Pokec_stats_matplotlib.png")
plt.show()

# -------------------------------------------------------------------
# 4. Create interactive visualizations using pyecharts
#    These HTML files can be opened in any web browser.
# -------------------------------------------------------------------

# --- Interactive Bar Chart for Age Groups ---
bar_groups = (
    Bar()
    .add_xaxis([str(g) for g in unique_groups.tolist()])
    .add_yaxis("Count", group_counts.tolist())
    .set_global_opts(title_opts=opts.TitleOpts(title="Age Groups Distribution (Bar Chart)"))
)
bar_groups.render(path1 + "Pokec_groups_bar.html")

# --- Interactive Pie Chart for Age Groups ---
pie_groups = (
    Pie()
    .add(
        "",
        [list(z) for z in zip([str(g) for g in unique_groups.tolist()], group_counts.tolist())],
        radius=["30%", "55%"],
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="Age Groups Distribution (Pie Chart)"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
)
pie_groups.render(path1 + "Pokec_groups_pie.html")

# --- Interactive Bar Chart for Class Labels ---
bar_labels = (
    Bar()
    .add_xaxis([str(l) for l in unique_labels.tolist()])
    .add_yaxis("Count", label_counts.tolist())
    .set_global_opts(title_opts=opts.TitleOpts(title="Class Labels Distribution (Bar Chart)"))
)
bar_labels.render(path1 + "Pokec_labels_bar.html")

# --- Interactive Pie Chart for Class Labels ---
pie_labels = (
    Pie()
    .add(
        "",
        [list(z) for z in zip([str(l) for l in unique_labels.tolist()], label_counts.tolist())],
        radius=["30%", "55%"],
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="Class Labels Distribution (Pie Chart)"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
)
pie_labels.render(path1 + "Pokec_labels_pie.html")

print("Visualizations created:")
print(" - Matplotlib chart saved as Pokec_stats_matplotlib.png")
print(" - Interactive HTML files saved for both age groups and class labels.")