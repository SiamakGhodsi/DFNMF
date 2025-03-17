# experiments/grid_search_experiment.py

import numpy as np
import pandas as pd
import torch
import igraph as ig
import networkx as nx

# Data loader: function that returns (A, F, dataset_name)
from data.data_loader import *
# Evaluation functions:
from utils.evaluations import reflow_clusters, compute_group_balance, compute_individual_balance
# Fairness utility (to compute the fairness graph for individual balance)
from utils.utils import compute_F

# Import model functions (adjust these imports to your actual modules)
from models.spectral_clustering import normal_sc, group_fair_sc, scalable_fair_sc
from models.nmf import NMF
from models.fairnmf import FairNMF

# Placeholder for GNN method (if available)
# from models.gnn import GNN_clustering

class GridSearchExperiment:
    def __init__(self, config):
        """
        config: a dictionary containing parameters for the grid search.
            Required keys include:
              - dataset_id: int, which dataset to load.
              - attr: str, attribute name (e.g., "Gender" or "Country").
              - methods: list of method names to run (e.g., ["fair_nmf", "fsc", "nmf", "sc", "gnn"]).
              - k_values: list of cluster numbers (k) to test.
              - lambda_values: dict mapping method names (that require lambda) to lists of lambda values.
                                For methods that do not require lambda, you can omit them.
              - repeats: int, number of repetitions per parameter combination.
        """
        self.config = config
        self.dataset_id = config.get("dataset_id", 1)
        self.attr = config.get("attr", "Gender")
        self.methods = config.get("methods", ["fair_nmf", "fsc", "nmf", "sc"])
        self.k_values = config.get("k_values", list(range(2, 15)))
        self.lambda_values = config.get("lambda_values", {"fair_nmf": [1, 4, 10]})
        self.repeats = config.get("repeats", 10)

        # Map dataset IDs to (dataset_name, loader_function)
        dataset_mapping = {
            1: ("Diaries", load_diaries),
            2: ("Facebook", load_facebook),
            3: ("Friendship", load_friendship),
            4: ("DrugNET", load_drugnet),
            5: ("NBA", load_nba),
            6: ("LastFM", load_lfm),
            7: ("Pokec", load_pokec_n)
        }

        if self.dataset_id not in dataset_mapping:
            raise ValueError("Invalid dataset_id provided.")

        self.dataset_name, loader_fn = dataset_mapping[self.dataset_id]
        # Unpack three outputs: A, F, L.
        self.A, self.F, self.L = loader_fn()

        # Ensure A is a NumPy array.
        self.A_np = self.A.numpy() if torch.is_tensor(self.A) else self.A

        # Obtain groups from F (using your reflow_clusters function).
        groups_temp = reflow_clusters(self.F)
        self.groups = groups_temp.numpy() if torch.is_tensor(groups_temp) else np.array(groups_temp)

        # Build a NetworkX graph from A (if needed elsewhere).
        self.G1 = nx.from_numpy_array(self.A_np, create_using=nx.Graph)

    def run(self):
        # Define the desired output columns.
        columns = [
            'network', 'attr', 'method', 'lam', 'nodes (n)', 'clusters (k)', 'groups (h)',
            'modularity', 'avg_balance', 'cluster_balances', 'avg_rho', 'min_rho', 'ind_rhos',
            'avg_alpha', 'min_alpha', 'Cluster_alphas', 'clusters'
        ]
        results_df = pd.DataFrame(columns=columns)
        n = self.A_np.shape[0]
        groups_h = len(np.unique(self.groups))

        # Loop over each method.
        for method in self.methods:
            # Determine the lambda values to use.
            # For methods that do not need lambda, use [None].
            lam_list = self.lambda_values.get(method, [None])
            for k_val in self.k_values:
                for lam in lam_list:
                    # Initialize lists to accumulate metrics over repeats.
                    modularity_list = []
                    avg_balance_list = []
                    cluster_balances_list = []
                    avg_rho_list = []
                    min_rho_list = []
                    ind_rhos_list = []
                    avg_alpha_list = []
                    min_alpha_list = []
                    cluster_alphas_list = []
                    clusters_list = []

                    # Run the experiment for the given (k, lam) combination.
                    for rep in range(self.repeats):
                        # Run the selected model.
                        if method == "sc":
                            pred = normal_sc(self.A_np, k_val)
                        elif method == "fsc":
                            pred = group_fair_sc(self.A_np, self.F_real, k_val)
                        elif method == "sfsc":
                            pred = scalable_fair_sc(self.A_np, self.F_real, k_val)
                        elif method == "nmf":
                            pred = NMF(self.A_np, k_val, eps=1e-6, iter=500)
                        elif method == "fair_nmf":
                            # For FairNMF, pass the lambda value.
                            # Assumes FairNMF returns a tuple where the second element is the predicted clustering.
                            pred = FairNMF(self.A_np, k_val, self.groups, lam, eps=1e-6, iter=500)[1]
                        elif method == "gnn":
                            # Placeholder: call your GNN clustering function here.
                            # pred = GNN_clustering(self.A_np, k_val)
                            raise NotImplementedError("GNN method not implemented yet.")
                        else:
                            raise ValueError("Unknown method: {}".format(method))

                        # Reflow the clusters to contiguous labels.
                        pred = reflow_clusters(pred)
                        # Ensure predicted clusters are a NumPy array.
                        pred = pred.numpy() if torch.is_tensor(pred) else np.array(pred)

                        # Compute igraph modularity.
                        g_ig = ig.Graph.Adjacency((self.A_np > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
                        membership = pred.tolist()
                        ig_mod = g_ig.modularity(membership)

                        # Compute group balance (which we denote as α).
                        cluster_balances, avg_group_balance = compute_group_balance(pred, self.groups)
                        # Compute individual balance (denoted as ρ).
                        ind_balances, avg_ind_balance = compute_individual_balance(pred, self.F_real)
                        min_ind_balance = float(np.min(ind_balances))
                        min_cluster_balance = float(np.min(cluster_balances))

                        # Accumulate the metrics.
                        modularity_list.append(ig_mod)
                        avg_balance_list.append(avg_group_balance)
                        cluster_balances_list.append(cluster_balances)
                        avg_rho_list.append(avg_ind_balance)
                        min_rho_list.append(min_ind_balance)
                        ind_rhos_list.append(ind_balances.tolist())
                        avg_alpha_list.append(avg_group_balance)
                        min_alpha_list.append(min_cluster_balance)
                        cluster_alphas_list.append(cluster_balances)
                        clusters_list.append(pred.tolist())

                    # Aggregate metrics over repeats (using mean for scalars; for lists, store the first repeat’s result).
                    row = {
                        'network': self.dataset_name,
                        'attr': self.attr,
                        'method': method,
                        'lam': lam if lam is not None else np.nan,
                        'nodes (n)': n,
                        'clusters (k)': k_val,
                        'groups (h)': groups_h,
                        'modularity': np.mean(modularity_list),
                        'avg_balance': np.mean(avg_balance_list),
                        'cluster_balances': cluster_balances_list[0],
                        'avg_rho': np.mean(avg_rho_list),
                        'min_rho': np.mean(min_rho_list),
                        'ind_rhos': ind_rhos_list[0],
                        'avg_alpha': np.mean(avg_alpha_list),
                        'min_alpha': np.mean(min_alpha_list),
                        'Cluster_alphas': cluster_alphas_list[0],
                        'clusters': clusters_list[0]
                    }
                    results_df = results_df.append(row, ignore_index=True)
        return results_df


if __name__ == "__main__":
    # Example configuration for grid-search.
    config = {
        'dataset_id': 2,  # For example, 2 corresponds to Facebook.
        'attr': "Gender",  # The protected attribute.
        'methods': ["fair_nmf", "fsc", "nmf", "sc"],
        'k_values': list(range(2, 15)),
        'lambda_values': {"fair_nmf": [1, 4, 10]},  # Only for methods that require lambda.
        'repeats': 10
    }

    experiment = GridSearchExperiment(config)
    df_results = experiment.run()
    print(df_results)
    df_results.to_csv(experiment.dataset_name + '_gridsearch_results.csv', index=False)
