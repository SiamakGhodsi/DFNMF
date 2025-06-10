import torch
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from numpy.ma.core import shape

# Import data loaders and evaluation functions.
from data.data_loader import *
from utils.evaluations_optmzd import *  # This imports your clustering_accuracy, compute_balanced_accuracy, etc.
from utils.utils import *

# Import model functions.
from models.dfnmf import Namespace, DFNMF  # Your DFNMF model.
from models.competitors import normal_sc, group_fair_sc, scalable_fair_sc_deflation, ind_fair_sc, \
    ifnmtf  # Competitor methods.
from models.dmon import dmon_clustering
from data.sbm_generator import gen_kleindessner  # ensure this function is in your path

print("Using device:", device)

class Experiment:
    def __init__(self, config):
        """
        Initialize the experiment.

        config (dict) should include:
          - dataset_id: int (e.g., 5 for NBA, 7 for Pokec)
          - attr: str (e.g., "Team" or "Gender")
          - supervised: bool, True if ground truth cluster labels are available.
          - methods: list of method identifiers, e.g., ["deep_fair_nmf", "nmtf", "ifnmtf", "sc", "fsc", "sfsc", "ifsc", "gnn"]
          - true_k: int, the known number of clusters (since k is fixed)
          - lambda_range: list of lambda values to grid search over (for methods that require it)
          - repeats: number of repetitions for each parameter setting.
          - iter: number of iterations for each run.
          - params: dict of additional parameters (e.g., 'eps', 'base_layers', 'init_method', etc.)
        """
        self.config = config
        self.dataset_id = config.get('dataset_id', 8)
        self.attr = config.get('attr', "Team")
        self.supervised = config.get('supervised', False)
        self.methods = config.get('methods', ["deep_fair_nmf"])
        self.true_k = config.get('true_k', None)
        if self.true_k is None:
            raise ValueError("For NBA/Pokec experiments, 'true_k' must be provided.")
        self.lambda_range = config.get('lambda_range', [None])
        self.repeats = config.get('repeats', 1)
        self.iterations = config.get('iter', 500)
        self.params = config.get('params', {})

        self.num_nodes = config.get('num_nodes', 1000)
        self.true_k = config.get('true_k', 2)
        self.num_groups = config.get('num_groups', 2)
        self.probs = config.get('probs', [0.2, 0.15, 0.1, 0.05])

        # Map dataset IDs to (dataset_name, loader_function)
        dataset_mapping = {
            1: ("Diaries", load_diaries),
            2: ("Facebook", load_facebook),
            3: ("Friendship", load_friendship),
            4: ("DrugNET", load_drugnet),
            5: ("NBA", load_nba),
            6: ("LastFM", load_lfm),
            7: ("Pokec", load_pokec_n),
            8: ("SBM", gen_kleindessner(num_nodes=1000, num_clusters=5, num_groups=5, p=0.2, q=0.15, r=0.1, s=0.05))
        }
        if self.dataset_id not in dataset_mapping:
            raise ValueError("Invalid dataset_id provided.")
        self.dataset_name, loader_fn = dataset_mapping[self.dataset_id]

    def _run_method(self, method, lambda_val, num_nodes, base_layers=None):
        """
        Runs a single experiment for the given method using the constant true_k.
        Computes evaluation metrics.
        """

        adj_mat, fair_mat, clusters_gt, groups_gt = gen_kleindessner(num_nodes, self.true_k, self.num_groups,
                                                    p=self.probs[0], q=self.probs[1], r=self.probs[2], s=self.probs[3])
        # Use only adj_mat, and for fairness, we use groups.
        standard_groups = reflow_clusters(groups_gt)
        groups_sbm = standard_groups.numpy() if hasattr(standard_groups, "numpy") else np.array(standard_groups)
        F_ = compute_F(groups_sbm)
        A_sbm = torch.tensor(adj_mat, dtype=torch.float, device=device)
        F_sbm = torch.tensor(F_, dtype=torch.float, device=device)
        # For evaluation, ground truth clusters and groups
        gt_clusters = clusters_gt
        gt_groups = groups_gt

        self.A, self.F, self.L = A_sbm, F_sbm, gt_clusters
        self.A_np = adj_mat if hasattr(adj_mat, "numpy") else adj_mat
        self.F_np = F_sbm.numpy() if hasattr(F_sbm, "numpy") else F_sbm

        # Build igraph graph.
        self.G_ig = ig.Graph.Adjacency((self.A_np > 0).tolist(), mode=ig.ADJ_UNDIRECTED)

        # Assume groups come from reflow_clusters on F.
        standard_groups = reflow_clusters(self.F_np)
        self.groups = standard_groups.numpy() if hasattr(standard_groups, "numpy") else np.array(standard_groups)
        self.F_real = compute_F(self.groups)

        # If supervised, ground truth cluster labels are in L.
        if self.supervised:
            self.gt_clusters = self.L.numpy() if hasattr(self.L, "numpy") else np.array(self.L).flatten()

        k = self.true_k
        loss = np.nan
        W = np.nan
        layers_used = np.nan

        if method == "nmtf":
            clusters, W, loss = ifnmtf(self.A_np, k, lam=0, groups=None, iter=self.iterations, eps=self.params['eps'])
        elif method == "ifnmtf":
            clusters, W, loss = ifnmtf(self.A_np, k, lam=lambda_val, groups=self.groups, iter=self.iterations,
                                       eps=self.params['eps'])
        elif method == "deep_fair_nmf":
            if lambda_val is None:
                raise ValueError("deep_fair_nmf requires a lambda value.")
            if base_layers is not None:
                layers_used = base_layers + [k]
            elif 'base_layers' in self.params:
                layers_used = self.params['base_layers'] + [k]
            else:
                layers_used = [k]
            args_obj = Namespace(
                type=self.params.get('type', torch.float),
                ft_itr=self.iterations,
                pr_itr=self.params.get('pr_itr', 50),
                layers=layers_used,
                lg=lambda_val,
                sparse=self.params.get('sparse', False),
                init_method=self.params.get('init_method', 'svd')
            )
            model = DFNMF(torch.tensor(self.A_np, dtype=torch.float, device=device),
                          torch.tensor(self.F_real, dtype=torch.float, device=device),
                          args_obj)
            Hs, Ws, loss = model.training()
            W = model.Ws[-1]
            clusters = reflow_clusters(torch.argmax(model.P, dim=1))
        elif method == "sc":
            clusters = normal_sc(self.A_np, k,
                                 normalize_laplacian=self.params.get('normalize_laplacian', False),
                                 normalize_evec=self.params.get('normalize_evec', False))
            clusters = reflow_clusters(clusters)
        elif method == "fsc":
            clusters = group_fair_sc(self.A_np, self.F_real, k,
                                     normalize_laplacian=self.params.get('normalize_laplacian', False),
                                     normalize_evec=self.params.get('normalize_evec', False))
            clusters = reflow_clusters(clusters)
        elif method == "ifsc":
            clusters = ind_fair_sc(self.A_np, self.groups, k,
                                   normalize_laplacian=self.params.get('normalize_laplacian', False),
                                   normalize_evec=self.params.get('normalize_evec', False))
            clusters = reflow_clusters(clusters)
        elif method == "sfsc":
            clusters = scalable_fair_sc_deflation(self.A_np, self.F_real, k,
                                                  tol_eig=self.params.get('tol_eig', 1e-8))
            clusters = reflow_clusters(clusters)
        elif method == "dmon":
            # Run the DMoN clustering competitor.
            # Here, we assume that F_np (features) and A_np (adjacency) are provided,
            # Call the imported function
            clusters = dmon_clustering(
                adjacency=self.A_np,  # Pass the adjacency matrix
                features=self.F_np,  # Pass the feature matrix
                n_clusters=k,  # Pass the desired number of clusters
                n_epochs=200,  # Set desired epochs (or use a variable)
                # --- Optional: Specify other parameters if needed ---
                architecture=[64],  # Default from the function
                dropout_rate=0,  # Default from the function
                collapse_regularization=1.0,  # Default from the function
                learning_rate=0.001  # Default from the function
            )  # Reflow clusters (i.e. map them to contiguous labels).
            clusters = reflow_clusters(clusters)
        else:
            raise ValueError("Unsupported method: {}".format(method))

        print("Run complete: ", method)

        # unsupervised clustering metrics
        membership = clusters.numpy().tolist() if hasattr(clusters, "numpy") else list(clusters)
        mod_ig = self.G_ig.modularity(membership)

        # Compute fairness metrics.
        delta_stat_par = delta_statistical_parity(
            clusters.numpy() if hasattr(clusters, "numpy") else clusters, self.groups)
        group_balances, avg_balance = compute_group_balance(
            clusters.numpy() if hasattr(clusters, "numpy") else clusters, self.groups)
        ind_balances, avg_balance_ind = compute_individual_balance(
            clusters.numpy() if hasattr(clusters, "numpy") else clusters, self.groups)

        # Supervised clustering metrics.
        if self.supervised:
            gt = self.gt_clusters
            clust_acc = clustering_accuracy(gt, clusters.numpy() if hasattr(clusters, "numpy") else clusters)
            bal_acc = compute_balanced_accuracy(gt, clusters.numpy() if hasattr(clusters, "numpy") else clusters)
            macro_f1 = compute_macro_f1(gt, clusters.numpy() if hasattr(clusters, "numpy") else clusters)
            nmi = normalized_mutual_info_score(gt, clusters.numpy() if hasattr(clusters, "numpy") else clusters)
            ari = adjusted_rand_score(gt, clusters.numpy() if hasattr(clusters, "numpy") else clusters)
            delta_eq_odds = delta_equalized_odds(
                clusters.numpy() if hasattr(clusters, "numpy") else clusters,
                self.gt_clusters,
                self.groups,
                favorable=0  # or choose an appropriate favorable outcome
            )
        else:
            clust_acc = bal_acc = macro_f1 = nmi = ari = delta_eq_odds = np.nan

        result = {
            'network': self.dataset_name,
            'attr': self.attr,
            'method': method,
            'lambda': lambda_val if lambda_val is not None else "",
            'nodes (n)': self.A_np.shape[0],
            'clusters (k)': k,
            'groups (h)': len(np.unique(self.groups)),
            'igraph_modularity': mod_ig,
            'average balance': avg_balance,
            'cluster balances': group_balances,
            'average individual balance': avg_balance_ind,
            'individual balances': ind_balances,
            'interaction matrix': W,
            'loss': loss,
            'layers': layers_used,
            'clustering_accuracy': clust_acc,
            'balanced_accuracy': bal_acc,
            'macro_f1': macro_f1,
            'NMI': nmi,
            'ARI': ari,
            'delta_stat_par': delta_stat_par,
            'delta_eq_odds': delta_eq_odds
        }

        return result

    def run_grid_search(self):
        """
        For NBA/Pokec, since k is fixed, grid search is over lambda and network architecture (base_layers).
        Returns a DataFrame with one row per parameter setting.
        """
        arch_options = self.params.get('base_layers', [[50]])
        rows = []
        for nodes in self.num_nodes:
            for lambda_val in self.lambda_range:
                for base_layers in arch_options:
                    result = self._run_method(self.methods[0], lambda_val, nodes, base_layers)
                    row = {
                        'network': self.dataset_name,
                        'attr': self.attr,
                        'method': self.methods[0],
                        'lambda': lambda_val,
                        'nodes (n)': self.num_nodes,
                        'clusters (k)': self.true_k,
                        'groups (h)': self.num_groups,
                        'igraph_modularity': result['igraph_modularity'],
                        'average balance': result['average balance'],
                        'cluster balances': result['cluster balances'],
                        'average individual balance': result['average individual balance'],
                        'individual balances': result['individual balances'],
                        'interaction matrix': result['interaction matrix'],
                        'loss': result['loss'],
                        'layers': result['layers'],
                        'clustering_accuracy': result['clustering_accuracy'],
                        'balanced_accuracy': result['balanced_accuracy'],
                        'macro_f1': result['macro_f1'],
                        'NMI': result['NMI'],
                        'ARI': result['ARI'],
                        'delta_stat_par': result['delta_stat_par'],
                        'delta_eq_odds': result['delta_eq_odds']
                    }
                    rows.append(row)
        df = pd.DataFrame(rows, columns=['network', 'attr', 'method', 'lambda', 'nodes (n)', 'clusters (k)',
                                         'groups (h)', 'igraph_modularity', 'average balance', 'cluster balances',
                                         'average individual balance', 'individual balances', 'interaction matrix',
                                         'loss', 'layers', 'clustering_accuracy', 'balanced_accuracy', 'macro_f1',
                                         'NMI', 'ARI', 'delta_eq_odds', 'delta_stat_par'])
        print(df)
        return df

    def run_real_comparison(self):
        """
        Runs a comparison experiment across methods (for a fixed number of clusters).
        Returns a DataFrame with one row per method.
        """
        rows = []
        for nodes in self.num_nodes:
            for method in self.methods:
                lambda_val = self.params.get('lam', None) if method in ["deep_fair_nmf", "nmtf", "ifnmtf"] else None
                result = self._run_method(method, lambda_val, nodes, base_layers=None)
                row = {
                    'network': self.dataset_name,
                    'attr': self.attr,
                    'method': self.methods[0],
                    'lambda': lambda_val,
                    'nodes (n)': self.num_nodes,
                    'clusters (k)': self.true_k,
                    'groups (h)': self.num_groups,
                    'igraph_modularity': result['igraph_modularity'],
                    'average balance': result['average balance'],
                    'cluster balances': result['cluster balances'],
                    'average individual balance': result['average individual balance'],
                    'individual balances': result['individual balances'],
                    'interaction matrix': result['interaction matrix'],
                    'loss': result['loss'],
                    'layers': result['layers'],
                    'clustering_accuracy': result['clustering_accuracy'],
                    'balanced_accuracy': result['balanced_accuracy'],
                    'macro_f1': result['macro_f1'],
                    'NMI': result['NMI'],
                    'ARI': result['ARI'],
                    'delta_stat_par': result['delta_stat_par'],
                    'delta_eq_odds': result['delta_eq_odds']
                }
                rows.append(row)

        df = pd.DataFrame(rows, columns=['network', 'attr', 'method', 'lam', 'nodes (n)', 'clusters (k)',
                                         'groups (h)', 'igraph_modularity', 'average balance', 'cluster balances',
                                         'average individual balance','individual node balances', 'interaction matrix',
                                         'loss', 'layers', 'clustering_accuracy', 'balanced_accuracy',
                                         'macro_f1', 'NMI', 'ARI', 'delta_stat_par', 'delta_eq_odds'])
        return df


if __name__ == "__main__":
    # ====================
    # For supervised analysis on SBM dataset:
    config_sbm = {
        'dataset_id': 8,  # SBM
        'attr': "Team",
        'num_nodes': [10000, 5000, 2000],
        'num_groups': 10,
        'probs': [0.2, 0.15, 0.1, 0.05],
        'supervised': True,
        'methods': ["nmtf", "ifnmtf", "sc", "fsc", "sfsc", "ifsc"],
        'true_k': 5,  # True number of clusters for NBA.
        'lambda_range': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100],
        'repeats': 1,
        'iter': 500,
        'params': {
            'eps': 1e-6,
            'lam': 1,  # Default lambda for methods that require it.
            'normalize_laplacian': False,
            'normalize_evec': False,
            'tol_eig': 1e-8,
            'base_layers': [[128,64,16], [50, 10], [40, 40], [50], [20]],  # List of architecture options.
            'init_method': 'svd',
            'sparse': True,
            'type': torch.float,
        }
    }

    exp_sbm = Experiment(config_sbm)
    df_sbm = exp_sbm.run_real_comparison()
    print("SBM Real Comparison Results:")
    print(df_sbm)
    df_sbm.to_csv(exp_sbm.dataset_name + '_comparison.csv', index=False)

    # ====================
    # For grid search on NBA dataset (supervised grid search over lambda and base_layers):
    config_grid_sbm = {
        'dataset_id': 8,  # SBM
        'attr': "Team",
        'num_nodes': [10000, 5000, 2000],
        'num_groups': 10,
        'probs': [0.2, 0.15, 0.1, 0.05],
        'supervised': True,
        'methods': ["deep_fair_nmf"],
        'true_k': 5,
        'lambda_range': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100],
        'repeats': 1,
        'iter': 500,
        'params': {
            'eps': 1e-6,
            'lam': 1,
            'normalize_laplacian': False,
            'normalize_evec': False,
            'tol_eig': 1e-8,
            'base_layers': [[128,64,16], [50, 10], [40, 40], [50], [20]],
            'init_method': 'svd',
            'sparse': True,
            'type': torch.float,
        }
    }

    exp_grid_sbm = Experiment(config_grid_sbm)
    df_grid_sbm = exp_grid_sbm.run_grid_search()
    print("SBM Grid Search Results:")
    print(df_grid_sbm)
    df_grid_sbm.to_csv(exp_grid_sbm.dataset_name + '_k_lam_gridsearch.csv', index=False)

    # 'num_nodes': [10000, 5000, 2000],
    # 'num_groups': 10,
    # true_k': 5,