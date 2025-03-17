import torch
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig

# Import data loader and evaluation functions.
from data.data_loader import *
from utils.evaluations_optmzd import *
from utils.utils import *

# Import model functions from your models modules.
from models.dfnmf import Namespace                                           # Vanilla NMTF
from models.competitors import normal_sc, group_fair_sc                      # SC and FairSC
from models.competitors import scalable_fair_sc, scalable_fair_sc_deflation  # sFairSC variants
from models.competitors import ind_fair_sc, ifnmtf                         # Individual (Fair SC, FairNMTF)

from models.dfnmf import DFNMF  # Fair DeepNMF, and also shallow NMTF

# You might also have a placeholder for a GNN method.
def gnn_placeholder(adj_mat, num_clusters, **kwargs):
    # Placeholder implementation; replace with your actual GNN code.
    # For now, we simply run k-means on a random embedding.
    n = adj_mat.shape[0]
    embedding = np.random.rand(n, num_clusters)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=500, random_state=42)
    return kmeans.fit_predict(embedding)

class Experiment:
    def __init__(self, config):
        """
        Initialize the experiment.

        config (dict) should include:
          - dataset_id: int (e.g., 1 for Diaries, 2 for Facebook, etc.)
          - attr: str (e.g., "Country")
          - grid_search: bool indicating if a grid search (over clusters and lambda) is performed.
          - methods: list of method identifiers, e.g. ["deep_fair_nmf", "nmtf", "ifnmtf" "sc",
                                                        "fsc", "ifsc", "sfsc", "gnn"]
          - cluster_range: list or range of cluster numbers to test.
          - lambda_range: list of lambda values to test (for methods that require it); otherwise, can be [None]
          - repeats: number of repetitions for each parameter setting.
          - iter: number of iterations to run in each method (if applicable).
          - params: additional parameters in a sub-dictionary under "params".
        """
        self.config = config
        self.dataset_id = config.get('dataset_id', 1)
        self.attr = config.get('attr', "Country")
        self.grid_search = config.get('grid_search', True)
        self.methods = config.get('methods', ["deep_fair_nmf"])
        self.cluster_range = config.get('cluster_range', list(range(2, 15)))
        self.lambda_range = config.get('lambda_range', [None])  # Use [None] if lambda not applicable.
        self.repeats = config.get('repeats', 10)
        self.iterations = config.get('iter', 500)
        self.params = config.get('params', {})  # Additional parameters (e.g., eps, tol_eig, etc.)

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

        # Convert to numpy arrays if needed.
        self.A_np = self.A.numpy() if hasattr(self.A, "numpy") else self.A
        self.F_np = self.F.numpy() if hasattr(self.F, "numpy") else self.F

        # Prepare the NetworkX and igraph graphs.
        #self.G_nx = nx.from_numpy_array(self.A_np, create_using=nx.Graph, parallel_edges=False)
        self.G_ig = ig.Graph.Adjacency((self.A_np > 0).tolist(), mode=ig.ADJ_UNDIRECTED)

        # Assume groups come from reflowing F (or from some function you already have)
        standard_groups = reflow_clusters(self.F_np)
        self.groups = np.transpose(standard_groups.numpy() if hasattr(standard_groups, "numpy")
                                   else standard_groups)
        # For compatibility, also compute F_real if needed for some methods.
        self.F_real = compute_F(self.groups)

    def _run_method(self, method, num_clusters, lambda_val):
        """
        Runs a single experiment for the given method, cluster number and lambda.
        Returns a dictionary with the computed metrics.
        """
        # values of loss, W, and layers should be initiated to "N/A" for methods that don't use them.
        loss = np.nan
        W = np.nan
        layers = np.nan

        # Depending on the method, call the corresponding model function.
        # For methods that require lam (e.g., deep_fair_nmf), lam will be used; otherwise, ignore.
        if method == "nmtf":
            # Vanilla NMTF: expects A, num_clusters, lam, groups, eps, iter. lam should be =0 and groups =Null
            clusters, W, loss = ifnmtf(self.A_np, num_clusters, lam=0, groups=self.groups,
                                       iter=self.iterations, eps=self.params['eps'])

        elif method == "ifnmtf":
            # iFairNMTF: expects A, num_clusters, lam != 0, groups= self,groups, eps, iter.
            clusters, W, loss = ifnmtf(self.A_np, num_clusters, lam=lambda_val, groups=self.groups,
                                       iter=self.iterations, eps=self.params['eps'])

        elif method == "deep_fair_nmf":
            # Deep Fair NMF using DFNMF:
            if lambda_val is None:
                raise ValueError("deep_fair_nmf requires a lambda value.")

            # Determine the full layers list. If 'base_layers' exists, use it and append num_clusters.
            # For example, if base_layers = [40, 40] and k (num_clusters) = 10, then layers becomes [40, 40, 10].
            if 'base_layers' in self.params:
                layers = self.params['base_layers'] + [num_clusters]  # e.g., [40, 40] + [k]
            else:
                layers = [40, 40, num_clusters]  # Default: last layer is set to num_clusters.

            # Create a Namespace object with the required DFNMF parameters.
            args_obj = Namespace(
                type=self.params.get('type', torch.float),
                ft_itr=self.iterations,
                pr_itr=self.params.get('pr_itr', 50),
                layers= layers,          #self.params.get('layers', [40, 40, 10]),
                lg=lambda_val
                #li=self.params.get('lg', 1)
            )
            # Instantiate DFNMF. Note: self.A and self.F are torch tensors.
            model = DFNMF(torch.tensor(self.A_np, dtype=torch.float), torch.tensor(self.F_real, dtype=torch.float), args_obj)
            Hs, Ws, loss = model.training()
            W = model.Ws[-1]
            clusters = reflow_clusters(torch.argmax(model.P, dim=1))
            # Psi is the final deep representation matrix (deep membership)   # P = H₁ @ H₂ @ … @ Hₚ.

        elif method == "sc":
            clusters = normal_sc(self.A_np, num_clusters,
                                 normalize_laplacian=self.params.get('normalize_laplacian', False),
                                 normalize_evec=self.params.get('normalize_evec', False))
            clusters = reflow_clusters(clusters)
        elif method == "fsc":
            clusters = group_fair_sc(self.A_np, self.F_real, num_clusters,
                                     normalize_laplacian=self.params.get('normalize_laplacian', False),
                                     normalize_evec=self.params.get('normalize_evec', False))
            clusters = reflow_clusters(clusters)
        elif method == "ifsc":
            clusters = ind_fair_sc(self.A_np, self.groups, num_clusters,
                                     normalize_laplacian=self.params.get('normalize_laplacian', False),
                                     normalize_evec=self.params.get('normalize_evec', False))
            clusters = reflow_clusters(clusters)
        elif method == "sfsc":
            clusters = scalable_fair_sc_deflation(self.A_np, self.F_real, num_clusters,
                                                    tol_eig=self.params.get('tol_eig', 1e-8))
            clusters = reflow_clusters(clusters)
        elif method == "gnn":
            clusters = gnn_placeholder(self.A_np, num_clusters)
            clusters = reflow_clusters(clusters)
        else:
            raise ValueError("Unsupported method: {}".format(method))
        
        # Convert clusters to a list for igraph if necessary.
        membership = clusters.numpy().tolist() if hasattr(clusters, "numpy") else list(clusters)

        # Compute igraph modularity.
        mod_ig = self.G_ig.modularity(membership)

        # Compute group balance.
        balances, avg_balance = compute_group_balance(clusters.numpy()
                                                      if hasattr(clusters, "numpy") else clusters, self.groups)
        # calculate individual balance
        balances_ind, avg_balance_ind = compute_individual_balance(clusters.numpy()
                                        if hasattr(clusters, "numpy") else clusters, self.groups)
        result = {
            'network': self.dataset_name,
            'attr': self.attr,
            'method': method,
            'lam': lambda_val if lambda_val is not None else "",
            'nodes (n)': self.A_np.shape[0],
            'clusters (k)': num_clusters,
            'groups (h)': len(np.unique(self.groups)),
            'igraph_modularity': mod_ig,
            'average balance': avg_balance,
            'cluster balances': balances,
            'average individual balance': avg_balance_ind,
            'individual balances': balances_ind,
            'interaction matrix': W,
            'loss': loss,
            'layers': layers  # always record the layers configuration (or "N/A")
        }
        return result

    def run_grid_search(self):
        """
        Runs a grid search over cluster numbers and lambda values.
        This is typically used for methods that require lambda (e.g., fair_nmf).
        Returns a DataFrame with one row per parameter setting.
        """
        rows = []
        for num_clusters in self.cluster_range:
            for lambda_val in self.lambda_range:
                ig_mod_list = []
                bal_list = []
                ind_bal_list = []
                for rep in range(self.repeats):
                    result = self._run_method(self.methods[0], num_clusters, lambda_val)
                    ig_mod_list.append(result['igraph_modularity'])
                    bal_list.append(result['average balance'])
                    ind_bal_list.append(result['average individual balance'])
                # Average over repeats.
                avg_ig_mod = np.mean(ig_mod_list)
                avg_bal = np.mean(bal_list)
                avg_ind_bal = np.mean(ind_bal_list)
                # Use the last computed cluster balances.
                row = {
                    'network': self.dataset_name,
                    'attr': self.attr,
                    'method': self.methods[0],
                    'lam': lambda_val,
                    'nodes (n)': self.A_np.shape[0],
                    'clusters (k)': num_clusters,
                    'groups (h)': len(np.unique(self.groups)),
                    'igraph_modularity': avg_ig_mod,
                    'average balance': avg_bal,
                    'cluster balances': result['cluster balances'],
                    'average individual balance': avg_ind_bal,
                    'individual node balances': result['individual balances'],
                    'interaction matrix': result['interaction matrix'],
                    'loss': result['loss'],
                    'layers': result['layers']
                }
                rows.append(row)
        df = pd.DataFrame(rows, columns=['network', 'attr', 'method', 'lam', 'nodes (n)', 'clusters (k)',
                                         'groups (h)', 'igraph_modularity', 'average balance',
                                         'cluster balances', 'average individual balance',
                                         'individual node balances', 'interaction matrix', 'loss', 'layers'])
        print(df)
        return df

    def run_real_comparison(self):
        """
        Runs a real comparison experiment where for each cluster number,
        multiple methods are run (without lambda for those that do not require it).
        It expects self.methods to be a list of methods to compare (e.g., 
        ["sc", "fsc", "ifsc", "sfsc", "nmtf", "ifnmtf"]).
        Returns a DataFrame with one row per method for each cluster number.
        """
        rows = []
        for num_clusters in self.cluster_range:
            # Prepare lists for each method.
            rep_results = {method: [] for method in self.methods}
            for rep in range(self.repeats):
                for method in self.methods:
                    # For methods that require lambda, assume lambda is provided in params.
                    lambda_val = self.params.get('lam', None) if method in ["deep_fair_nmf","nmtf","ifnmtf"] else None
                    result = self._run_method(method, num_clusters, lambda_val)
                    rep_results[method].append(result)
            # Aggregate results for this cluster number.
            for method in self.methods:
                igmods = [r['igraph_modularity'] for r in rep_results[method]]
                bals = [r['average balance'] for r in rep_results[method]]
                ind_bals = [r['average individual balance'] for r in rep_results[method]]
                # Use the last repetition's cluster balances.
                avg_ig_mod = np.mean(igmods)
                avg_bal = np.mean(bals)
                avg_ind_bal = np.mean(ind_bals)
                row = {
                    'network': self.dataset_name,
                    'attr': self.attr,
                    'method': method,
                    'lam': "" if method not in ["deep_fair_nmf","ifnmtf","nmtf"] else self.params.get('lam', ""),
                    'nodes (n)': self.A_np.shape[0],
                    'clusters (k)': num_clusters,
                    'groups (h)': len(np.unique(self.groups)),
                    'igraph_modularity': avg_ig_mod,
                    'average balance': avg_bal,
                    'cluster balances': result['cluster balances'],
                    'average individual balance': avg_ind_bal,
                    'individual node balances': result['individual balances'],
                    'interaction matrix': result['interaction matrix'],
                    'loss': result['loss'],
                    'layers': result['layers']
                }
                rows.append(row)

        df = pd.DataFrame(rows, columns=['network', 'attr', 'method', 'lam', 'nodes (n)', 'clusters (k)',
                                         'groups (h)', 'igraph_modularity', 'average balance',
                                         'cluster balances', 'average individual balance',
                                         'individual node balances', 'interaction matrix', 'loss', 'layers'])
        return df

# For testing, one might include the following block.
if __name__ == "__main__":

    """
    # Example configuration for real comparison:
    config_real = {
        'dataset_id': 2,  # Facebook
        'attr': "Country",
        'grid_search': False,
        'methods': ["sc", "fsc", "ifsc", "sfsc", "nmtf", "ifnmtf"],  # methods to compare
        'cluster_range': list(range(2, 4)),
        'lambda_range': [None],  # not used here
        'repeats': 1,
        'iter': 500,
        'params': {
            'eps': 1e-6,
            'lam': 1,  # Only used for ifnmtf and for nmtf is overriden to be 0.
            'normalize_laplacian': False,
            'normalize_evec': False,
            'tol_eig': 1e-8,
            'base_layers': [40, 40]  # the last layer will be set to k automatically.
        }
    }
    
    exp_real = Experiment(config_real)
    df_real = exp_real.run_real_comparison()
    print("Real Comparison Results:")
    print(df_real)
    df_real.to_csv(exp_real.dataset_name + '_comparison.csv', index=False)
    """
    # Example configuration for grid search (for deep_fair_nmf):
    config_grid = {
        'dataset_id': 2,  # For example, Facebook
        'attr': "Country",
        'grid_search': True,
        'methods': ["deep_fair_nmf"],
        'cluster_range': list(range(2, 5)),
        'lambda_range': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000],
        'repeats': 1,
        'iter': 500,
        'params': {
            'eps': 1e-6,
            'lam': 1,
            # Default lambda value for real comparisons only. It is overridden by lambda_range for deep_fair_nmf in params.
            'normalize_laplacian': False,
            'normalize_evec': False,
            'base_layers': [40, 40]  # the last layer will be set to k automatically.
        }
    }

    exp_grid = Experiment(config_grid)
    df_grid = exp_grid.run_grid_search()
    print("Grid Search Results:")
    print(df_grid)
    df_grid.to_csv(exp_grid.dataset_name + '_k_lam_gridsearch.csv', index=False)
