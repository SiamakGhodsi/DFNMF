import scipy.io as sio
import numpy as np
import copy
import pandas as pd
from scipy.sparse import load_npz
import os

def load_diaries():
    path1 = "data/School/"
    diaries_file = os.path.join(path1, "diaries.csv")
    attr_file = os.path.join(path1, "cd_attr.csv")

    # Check if the processed files exist.
    if os.path.exists(diaries_file) and os.path.exists(attr_file):
        # If both files exist, load them directly.
        print("Processed files found. Loading diaries.csv and cd_attr.csv ...")
        A = pd.read_csv(diaries_file, header=None).to_numpy()
        F = pd.read_csv(attr_file, header=None).to_numpy()
    else:
        # If the processed files do not exist, reconstruct from raw files.
        print("Processed files not found. Reconstructing the data ...")
        # Load the processed graph from the raw diaries CSV (using np.loadtxt).
        A = np.loadtxt(os.path.join(path1, 'diaries.csv'), delimiter=',', dtype=np.int32)
        # Read metadata, drop column 1, and replace gender strings with numbers.
        df = pd.read_csv(os.path.join(path1, 'metadata_2013.txt'), delimiter='\t', header=None)
        df = df.drop(columns=[1])
        df = df.replace(['F', 'M', 'Unknown'], [1, 2, 0])
        df = df.set_index(0)
        # Read the original diaries network to extract node list.
        E = np.genfromtxt(os.path.join(path1, 'Contact-diaries-network_data_2013.csv'),
                           delimiter=' ', dtype=np.int32)
        # Assume E has at least two columns (node1, node2).
        N = np.unique(E[:, :-1])
        n = N.shape[0]
        # Reconstruct feature vector F based on metadata.
        F = np.zeros(n, dtype=np.int32)
        for i in range(E.shape[0]):
            l1 = int(np.where(N == E[i, 0])[0])
            l2 = int(np.where(N == E[i, 1])[0])
            F[l1] = df.loc[E[i, 0]].values[0]
            F[l2] = df.loc[E[i, 1]].values[0]

    # Create a dummy label vector L (all ones).
    L = np.ones(len(F), dtype=np.int32)

    uniqe_vals, count = np.unique(F, return_counts=True)
    Diaries_balance = min(count) / max(count)

    print("Shape of F matrix and A matrix: ", np.shape(F), np.shape(A))
    print("Dataset balance = ", Diaries_balance)

    return A, F, L

def load_friendship():
    path1 = "data/School/"
    diaries_file = os.path.join(path1, "friendship.csv")
    attr_file = os.path.join(path1, "fr_attr.csv")

    # Check if the processed files exist.
    if os.path.exists(diaries_file) and os.path.exists(attr_file):
        # If both files exist, load them directly.
        print("Processed files found. Loading diaries.csv and cd_attr.csv ...")
        A = pd.read_csv(diaries_file, header=None).to_numpy()
        F = pd.read_csv(attr_file, header=None).to_numpy()
    else:
        print("Processed Friendship files not found. Reconstructing from raw files ...")
        # Old reconstruction code.
        A = np.loadtxt(path1 + 'Friendship.csv', delimiter=',', dtype=np.int32)

        # Re-read metadata to get node features.
        df = pd.read_csv(path1 + 'metadata_2013.txt', delimiter='\t', header=None)
        df = df.drop(columns=[1])
        df = df.replace(['F', 'M', 'Unknown'], [1, 2, 0])
        df = df.set_index(0)

        # Load the original Friendship network data.
        E = np.genfromtxt(path1 + 'Friendship-network_data_2013.csv', delimiter=' ', dtype=np.int32)
        N = np.unique(E)  # Here, the network file uses one column per node.
        n = N.shape[0]

        # Reconstruct feature vector F.
        F = np.zeros(n, dtype=np.int32)
        for i in range(E.shape[0]):
            l1 = int(np.where(N == E[i, 0])[0])
            l2 = int(np.where(N == E[i, 1])[0])
            F[l1] = df.loc[E[i, 0]].values[0]
            F[l2] = df.loc[E[i, 1]].values[0]

    # Create a dummy label vector L (all ones).
    L = np.ones(len(F), dtype=np.int32)

    uniqe_vals, count = np.unique(F, return_counts=True)
    Friendship_balance = min(count) / max(count)

    print("Shape of F matrix and A matrix: ", np.shape(F), np.shape(A))
    print("Dataset balance = ", Friendship_balance)

    return A, F, L

def load_facebook():
    path1 = "data/School/"
    graph_file = os.path.join(path1, "facebook.csv")
    attr_file = os.path.join(path1, "fb_attr.csv")

    if os.path.exists(graph_file) and os.path.exists(attr_file):
        print("Processed Facebook files found. Loading facebook.csv and fb_attr.csv ...")
        A = pd.read_csv(graph_file, header=None).to_numpy()
        F = pd.read_csv(attr_file, header=None).to_numpy()
    else:
        print("Processed Facebook files not found. Reconstructing from raw files ...")
        # Old reconstruction code.
        A = np.loadtxt(path1 + 'facebook.csv', delimiter=',', dtype=np.int32)

        # Re-read metadata to obtain features.
        df = pd.read_csv(path1 + 'metadata_2013.txt', delimiter='\t', header=None)
        df = df.drop(columns=[1])
        df = df.replace(['F', 'M', 'Unknown'], [1, 2, 0])
        df = df.set_index(0)

        # Load the original Facebook network data.
        E = np.genfromtxt(path1 + 'Facebook-known-pairs_data_2013.csv', delimiter=' ', dtype=np.int32)
        # Use the first two columns (ignoring the third column which is a weight).
        N = np.unique(E[:, :-1])
        n = N.shape[0]

        # Reconstruct the feature vector F.
        F = np.zeros(n, dtype=np.int32)
        for i in range(E.shape[0]):
            l1 = int(np.where(N == E[i, 0])[0])
            l2 = int(np.where(N == E[i, 1])[0])
            F[l1] = df.loc[E[i, 0]].values[0]
            F[l2] = df.loc[E[i, 1]].values[0]

    # Create a dummy label vector L (all ones).
    L = np.ones(len(F), dtype=np.int32)

    uniqe_vals, count = np.unique(F, return_counts=True)
    Facebook_balance = min(count) / max(count)

    print("Shape of F matrix and A matrix: ", np.shape(F), np.shape(A))
    print("Dataset balance = ", Facebook_balance)

    return A, F, L

def load_drugnet():
    path1 = "DrugNet/CSV/"
    graph_file = os.path.join(path1, "DrugNetgraph.csv")
    feature_file = os.path.join(path1, "DrugNetfeature.csv")

    if os.path.exists(graph_file) and os.path.exists(feature_file):
        print("Processed DrugNET files found. Loading DrugNetgraph.csv and DrugNetfeature.csv ...")
        A = pd.read_csv(graph_file, header=None).to_numpy()
        F = pd.read_csv(feature_file, header=None).to_numpy()
        # Optionally, extract the ethnicity feature if needed:
        F_ethn = F[:, 0]
    else:
        print("Processed DrugNET files not found. Reconstructing from raw files ...")
        A0 = np.genfromtxt(path1 + 'DRUGNET.csv', delimiter=',')[1:, 1:]
        A0 = np.maximum(A0, A0.T)  # transform to undirected net
        F0 = np.genfromtxt(path1 + 'DRUGATTR.csv', delimiter=',').astype(np.int64)[1:, 1:]

        s = np.sum(A0, axis=1) + np.sum(A0, axis=0)
        nze = np.where(s != 0)[0]

        A = A0[nze, :]
        A = A[:, nze]

        # identify unlinked nodes
        sm = np.array([105, 151, 51, 135, 145, 147, 35, 176, 181, 158, 166, 114, 117, 11, 73, 98, 120, 126, 192])
        nn = A.shape[0]
        F = F0[nze, :]
        inter = np.setdiff1d(np.arange(nn), sm)

        # exclude unlinked nodes out of network
        A = A[inter, :]
        A = A[:, inter]
        F = F[inter, :]

        # according to the ICML paper, ethnicity is categorized to three groups. We keep groups 2,3 and put 1,5,6,7 to another
        # category according to: https://sites.google.com/site/ucinetsoftware/datasets/covert-networks/drugnet
        F_new = copy.deepcopy(F)
        F_new[(F[:, 0] < 2) | (F[:, 0] > 3), 0] = 1

    all_in_one = np.ones(F_new.shape[0])
    uniqe_vals, count = np.unique(F_new[:, 0], return_counts=True)
    DrugNet_balance_ethnicity = min(count) / max(count)

    print("Shape of F matrix and A matrix: ", np.shape(F_ethn), np.shape(A))
    print("Dataset balance = ", DrugNet_balance_ethnicity)
    L = np.ones(F_ethn, dtype=np.int32)
    return A, F_ethn, L

def load_lfm():
    path = "LastFM/"
    A = sio.loadmat(path+'LastFM.mat')['A'].toarray()
    F0 = sio.loadmat(path+'LastFM.mat')['F'].flatten()
    F =np.unique(F0, return_inverse=True)[1]

    F_lastFM = copy.deepcopy(F)

    all_in_one = np.ones(len(F_lastFM))
    uniqe_vals, count = np.unique(F_lastFM, return_counts=True)
    LFM_balance_gender = min(count)/max(count)
    L = np.ones(F.shape[0])
    print("Shape of F matrix and A matrix, and Label matrix: ", np.shape(F), np.shape(A), np.shape(L))
    print("Dataset balance = ", LFM_balance_gender)

    return A, F, L

def load_nba():
    path1 = "NBA/"

    A = (pd.read_csv(path1 + "NBAgraph.csv", header=None)).to_numpy()
    F = (pd.read_csv(path1 + "NBAfeature.csv", header=None)).to_numpy()
    L = (pd.read_csv(path1 + "NBAlabel_binary.csv", header=None)).to_numpy()

    all_in_one = np.ones(F.shape[0])
    uniqe_vals, count = np.unique(F, return_counts=True)
    NBA_balance = min(count) / max(count)
    print("Shape of F matrix and A matrix, and Label matrix: ", np.shape(F), np.shape(A), np.shape(L))
    print("Dataset balance = ", NBA_balance)

    return A, F, L

def load_pokec_n():
    path1 = "Pokec/pre_processed/n/"

    A_sparse = load_npz(path1+"sparse_Pokec_graph_reg_1.npz").toarray()
    F = (pd.read_csv(path1 + "Pokec_feature_reg1.csv", header=None)).to_numpy()
    L = (pd.read_csv(path1 + "Pokec_binary_lable_reg1.csv", header=None)).to_numpy()

    all_in_one = np.ones(F.shape[0])
    uniqe_vals, count = np.unique(F, return_counts=True)
    Pokec_balance = min(count) / max(count)
    print("Shape of F matrix and A matrix, and Label matrix: ", np.shape(F), np.shape(A), np.shape(L))
    print("Dataset balance = ", Pokec_balance)

    return A_sparse, F, L
