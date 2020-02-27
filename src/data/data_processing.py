############################
# imports
############################
# external libraries
import pdb

import numpy as np
import os
import json
import torch
import sys
import pandas as pd
from scipy.sparse.linalg import eigs

# custom libraries
from utils.utils import z_score


############################
# functions
############################


#TODO docs
def data_interface(data_dir,
                    dataset,
                    n_nodes,
                    ks,
                    approx,
                    device,
                    inf_mode,
                    n_timesteps_per_day,
                    n_timesteps_in,
                    n_timesteps_future):

    Lk = process_adjacency(data_dir,
                    dataset,
                    ks,
                    n_nodes,
                    approx,
                    device)

    #TODO use as the interface to pull out the featurse
    features = ["arr_delay", "scheduled_flights"]
    labels = "arr_delay"
    if dataset == "airport_delay_50":
        process_data(data_dir,
                    dataset,
                    features,
                    n_nodes)

    dataset_seq = sequence_data(data_dir,
                                dataset,
                                n_nodes,
                                n_timesteps_per_day,
                                n_timesteps_in,
                                n_timesteps_future)

    data_train, data_test, data_val, output_stats = split_data(dataset_seq, n_timesteps_in, inf_mode)

    return Lk, data_train, data_test, data_val, output_stats


#TODO docs
def create_output_stats(df, cols):
    output_stats = np.zeros((2, len(cols)))
    for i in range(len(cols)):
        mean = df[cols[i]].mean()
        std = df[cols[i]].std(ddof=0)
        output_stats[i, 0] = mean
        output_stats[i, 1] = std

    return output_stats


#TODO docs
def process_data(data_dir, dataset, feature_cols, n_nodes):
    df_data = pd.read_pickle(data_dir + "enriched")

    # process database to form numpy database
    if not os.path.exists(data_dir + "dataset.npy"):
        date_list = df_data["datetime"].unique()
        n_timesteps = len(date_list)
        drop_list_1 = ["carrier_delay", "day_of_week", "day_of_year", "dep_delay", "hour_of_day",
                    "late_aircraft_delay", "weather_delay", "arr_delay_zscore", "scheduled_flights_zscore",
                    "arr_delay_class", "carrier_delay_zscore", "dep_delay_zscore", "late_aircraft_delay_zscore",
                    "weather_delay_zscore", "day_of_week_cos", "day_of_week_sin", "day_of_year_cos",
                    "day_of_year_sin", "hour_of_data", "hour_of_day_cos", "hour_of_day_sin"]
        df_data = df_data.drop(labels=drop_list_1, axis=1)

        #TODO be able to reverse this process and assign predictions to specific airports
        # drop everything except feature_cols
        drop_list_2 = ["datetime", "origin_airport_code"]

        n_features = np.shape(df_data)[1] - len(drop_list_2)
        dataset = np.zeros((n_timesteps, n_nodes, n_features))

        print("Extracting Data")
        for i in range(n_timesteps):
            if i % 1000 == 0:
                print("{}/{}".format(i, len(date_list)))

            # get the data for all airports at each unique datetime
            df_tmp = df_data.loc[df_data["datetime"] == date_list[i]]
            df_tmp.sort_values("origin_airport_code")

            # drop unnecessary features
            df_tmp = df_tmp.drop(columns = drop_list_2)
            feature_vec = df_tmp.to_numpy()
            dataset[i] = feature_vec

        np.save(data_dir + "dataset.npy", dataset)
        del dataset


def sequence_data(data_dir, dataset_name, n_nodes, n_timesteps_per_day, n_timesteps_in, n_timesteps_out):
    """loads data from disk and processes into sequences

    Args:
        data_dir (str): path of the directory where all loaded data is saved
        dataset.shape = (n_timesteps_total, n_nodes, n_features_in)
        dataset_name (str): same as args.dataset
        n_nodes (int): number of nodes on the graph
        n_timesteps_per_day (int): number of timesteps included in each day of the data (EX: 0400 - 2400: 20 hours)
        n_timesteps_in (int): number of timesteps to include as model input
        n_timesteps_out (int): number of timesteps to include as model labels, also number of timesteps that are predicted

    Returns:
        torch.Tensor: input and labels for the model, only the first dimension chances for the other outputs
        shape(train_input) = (n_sequences_train, n_nodes, n_timesteps_in, n_features_in)
        shape(train_label) = (n_sequences_train, n_nodes, n_timesteps_out, n_features_out = 1)
    """
    if dataset_name == "airport_delay_50":
        dataset = np.load(data_dir + "dataset.npy")
        n_features_in = np.shape(dataset)[2]

    elif dataset_name == "pems_228":
        dataset = pd.read_csv(data_dir + "V_228.csv", header=None).values
        dataset = np.expand_dims(dataset, 2)
        n_features_in = 1

    elif dataset_name == "raildelays":
        dataset = np.load(data_dir + "raildelays.npy")
        n_features_in = 1

    n_days = int(np.shape(dataset)[0] / n_timesteps_per_day)
    n_timesteps_seq = n_timesteps_in + n_timesteps_out

    # number of sequences per day
    n_slot = n_timesteps_per_day - n_timesteps_seq + 1

    # total number of sequences
    n_sequences = (n_slot-1) * n_days
    dataset_seq = np.zeros((n_sequences, n_timesteps_seq, n_nodes, n_features_in))

    # get to the correct day
    counter = 0
    for i in range(n_days):
        curr_data = dataset[i * n_timesteps_per_day : (i + 1) * n_timesteps_per_day]

        # get the input-output sequences within the day
        for j in range(n_slot-1):
            input_seq = curr_data[j : j + n_timesteps_in]
            output_seq = curr_data[j + n_timesteps_in : j + n_timesteps_in + n_timesteps_out]
            tmp_data = np.expand_dims(np.concatenate((input_seq, output_seq)), 0)

            dataset_seq[counter] = tmp_data
            counter += 1

    return dataset_seq


def split_data(dataset_seq, n_timesteps_in, inf_mode, percent_train=0.7, percent_test=0.1, percent_val=0.2):
    """Splits a dataset of sequences into train, test, val

    Args:
        dataset_seq (array): dataset of sequences, of size (n_sequences, n_timesteps_in + n_timesteps_out , n_nodes, n_features_in)
        n_timesteps_in (int): number of timesteps to include as model input
        percent_train (float, optional): proportion of data used for training. Defaults to 0.7.
        percent_test (float, optional): proportion of data used for testing. Defaults to 0.2.
        percent_val (float, optional): proportion of data used for validation. Defaults to 0.1.

    Returns:
        input and labels for the model

        data_train, data_test, data_val

        An example of the output shapes is given below, only the first dimension changes for the other outputs
        train_input, train_label = data_train[0], data_train[1]
        train_input.shape = (n_sequences_train, n_timesteps_in, n_nodes, n_features_in)
        train_label.shape = (n_sequences_train, n_timesteps_out, n_nodes, n_features_out)
    """
    # randomize data
    dataset_seq = np.random.permutation(dataset_seq)
    train_start = 0
    train_end = train_start + int(len(dataset_seq) * percent_train)

    val_start = train_end + 1
    val_end = val_start + int(len(dataset_seq) * percent_val)

    test_start = val_end + 1
    test_end = test_start + int(len(dataset_seq) * percent_test)

    dataset_train = dataset_seq[train_start:train_end, :]
    dataset_test = dataset_seq[test_start:test_end, :]
    dataset_val = dataset_seq[val_start:val_end, :]

    train_input, train_label = generate_input_label(dataset_train, n_timesteps_in, inf_mode)
    test_input, test_label = generate_input_label(dataset_test, n_timesteps_in, inf_mode)
    val_input, val_label = generate_input_label(dataset_val, n_timesteps_in, inf_mode)

    #TODO only works for 1 output feature
    output_data = np.vstack((train_label, test_label, val_label))
    output_stats = {"mean":np.mean(output_data), "std":np.std(output_data)}

    #TODO this is fucked, you can't use the hourly delay z-score to normalize the fucking # of flights!
    train_input = z_score(train_input, output_stats["mean"], output_stats["std"])
    train_label = z_score(train_label, output_stats["mean"], output_stats["std"])
    test_input = z_score(test_input, output_stats["mean"], output_stats["std"])
    test_label = z_score(test_label, output_stats["mean"], output_stats["std"])
    val_input = z_score(val_input, output_stats["mean"], output_stats["std"])
    val_label = z_score(val_label, output_stats["mean"], output_stats["std"])

    data_train = train_input, train_label
    data_test = test_input, test_label
    data_val = val_input, val_label

    return data_train, data_test, data_val, output_stats


#TODO docs
def generate_input_label(dataset, n_timesteps_in, inf_mode):
    data_input = torch.from_numpy(dataset[:, 0:n_timesteps_in, :, :]).double()
    if inf_mode == "individual":
        #TODO this is where the output label is chosen
        data_label = torch.from_numpy(dataset[:, -1, :, 0]).double().unsqueeze(1)
    elif inf_mode == "multiple":
        #TODO this is where the output label is being chose right now
        data_label = torch.from_numpy(dataset[:, n_timesteps_in:, :, 0]).double()
    # standardize to 4 dimensional tensor
    if len(data_label == 3):
        data_label = data_label.unsqueeze(3)

    return data_input, data_label


#TODO docs
def process_adjacency(data_dir, dataset, ks, n_nodes, approx, device):
    if dataset == "airport_delay_50":
        A = np.load(data_dir + "adj_matrix.npy")
    elif dataset == "pems_228":
        A = pd.read_csv(data_dir + "W_228.csv", header=None).values
        A = weight_matrix(A)
    elif dataset == "raildelays":
        A = np.load(data_dir + "raildelays_adj.npy")

    L = scaled_laplacian(A)
    if approx == "cheb_poly":
        Lk = cheb_poly_approx(L, ks, n_nodes)
    elif approx == "first_order":
        Lk = first_approx(L, n_nodes)

    Lk = torch.from_numpy(Lk).to(device)
    return Lk


# copied from the original paper's implementation
def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which="LR")[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


# copied from the original paper's implementation
def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError("ERROR: the size of spatial kernel must be greater than 1, but received {}".format(Ks))


# copied from the original paper's implementation
def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


# copied from the original paper's implementation
def weight_matrix(W, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print("The input graph is a 0/1 matrix; set 'scaling' to False.")
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2 = W * W
        W_mask = np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W
