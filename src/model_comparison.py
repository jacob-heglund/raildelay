import os
import sys
project_root = "/home/jacobheglund/dev/raildelay/"
sys.path.append(project_root)
os.chdir(project_root)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_processing import data_interface

import pdb


import src.utils.model_comparison_utils as model_comparison_utils

## using the F1 features
# data_train = np.load("./data/processed/baseline_train.npy")
# data_test = np.load("./data/processed/baseline_test.npy")
data_dir = "./data/processed/raildelays/"
dataset = "raildelays"
n_nodes = 40
ks = 3
approx = "cheb_poly"
device = "cuda"
inf_mode = "individual"
n_timesteps_per_day = 42
n_timesteps_in = 12
n_timesteps_future = 6
model_type = "MLP"


Lk, data_train, data_test, data_val, output_stats = data_interface(data_dir,
                                                    dataset,
                                                    n_nodes,
                                                    ks,
                                                    approx,
                                                    device,
                                                    inf_mode,
                                                    n_timesteps_per_day,
                                                    n_timesteps_in,
                                                    n_timesteps_future)


mae_node = []
rmse_node = []

if model_type == "LR":
    for i in range(n_nodes):
        # pick out particular node's data
        model_data_train = data_train[0][:, :, i, :].squeeze(), data_train[1][:, :, i, :].squeeze()
        model_data_test = data_test[0][:, :, i, :].squeeze(), data_test[1][:, :, i, :].squeeze()

        # fit model, report results
        err = model_comparison_utils.linear_regression(model_data_train, model_data_test)
        mae_node.append(err[0])
        rmse_node.append(err[1])


elif model_type == "MLP":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU TIME BABY!")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    n_epochs = 50
    model = model_comparison_utils.MLP(n_timesteps_in, 100, 1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for i in range(n_nodes):
        # pick out particular node's data
        model_data_train = data_train[0][:, :, i, :].squeeze().to(device).float(), data_train[1][:, :, i, :].squeeze(axis=2).float()
        model_data_test = data_test[0][:, :, i, :].squeeze().to(device).float(), data_test[1][:, :, i, :].squeeze(axis=2).float()
        # fit model, report results
        err = model_comparison_utils.model_train(model_data_train, model_data_test, model, optimizer, criterion, n_epochs, device)
        mae_node.append(err[0])
        rmse_node.append(err[1])

print(model_type)
print("MAE", np.mean(mae_node))
print("RMSE:", np.mean(rmse_node))

# LR
# MAE 0.30547151399256933
# RMSE: 0.699626551173839

# MLP
# MAE 0.34320855
# RMSE: 0.94689673
