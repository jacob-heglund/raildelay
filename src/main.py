#######################################
# TODO
# Big Stuff
# code_goals
## remove the if dataset== blah statements in data processing, everything will be in separate notebooks
##  and only 1 dataset will be processed at a time

# fix the data preprocessing
## this will be fixed as part of the #code_goals update

# improve training speed
## removing as many permute and contiguous commands improved by 3-4%%, we need an order of magnitude improvement
## look at Pytorch STGCN, see how they did it
## my model is using tons of memory that the other implementations don't, what's up with that?

# other stuff for a finalized version of the model
## get metrics on test dataset
## save best models

# data pipeline cleanup
## this will be fixed as part of the #code_goals update
## integrate section 5 into the architecture
## integrate adjacency matrix calculation data preprocessing, make it more efficient

#######################################
# multiple output features
## this can be included as part of the #code_goals update
# parser.add_argument("n_features_out", type=int, default=1)
## as part of data preprocessing, add singleton dimension, be careful of what features you're selecting (need to know column ordering from notebook)


############################
# imports
############################
print("\n\n\n\nimporting libraries")
import pdb

# external libraries
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import argparse
import json
from datetime import datetime

# custom libraries
from models.model import STGCN
from models.model_train import model_train, model_test
from data.data_processing import data_interface

# base_dir = r"C:\home\dev\research\delayprediction\delayprediction\models\stgcn_dev\miso"
# base_dir = r"/home/jacobheglund/dev/miso"
base_dir = r"/home/jacobheglund/dev/raildelay"
os.chdir(base_dir)
print(os.getcwd())

# torch.manual_seed(225)
parser = argparse.ArgumentParser()
############################
# initial setup
############################
print("\n\n\n\ninitial setup")
parser.add_argument("--dataset", type=str, default="raildelays", choices={"airport_delay_50", "pems_228", "raildelays"})
#TODO fix the argument parsing, when I parse it here I can't add the stuff below like n_epochs (#code_goals)
args = parser.parse_args()

data_dir = "./data/processed/" + args.dataset + "/"
log_dir = "./models/" + args.dataset + "/output/"
log_dir += datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
ckpt_dir = "./models/" + args.dataset + "/checkpoints/"
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

writer = tf.summary.create_file_writer(log_dir)

#TODO this is not the right setup for top_30 airports! (#code_goals)
if args.dataset == "airport_delay_50":
    n_nodes = 50
    n_timesteps_per_day = 20
    n_timesteps_in = 5
    n_timesteps_future = 1
    inf_mode = "individual"
    n_features_in = 2
    n_features_out = 1

#TODO make sure this setup replicates the original paper (#code_goals)
elif args.dataset == "pems_228":
    n_nodes = 228
    n_timesteps_per_day = 288
    n_timesteps_in = 12
    n_timesteps_future = 3
    inf_mode = "individual"
    n_features_in = 1
    n_features_out = 1

elif args.dataset == "raildelays":
    n_nodes = 40
    n_timesteps_per_day = 42
    n_timesteps_in = 12
    n_timesteps_future = 6
    inf_mode = "individual"
    n_features_in = 1
    n_features_out = 1

else:
    print("Specified dataset currently not supported")
    exit()


# data parameters
parser.add_argument("--n_nodes", type=int, default=n_nodes)
parser.add_argument("--n_timesteps_per_day", type=int, default=n_timesteps_per_day)
parser.add_argument("--n_timesteps_in", type=int, default=n_timesteps_in)
parser.add_argument("--n_timesteps_future", type=int, default=n_timesteps_future)
parser.add_argument("--inf_mode", type=str, default=inf_mode, choices={"individual", "multiple"}, help="available inference modes\n\
                                                individual: the model is fit to a single timestep at n_timesteps_future in the future\n\
                                                multiple: the model is fit to all timesteps between present and n_timesteps_future in the future")
parser.add_argument("--approx", type=str, default="cheb_poly", choices={"cheb_poly", "first_order"})

# kernel sizes for spatial (graph) and temporal convolutions
# don't change these sizes, the convolutions won't work otherwise
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--kt", type=int, default=3)

# training parameters
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=25)
parser.add_argument("--learning_rate", type=int, default=1e-3)
parser.add_argument("--optimizer", type=str, default="ADAM", choices={"ADAM"})
parser.add_argument("--drop_prob", type=int, default=0.0)

# GPU setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU TIME BABY!")
else:
    device = torch.device("cpu")
    print("Using Device: CPU")
parser.add_argument("--device", type=str, default=device)
args = parser.parse_args()

# check inference mode
if args.inf_mode == "individual":
    pass
elif args.inf_mode == "multiple":
    pass
else:
    print("Please enter a valid inf_mode setting.")
    sys.exit()

############################
# data interface
############################
#TODO none of this should be done here for "in-house" code, put this in a notebook #code_goals
## it should be here for published code (i.e. public facing for other researchers), but that's
## not what i'm working on right now

print("\n\n\n\nload from disk: dataset={}".format(args.dataset))

Lk, data_train, data_test, data_val, output_stats = data_interface(data_dir,
                                                    args.dataset,
                                                    args.n_nodes,
                                                    args.ks,
                                                    args.approx,
                                                    args.device,
                                                    args.inf_mode,
                                                    args.n_timesteps_per_day,
                                                    args.n_timesteps_in,
                                                    args.n_timesteps_future)

# print(output_stats["mean"])
# print(output_stats["std"])
############################
# training setup
############################
print("\n\n\n\ntraining setup")

blocks = [[n_features_in, 32, 64], [64, 32, 128], [128, n_features_out]]
model = STGCN(blocks,
                args.n_timesteps_in,
                args.n_timesteps_future,
                args.n_nodes,
                args.device,
                args.inf_mode,
                args.ks,
                args.kt,
                args.drop_prob).to(args.device)

loss_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

############################
# training loop
############################
if __name__ == "__main__":
    with writer.as_default():
        print("\n\n\n\ntraining loop")
        model_train(data_train, data_val, data_test, output_stats, Lk, model, optimizer, scheduler, loss_criterion, writer, args, ckpt_dir)
