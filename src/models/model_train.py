############################
# imports
############################
# external libraries
import pdb

import numpy as np
import tensorflow as tf
import time
import torch

# custom libraries
from utils.utils import evaluation

############################
# functions
############################


def model_train(data_train, data_val, output_stats, graph_kernel, model, optimizer, loss_criterion, writer, args):
    # randomize and batch data
    train_input, train_label = data_train
    # TODO make input to STGCN X.shape = (batch_size, c_in, n_timesteps_in, n_nodes)
    train_input, train_label = train_input.permute(0, 3, 1, 2), train_label.permute(0, 3, 1, 2)

    val_input, val_label = data_val
    val_input, val_label = val_input.permute(0, 3, 1, 2), val_label.permute(0, 3, 1, 2)

    perm = torch.randperm(train_input.shape[0])
    val_idx = np.arange(0, val_input.shape[0], 1)

    for epoch in range(args.n_epochs):
        print("Epoch: {}/{}".format(epoch, args.n_epochs))
        avg_training_loss = 0
        batch_count = 0
        # training
        model.train()
        start_time = time.time()
        for i in range(0, train_input.shape[0], args.batch_size):
            optimizer.zero_grad()
            idx = perm[i:i+args.batch_size]
            X, y = train_input[idx].to(args.device), train_label[idx].to(args.device)
            y_hat = model(X, graph_kernel)
            loss = loss_criterion(y_hat, y)
            avg_training_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if i % 1000 == 0:
                # for PEMS traffic
                # print("Step: {}/{}".format(i, 8223))
                
                # for airline delays
                print("Step: {}/{}".format(i, 15000))
        
        print("Training Time: {} sec".format(round(time.time() - start_time, 2)))
        avg_training_loss /= batch_count
        tf.summary.scalar("Average Training Loss", avg_training_loss, epoch)

        # validation
        mape = 0
        rmse = 0
        mae = 0
        batch_count = 0
        with torch.no_grad():
            model.eval()
            start_time = time.time()
            for i in range(0, val_input.shape[0], args.batch_size):
                idx = val_idx[i:i+args.batch_size]
                X = val_input[idx].to(args.device)
                y = val_label[idx]
                y_hat = model(X, graph_kernel)
                y_hat = y_hat.cpu()

                mape_batch, rmse_batch, mae_batch = evaluation(y, y_hat, output_stats)
                mape += mape_batch
                rmse += rmse_batch
                mae += mae_batch
                batch_count += 1

            print("Inference Time: {} sec".format(round(time.time() - start_time, 2)))
            mape /= batch_count
            rmse /= batch_count
            mae /= batch_count
            
            print("MAPE Batch: ", mape)
            print("RMSE Batch: ", rmse)
            print("MAE Batch: ", mae)            

            # record metrics
            tf.summary.scalar("Mean Absolute Percentage Error", mape, epoch)
            tf.summary.scalar("Root Mean Squared Error", rmse, epoch)
            tf.summary.scalar("Mean Absolute Error", mae, epoch)
            writer.flush()
            
#TODO implement model testing
def model_test():
    pass


