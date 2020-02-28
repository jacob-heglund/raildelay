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

    # make input to STGCN X.shape = (batch_size, c_in, n_timesteps_in, n_nodes)
    # TODO specify this size in the model docs (#code_goals)
    train_input, train_label = train_input.permute(0, 3, 1, 2), train_label.permute(0, 3, 1, 2)

    val_input, val_label = data_val
    val_input, val_label = val_input.permute(0, 3, 1, 2), val_label.permute(0, 3, 1, 2)

    perm = torch.randperm(train_input.shape[0])
    val_idx = np.arange(0, val_input.shape[0], 1)

    best_metrics = np.ones((2, 1)) * 1e6


    for epoch in range(args.n_epochs):
        print("\n\nEpoch: {}/{}".format(epoch, args.n_epochs))
        # training
        print("Training")
        avg_training_loss = 0
        batch_count = 0
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

            if i % 5000 == 0:
                print("Step: {}/{}".format(i, train_input.shape[0]))

        print("Training Time: {} sec".format(round(time.time() - start_time, 2)))
        avg_training_loss /= batch_count
        tf.summary.scalar("Average Training Loss", avg_training_loss, epoch)

        # validation
        print("\nValidation")
        # mape = 0
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

                # mape_batch, rmse_batch, mae_batch = evaluation(y, y_hat, output_stats)
                # mape += mape_batch

                rmse_batch, mae_batch = evaluation(y, y_hat, output_stats)
                rmse += rmse_batch
                mae += mae_batch
                batch_count += 1

            # mape /= batch_count
            rmse /= batch_count
            mae /= batch_count

            print("RMSE: ", rmse)
            print("MAE: ", mae)
            print("Inference Time: {} sec".format(round(time.time() - start_time, 2)))
            if rmse < best_metrics[0] and mae < best_metrics[1]:
                best_metrics[0] = rmse
                best_metrics[1] = mae
                # save highest-performing model

            # record metrics
            # tf.summary.scalar("Mean Absolute Percentage Error", mape, epoch)
            tf.summary.scalar("Root Mean Squared Error", rmse, epoch)
            tf.summary.scalar("Mean Absolute Error", mae, epoch)
            writer.flush()

    return best_model

#TODO implement model testing
def model_test():
    pass


