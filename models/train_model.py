import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# train an FFNN, CNN, or UNet
def train_model(
        model: torch.nn.Module, 
        epochs: int, 
        train_loader: DataLoader,
        valid_loader: DataLoader = None,
        loss_fn: torch.nn = torch.nn.MSELoss(),
        baseline: bool = True,
        flatten: bool = False,
        plot_learning_curves: bool = True,
        verbose: bool = False,
    ):

    optim = torch.optim.Adam(params=model.parameters())

    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        # train
        model.train()

        epoch_train_loss = []
        for batch_num, (batch, label) in enumerate(train_loader):
            # zero gradients
            optim.zero_grad()
            # NOTE so dynamic!
            batch_len = len(batch)

            # reshape for network
            if flatten: batch = torch.reshape(batch, (batch_len, model.input_dim))

            # make prediction
            pred = model.forward(batch)
            if flatten: pred = torch.reshape(pred, (batch_len, model.input_dim))

            # compute loss
            loss_value = loss_fn(pred, label)
            epoch_train_loss.append(loss_value)
            
            # backprop
            loss_value.backward()
            # update paramts
            optim.step()

        # training loss tracking
        avg_loss_after_epoch = sum(epoch_train_loss)/len(epoch_train_loss)
        if verbose: print(f"Train loss value: {avg_loss_after_epoch}")
        train_losses.append(avg_loss_after_epoch)

        # get average of data for baseline
        if baseline:
            num_train_datapoints = 0
            avg_of_data = torch.zeros([1]+list(batch.shape[1:]))
            for batch_num, (batch, label) in enumerate(train_loader):
                num_train_datapoints += batch_len
                avg_of_data += torch.sum(batch, dim=0, keepdim=True)
            
            avg_of_data = avg_of_data / num_train_datapoints


        # validation
        if valid_loader:
            model.eval()

            epoch_valid_loss = []
            baselines_valid_loss = []
            for batch_num, (batch, label) in enumerate(valid_loader):
                # NOTE still dynamic!
                batch_len = len(batch)
                # reshape for network
                if flatten: batch = torch.reshape(batch, (batch_len,8*128*128))

                # make prediction
                pred = model.forward(batch)
                if flatten: pred = torch.reshape(pred, (batch_len,8,128,128))

                # compute loss
                loss_value = loss_fn(pred, label)
                epoch_valid_loss.append(loss_value)

                if epoch == 0 and baseline: 
                    # compute validation loss if one predicts the average of the data
                    
                    temp = avg_of_data.repeat([batch_len] + [1 for _ in range(len(batch.shape)-1)])
                    epoch_baseline_loss_value = loss_fn(temp, label)
                    baselines_valid_loss.append(epoch_baseline_loss_value)

            avg_vloss_after_epoch = sum(epoch_train_loss)/len(epoch_train_loss)
            if verbose: print(f"Valid loss value: {avg_loss_after_epoch}")
            valid_losses.append(avg_vloss_after_epoch)

            if epoch == 0:
                avg_baseline_loss = sum(baselines_valid_loss)/len(baselines_valid_loss)
                if verbose: print(f"Baseline valid loss value: {avg_baseline_loss}")


    # plot learning
    if plot_learning_curves:
        plt.plot([i for i in range(len(train_losses))], [loss.item() for loss in train_losses], label='Train Loss')
        plt.plot([i for i in range(len(train_losses))], [avg_baseline_loss for _ in range(len(train_losses))], label='Predicting Avg Loss', linestyle='dashed')
        if valid_loader: 
            plt.plot([i for i in range(len(valid_losses))], [loss.item() for loss in valid_losses], label='Validation Loss')

        plt.title(f'Training and Validation Curve')
        plt.xlabel(f'Number of Batches')
        plt.ylabel(f'Loss (MSE)')
        if baseline: plt.ylim(top=2*avg_baseline_loss)
        plt.legend()
        plt.show()

    return train_losses, valid_losses, training_baseline