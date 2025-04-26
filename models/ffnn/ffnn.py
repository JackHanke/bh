import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_ffnn(
        model: torch.nn.Module, 
        epochs: int, 
        train_loader: DataLoader,
        valid_loader: DataLoader = None,
        plot_learning_curves: bool = True,
        verbose: bool = False,
    ):

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters())

    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        # train
        model.train()

        epoch_train_loss, epoch_valid_loss = [], []
        for batch_num, (batch, label) in enumerate(train_loader):
            # zero gradients
            optim.zero_grad()
            # NOTE so dynamic!
            batch_len = len(batch)

            # reshape for network
            batch_reshaped = torch.reshape(batch, (batch_len,8*128*128))

            # make prediction
            pred = model.forward(batch_reshaped)
            pred_reshaped = torch.reshape(pred, (batch_len,8,128,128))

            # compute loss
            loss_value = loss_fn(pred_reshaped, label)
            epoch_train_loss.append(loss_value)
            # backprop
            loss_value.backward()
            # update paramts
            optim.step()

        avg_loss_after_epoch = sum(epoch_train_loss)/len(epoch_train_loss)
        if verbose: print(f"Train loss value: {avg_loss_after_epoch}")
        train_losses.append(avg_loss_after_epoch)


        # validation
        if valid_loader:
            model.eval()

            for batch_num, (batch, label) in enumerate(valid_loader):
                # NOTE still dynamic!
                batch_len = len(batch)
                # reshape for network
                batch_reshaped = torch.reshape(batch, (batch_len,8*128*128))

                # make prediction
                pred = model.forward(batch_reshaped)
                pred_reshaped = torch.reshape(pred, (batch_len,8,128,128))

                # compute loss
                loss_value = loss_fn(pred_reshaped, label)
                epoch_valid_loss.append(loss_value)

            avg_vloss_after_epoch = sum(epoch_train_loss)/len(epoch_train_loss)
            if verbose: print(f"Valid loss value: {avg_loss_after_epoch}")
            valid_losses.append(avg_vloss_after_epoch)

    # plot learning
    if plot_learning_curves:
        plt.plot([i for i in range(len(train_losses))], [loss.item() for loss in train_losses], label='Train Loss')
        if valid_loader: 
            plt.plot([i for i in range(len(valid_losses))], [loss.item() for loss in valid_losses], label='Validation Loss')
        plt.title(f'Training and Validation Curve')
        plt.xlabel(f'Number of Batches')
        plt.ylabel(f'Loss (MSE)')
        plt.legend()
        plt.show()
    return train_losses, valid_losses


class FFNN(nn.Module):
    def __init__(self, input_dim:int, version_str:str = 'v0.0.0'):
        super().__init__()
        self.version_num = version_str
        self.save_path = f'models/ffnn/saves/ffnn_{self.version_num}.pth'
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.save_path)
        print(f'Saved model as {self.save_path}')
