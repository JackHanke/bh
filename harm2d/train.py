# system imports
import os
import sys
import subprocess
import logging
import time

# training imports
import torch
import numpy as np
from tqdm import tqdm

# training utilities
from utils.sc_utils import custom_batcher, tensorize_globals
from models.cnn.cnn import CNN_3D


# training script
def train():
    # from harm2d.pp import rgdump_griddata
    # from harm2d.pp import *
    # import harm2d.pp as locpp

    # path to dumps
    dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'
    os.chdir(dumps_path)

    print('----Training script running!----')

    # number of data points
    num_dumps = 11 - 1
    # batch size
    batch_size = 2
    # number of epochs
    num_epochs = 2
    # access device, cuda device if accessible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Training on {num_dumps} dumps for {num_epochs} epochs at batch size = {batch_size}.')

    # set model
    model = CNN_3D().to(device)
    # set loss
    optim = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.MSELoss()

    # get indexes for training data
    train_indexes, validation_indexes = custom_batcher(
        batch_size=batch_size,
        num_dumps=num_dumps,
        split = 0.8,
        seed=1
    )

    num_train_batches = len(train_indexes)//batch_size
    num_valid_batches = len(validation_indexes)//batch_size

    best_validation = float('inf')

    # rblock_new replacement, improves performance
    global block, nmax, n_ord, AMR_TIMELEVEL
    with open("gdumps/grid", "rb") as fin:
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
        block = np.zeros((nmax, 200), dtype=np.int32, order='C')
        n_ord = np.zeros((nmax), dtype=np.int32, order='C')
        gd = np.fromfile(fin, dtype=np.int32, count=NV * nmax, sep='')
        gd = gd.reshape((NV, nmax), order='F').T
        start = time.time()
        block[:,0:NV] = gd
        if(NV<170):
            block[:, AMR_LEVEL1] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL2] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL3] = gd[:, AMR_LEVEL]

    locpp.rgdump_griddata(dumps_path)

    for epoch in range(num_epochs):
        ## Training
        model.train()
        epoch_train_loss = []

        # shuffle training indexes
        np.random.shuffle(train_indexes)

        # list of average train/validation losses after each epoch
        train_losses, valid_losses = [], []

        prog_bar = tqdm(enumerate(train_indexes.reshape(-1, batch_size)), total=num_train_batches)
        for batch_num, batch_indexes in prog_bar:
            ## fetch and tensorize data
            # NOTE everything is a global variable so it has to be this way. im sorry
            batch_data, label_data = [], []
            # batch_idx is the dump number
            for batch_idx in batch_indexes:

                # at every batch of size batch_size, we need to read in 2 * batch_size dumps
                
                ## get data frame
                # get data into global context NOTE this is really slow
                # rblock_new(batch_idx)
                rpar_new(batch_idx)
                rdump_griddata(dumps_path, batch_idx)
                # format data as tensor
                data_tensor = tensorize_globals(rho=rho, ug=ug, uu=uu, B=B)
                # add to batch
                batch_data.append(data_tensor)

                ## get label frame
                # get data into global context
                # rblock_new(batch_idx+1)
                rpar_new(batch_idx+1)
                # rgdump_griddata(dumps_path)
                rdump_griddata(dumps_path, batch_idx+1)
                # format data as tensor
                data_tensor = tensorize_globals(rho=rho, ug=ug, uu=uu, B=B)
                # add to batch
                label_data.append(data_tensor)

            # final tensorize
            batch_data = torch.cat(batch_data, dim=0).to(device)
            label_data = torch.cat(label_data, dim=0).to(device)

            ## train model
            # make prediction
            pred = model.forward(batch_data)
            # compute loss
            loss_value = loss_fn(pred, label_data)
            epoch_train_loss.append(loss_value)
            # backprop
            loss_value.backward()
            # update paramts
            optim.step()

            prog_bar.set_description(f'Train batch {batch_num+1} completed with loss {loss_value.item():.4f}')

        # training loss tracking
        avg_loss_after_epoch = sum(epoch_train_loss)/len(epoch_train_loss)
        train_losses.append(avg_loss_after_epoch)
        print(f"Train loss value: {avg_loss_after_epoch}")


        ## Validation
        model.eval()
        epoch_valid_loss = []

        prog_bar = tqdm(enumerate(validation_indexes.reshape(-1, batch_size)), total=num_valid_batches)
        for batch_num, batch_indexes in prog_bar:
            ## fetch and tensorize data
            # NOTE everything is a global variable so it has to be this way. im sorry
            batch_data, label_data = [], []
            # batch_idx is the dump number
            for batch_idx in batch_indexes:
                ## get data frame
                # get data into global context
                rblock_new(batch_idx)
                rpar_new(batch_idx)
                rgdump_griddata(dumps_path)
                rdump_griddata(dumps_path, batch_idx)
                # format data as tensor
                data_tensor = tensorize_globals(rho=rho, ug=ug, uu=uu, B=B)
                # add to batch
                batch_data.append(data_tensor)

                ## get label frame
                # get data into global context
                rblock_new(batch_idx+1)
                rpar_new(batch_idx+1)
                rgdump_griddata(dumps_path)
                rdump_griddata(dumps_path, batch_idx+1)
                # format data as tensor
                data_tensor = tensorize_globals(rho=rho, ug=ug, uu=uu, B=B)
                # add to batch
                label_data.append(data_tensor)

            # final tensorize
            batch_data = torch.cat(batch_data, dim=0).to(device)
            label_data = torch.cat(label_data, dim=0).to(device)

            # make prediction
            pred = model.forward(batch_data)

            # compute loss
            loss_value = loss_fn(pred, label_data)
            epoch_valid_loss.append(loss_value)
            
            prog_bar.set_description(f'Validation batch {batch_num+1} completed with loss {loss_value.item():.4f}.')
            
        avg_vloss_after_epoch = sum(epoch_train_loss)/len(epoch_train_loss)
        valid_losses.append(avg_vloss_after_epoch)
        print(f"Valid loss value: {avg_loss_after_epoch}")

        # checkpointing
        if avg_vloss_after_epoch < best_validation:
            best_validation = avg_vloss_after_epoch
            save_path = os.environ['HOME'] + '/bh/' + model.save_path
            model.save(save_path=save_path)

