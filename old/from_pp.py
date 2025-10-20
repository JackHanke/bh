## MSAI work starts here


# rblock_new_ml()
def rblock_new_ml():
    global AMR_ACTIVE, AMR_LEVEL,AMR_LEVEL1,AMR_LEVEL2,AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    global AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8
    global AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4, AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP
    global AMR_CORN1, AMR_CORN2, AMR_CORN3, AMR_CORN4, AMR_CORN5, AMR_CORN6
    global AMR_CORN7, AMR_CORN8, AMR_CORN9, AMR_CORN10, AMR_CORN11, AMR_CORN12
    global AMR_NBR1_3, AMR_NBR1_4, AMR_NBR1_7, AMR_NBR1_8, AMR_NBR2_1, AMR_NBR2_2, AMR_NBR2_3, AMR_NBR2_4, AMR_NBR3_1, AMR_NBR3_2, AMR_NBR3_5, AMR_NBR3_6, AMR_NBR4_5, AMR_NBR4_6, AMR_NBR4_7, AMR_NBR4_8
    global AMR_NBR5_1, AMR_NBR5_3, AMR_NBR5_5, AMR_NBR5_7, AMR_NBR6_2, AMR_NBR6_4, AMR_NBR6_6, AMR_NBR6_8
    global AMR_NBR1P, AMR_NBR2P, AMR_NBR3P, AMR_NBR4P, AMR_NBR5P, AMR_NBR6P
    global block, nmax, n_ord, AMR_TIMELEVEL

    AMR_ACTIVE, AMR_LEVEL, AMR_REFINED = 0,1,2
    AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT = 3,4,5,6
    AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8 = 7, 8, 9, 10, 11, 12, 13, 14
    AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4, AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP = 15,16,17,18,19,20,21,22,23
    AMR_CORN1, AMR_CORN2, AMR_CORN3, AMR_CORN4, AMR_CORN5, AMR_CORN6, AMR_CORN7, AMR_CORN8, AMR_CORN9, AMR_CORN10, AMR_CORN11, AMR_CORN12 = 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
    AMR_LEVEL1, AMR_LEVEL2, AMR_LEVEL3 = 110,111,112
    AMR_NBR1_3, AMR_NBR1_4, AMR_NBR1_7, AMR_NBR1_8, AMR_NBR2_1, AMR_NBR2_2, AMR_NBR2_3, AMR_NBR2_4, AMR_NBR3_1, AMR_NBR3_2, AMR_NBR3_5, AMR_NBR3_6, AMR_NBR4_5, AMR_NBR4_6, AMR_NBR4_7, AMR_NBR4_8, AMR_NBR5_1, AMR_NBR5_3, AMR_NBR5_5, AMR_NBR5_7, AMR_NBR6_2, AMR_NBR6_4, AMR_NBR6_6, AMR_NBR6_8=113, 114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136
    AMR_NBR1P, AMR_NBR2P, AMR_NBR3P, AMR_NBR4P, AMR_NBR5P, AMR_NBR6P=161,162,163,164,165,166
    AMR_TIMELEVEL=36

    if(os.path.isfile("gdumps/grid")):
        fin = open("gdumps/grid", "rb")
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
        # end = time.time()
        # print(f"End of elif: {end - start}")

    else:
        print("Cannot find grid file!")
        return

    with open("gdumps/grid", "rb") as fin:
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
        block = np.zeros((nmax, 200), dtype=np.int32, order='C')
        n_ord = np.zeros((nmax), dtype=np.int32, order='C')
        gd = np.fromfile(fin, dtype=np.int32, count=NV * nmax, sep='')
        gd = gd.reshape((NV, nmax), order='F').T
        # start = time.time()
        block[:,0:NV] = gd
        if(NV<170):
            block[:, AMR_LEVEL1] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL2] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL3] = gd[:, AMR_LEVEL]

# system imports
import os
import sys
import subprocess
import logging
import time
import pickle
import yaml

# training imports
import numpy as np
from tqdm import tqdm
import torch
from torchinfo import summary
# distributed training
import torch.distributed as dist  # NEW: Import for distributed training
import torch.multiprocessing as mp  # NEW: Import for multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP  # NEW: Import DDP wrapper
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# local training utilities
from utils.sc_utils import custom_batcher, tensorize_globals
from models.cnn.threed_cnn import *


# set params
lowres1 = 1 # 
lowres2 = 1 # 
lowres3 = 1 # 
r_min, r_max = 1.0, 100.0
theta_min, theta_max = 0.0, 9
phi_min, phi_max = -1, 9
do_box=0
set_cart=0
set_mpi(0)
axisym=1
print_fieldlines=0
export_raytracing_GRTRANS=0
export_raytracing_RAZIEH=0
kerr_schild=0
DISK_THICKNESS=0.03
check_files=1
notebook=1
interpolate_var=0
AMR = 0 # get all data in grid

# make batch from batch_indexes
def construct_batch(batch_indexes: list, dumps_path: str):
    batch_data, label_data = [], []
    for idx in batch_indexes:
        idx = idx.item()
        # create single data frame
        rpar_new(idx)
        rgdump_griddata(dumps_path)
        rdump_griddata(dumps_path, idx)
        batch_data.append(tensorize_globals(rho=np.log10(rho), ug=np.log10(ug), uu=uu, B=B))
        # create single label frame
        rpar_new(idx+1)
        rdump_griddata(dumps_path, idx+1)
        label_data.append(tensorize_globals(rho=np.log10(rho), ug=np.log10(ug), uu=uu, B=B))
    batch_data = torch.cat(batch_data)
    label_data = torch.cat(label_data)
    return batch_data, label_data


# training script
def train(model_path: str, device):
    global notebook, axisym,set_cart,axisym,REF_1,REF_2,REF_3,set_cart,D,print_fieldlines
    global lowres1,lowres2,lowres3, RAD_M1, RESISTIVE, export_raytracing_GRTRANS, export_raytracing_RAZIEH,r1,r2,r3
    global r_min, r_max, theta_min, theta_max, phi_min,phi_max, do_griddata, do_box, check_files, kerr_schild

    logger = logging.getLogger(__name__)
    # logs saves to training.log in harm2d directory
    logging.basicConfig(
        filename='training.log',
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    import yaml
    with open('train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # path to dumps
    dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'
    os.chdir(dumps_path)

    print('--- Training script running! ---')

    # number of data points
    num_dumps = config['num_dumps']
    # batch size
    batch_size = config['batch_size']
    # number of epochs
    num_epochs = config['num_epochs']
    # get range of dumps, from start inclusive to end exclusive
    start_dump = config['start_dump']
    end_dump = config['end_dump']
    # access device, cuda device if accessible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_hyperparams_str = f'Training on dumps {start_dump} - {end_dump} for {num_epochs} epochs at batch size = {batch_size} on {device} device.'
    print(training_hyperparams_str)
    logger.info(training_hyperparams_str)

    # set model
    model = B3_CNN().to(device)

    # bring in model weights if model_path is provided
    if model_path is not None:
        model_dict_path = model_path
        model_dict = torch.load(model_dict_path)
        model.load_state_dict(model_dict)
        if rank == 0:
            model_weights_info_str = f"Loaded weights from: {model_path}"
            logger.info(model_weights_info_str)
            print(model_weights_info_str)
    else:
        if rank == 0:
            model_weights_info_str = f"Randomly initializing weights."
            logger.info(model_weights_info_str)
            print(model_weights_info_str)
    
    summary_str = summary(model, input_size=(batch_size, 8, 224, 48, 96))
    logger.info('\n'+str(summary_str))

    # set loss
    optim = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.MSELoss()

    # get indexes for training data
    train_indexes, validation_indexes = custom_batcher(
        batch_size=batch_size,
        num_dumps=num_dumps,
        split = 0.8,
        seed=1,
        start=start_dump,
        end=end_dump,
    )

    num_train_batches = len(train_indexes)//batch_size
    num_valid_batches = len(validation_indexes)//batch_size

    best_validation = float('inf')

    # rewrite for performance
    rblock_new_ml()

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
            start = time.time()
            
            # construct batch of data manually
            batch_data, label_data = construct_batch(
                batch_indexes=batch_indexes, 
                dumps_path=dumps_path,
                device=device
            )
            
            # send data to device
            batch_data, label_data = batch_data.to(device), label_data.to(device)

            logger.info(f'batch size {batch_size} data made in {time.time()-start:.4f} ')

            ## train model
            # make prediction
            pred = model.forward(batch_data)
            # compute loss
            loss_value = loss_fn(pred, label_data)
            epoch_train_loss.append(loss_value)
            # backprop
            loss_value.backward()
            # clip gradients to 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # update paramts
            optim.step()

            # memory save maybe idk
            batch_data = None
            label_data = None
            torch.cuda.empty_cache()

            # training batch logging
            batch_str = f'Epoch {epoch+1} train batch {batch_num+1} completed with loss {loss_value.item():.4f} in {time.time()-start:.2f}s'
            prog_bar.set_description(batch_str)
            logger.debug(batch_str)

        # training loss tracking
        avg_loss_after_epoch = sum(epoch_train_loss)/len(epoch_train_loss)
        train_losses.append(avg_loss_after_epoch)

        # training logging
        train_loss_str = f"Epoch {epoch+1} train loss: {avg_loss_after_epoch:.4f}"
        logger.info(train_loss_str)
        print(train_loss_str)


        ## Validation
        with torch.no_grad():
            model.eval()
            epoch_valid_loss = []
    
            prog_bar = tqdm(enumerate(validation_indexes.reshape(-1, batch_size)), total=num_valid_batches)
            for batch_num, batch_indexes in prog_bar:
                # construct batch of data manually
                batch_data, label_data = construct_batch(
                    batch_indexes=batch_indexes, 
                    dumps_path=dumps_path,
                    device=device
                )
                
                # send data to device
                batch_data, label_data = batch_data.to(device), label_data.to(device)
    
                # make prediction
                pred = model.forward(batch_data)
    
                # compute loss
                loss_value = loss_fn(pred, label_data)
                epoch_valid_loss.append(loss_value)

                # memory save maybe idk
                batch_data = None
                label_data = None
                torch.cuda.empty_cache()
                
                # validation batch logging
                validation_str = f'Epoch {epoch+1} validation batch {batch_num+1} completed with loss {loss_value.item():.4f} in {time.time()-start:.2f}s.'
                prog_bar.set_description(validation_str)
            
        avg_vloss_after_epoch = sum(epoch_valid_loss)/len(epoch_valid_loss)
        valid_losses.append(avg_vloss_after_epoch)

        # validation logging
        validation_loss_str = f"Epoch {epoch+1} valid loss value: {avg_vloss_after_epoch:.4f}"
        print(validation_loss_str)
        logger.info(validation_loss_str)

        # checkpointing
        if avg_vloss_after_epoch < best_validation:
            best_validation = avg_vloss_after_epoch
            save_path = os.environ['HOME'] + '/bh/harm2d/' + model.save_path
            model.save(save_path=save_path)

    ## pickle training and validation loss (for external plotting)
    workdir = os.environ['HOME']+'/bh/harm2d/'
    with open(workdir+'train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open(workdir+'valid_losses.pkl', 'wb') as f:
        pickle.dump(valid_losses, f)


# 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 
def cleanup():
    dist.destroy_process_group()

## main training function for multi GPU training
def main_worker(rank, world_size, model_path: str = None):
    global notebook, axisym,set_cart,axisym,REF_1,REF_2,REF_3,set_cart,D,print_fieldlines
    global lowres1,lowres2,lowres3, RAD_M1, RESISTIVE, export_raytracing_GRTRANS, export_raytracing_RAZIEH,r1,r2,r3
    global r_min, r_max, theta_min, theta_max, phi_min,phi_max, do_griddata, do_box, check_files, kerr_schild

    # setup environment
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # if main GPU, init logging
    if rank == 0:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename='training.log',
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    # load configs
    with open('train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # read in config variables
    num_dumps = config['num_dumps']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']

    # change to dumps location for data reading
    dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'
    os.chdir(dumps_path)
    

    ## setup model
    model = B3_CNN().to(device)

    # bring in model weights if model_path is provided
    if model_path is not None:
        model_dict_path = model_path
        model_dict = torch.load(model_dict_path)
        model.load_state_dict(model_dict)
        if rank == 0:
            model_weights_info_str = f"Loaded weights from: {model_path}"
            logger.info(model_weights_info_str)
            print(model_weights_info_str)
    else:
        if rank == 0:
            model_weights_info_str = f"Randomly initializing weights."
            logger.info(model_weights_info_str)
            print(model_weights_info_str)

    # get best validation from model, initially float('inf') for new model
    best_val_loss = model.best_val_seen
    
    if rank == 0:
        # summarize model 
        summary_str = summary(model, input_size=(batch_size, 8, 224, 48, 96))
        # model summary
        model_summary_str = '\n'+str(summary_str)
        logger.info(model_summary_str)
        print(model_summary_str)

        # training parameters
        training_hyperparams_str = f'''
        Training on dumps {start_dump} - {end_dump} 
            number of epochs: {num_epochs}
            batch size: {batch_size}
            logging device: {device}
        
        '''
        logger.info(training_hyperparams_str)
        print(training_hyperparams_str)

    # get indexes for training data
    train_idxs, valid_idxs = custom_batcher(
        batch_size=batch_size,
        num_dumps=num_dumps,
        split = 0.8,
        seed=1,
        start=start_dump,
        end=end_dump,
    )
    
    # distribute model to GPU devices
    model = DDP(model, device_ids=[rank])
    
    # loss and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # distributed sampler to shared data across GPUs 
    train_sampler = DistributedSampler(train_idxs, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_idxs, num_replicas=world_size, rank=rank, shuffle=False)

    # read in grid data for dumps
    rblock_new_ml()
    
    # loss tracking
    train_losses, valid_losses = [], []

    ## training
    for epoch in range(num_epochs):
        train_start_time = time.time()
        model.train()

        train_sampler.set_epoch(epoch)
        # train loss tracking
        epoch_train_loss = []
        # track the training batch number
        train_batch_num = 1

        # batch training
        train_batches = torch.utils.data.DataLoader(train_idxs, batch_size=batch_size, sampler=train_sampler)
        prog_bar = tqdm(train_batches, disable=rank != 0)
        for batch_indexes in prog_bar:
            start = time.time()

            # construct batch of data manually
            batch_data, label_data = construct_batch(
                batch_indexes=batch_indexes, 
                dumps_path=dumps_path,
            )
            # send data to device
            batch_data, label_data = batch_data.to(device), label_data.to(device)
            
            # zero gradients
            optimizer.zero_grad()
            # compute prediction
            pred = model(batch_data)
            # compute loss
            loss = loss_fn(pred, label_data)
            # backprop and update gradients
            loss.backward()
            optimizer.step()
            # add loss to tracking
            loss_val = loss.item()
            epoch_train_loss.append(loss.item())
            
            # increment batch number
            train_batch_num += 1

            # memory save maybe idk
            batch_data = None
            label_data = None
            torch.cuda.empty_cache()

            # training batch logging
            if rank == 0: 
                batch_str = f'Train loss for epoch {epoch+1}, batch {train_batch_num}: {loss_val:.4f} in {time.time()-start:.2f}s'
                prog_bar.set_description(batch_str)
                logger.info(batch_str)
                # print(batch_str)

        train_loss_avg = sum(epoch_train_loss)/len(epoch_train_loss)
        train_losses.append(train_loss_avg)
        
        if rank == 0:
            train_str = f"Completed train loss for epoch {epoch+1}: {train_loss_avg:.4f} in {time.time()-train_start_time:.2f} s"
            prog_bar.set_description(train_str)
            logger.info(train_str)
            # print(train_str)


        ## validation
        valid_start_time = time.time()
        model.eval()
        # loss tracking
        epoch_valid_loss = []
        # batch number counter
        valid_batch_num = 1

        ## batch validation
        valid_batches = torch.utils.data.DataLoader(valid_idxs, batch_size=batch_size, sampler=valid_sampler)
        prog_bar = tqdm(valid_batches, disable=rank != 0)
        for batch_indexes in prog_bar:
            start = time.time()
            batch_data, label_data = [], []

            # construct batch of data manually
            batch_data, label_data = construct_batch(
                batch_indexes=batch_indexes, 
                dumps_path=dumps_path,
            )
            # send data to device
            batch_data, label_data = batch_data.to(device), label_data.to(device)
            
            # compute prediction
            with torch.no_grad():
                pred = model(batch_data)
            # compute loss
            loss = loss_fn(pred, label_data)
            # log validation loss
            epoch_valid_loss.append(loss.item())
            # increment batch number
            valid_batch_num += 1
            # validation batch logging
            if rank == 0: 
                batch_str = f'Validation loss for epoch {epoch+1}, batch {valid_batch_num}: {loss.item():.4f} in {time.time()-start:.2f}s'
                prog_bar.set_description(batch_str)
                logger.info(batch_str)
                # print(batch_str)

        if rank == 0:
            val_loss_avg = sum(epoch_valid_loss)/len(epoch_valid_loss)

            valid_str = f"Completed validation loss for epoch {epoch+1}: {val_loss_avg:.4f} in {time.time()-valid_start_time:.2f} s"
            prog_bar.set_description(train_str)
            logger.info(valid_str)
            print(valid_str)

            valid_losses.append(val_loss_avg)

            # save best model on rank 0
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                model.best_val_seen = best_val_loss # have model track best val for tracking
                model_save_path = os.environ['HOME'] + '/bh/harm2d/' + model.module.save_path
                model_save_info = f'Model saved at: {model_save_path}'
                model.module.save(model_save_path)
                logger.info(model_save_info)
                print(model_save_info)

    # Save training stats
    if rank == 0:
        with open(os.environ['HOME']+'/bh/harm2d/train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f)
        with open(os.environ['HOME']+'/bh/harm2d/valid_losses.pkl', 'wb') as f:
            pickle.dump(valid_losses, f)

    cleanup()

# 
global do_save
do_save = 1

# plot and save range of dumps between start and end, save to save_path
def plot_and_save_range(start: int, end: int, save_path: str):
    global notebook, axisym,set_cart,axisym,REF_1,REF_2,REF_3,set_cart,D,print_fieldlines
    global lowres1,lowres2,lowres3, RAD_M1, RESISTIVE, export_raytracing_GRTRANS, export_raytracing_RAZIEH,r1,r2,r3
    global r_min, r_max, theta_min, theta_max, phi_min,phi_max, do_griddata, do_box, check_files, kerr_schild

    # path to dumps
    dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'
    os.chdir(dumps_path)
    
    # rewrite for performance
    rblock_new_ml()

    indexes = np.arange(start=start, stop=end)
    for index in indexes:
        read_time_start = time.time()
        # get dumps and grid data
        rpar_new(index)
        rgdump_griddata(dumps_path)
        rdump_griddata(dumps_path, index)
        print(f'Read in dump {index} in {time.time()-read_time_start:.4f} s')

        plot_time_start = time.time()
        # plot and save
        plc_cart(
            var=(rho), 
            min=-2,
            max=2, 
            rmax=100, 
            offset=0, 
            name=save_path+f'rho_{index}', 
            label=r"$\sigma r {\rm sin}\theta$ at %d $r_g/c$" % t
        )
        print(f'Plotted and saved in {time.time()-plot_time_start:.4f} s')

## plc cart edit for msai project
def plc_cart_ml(var, min, max, rmax, offset, name, label):
    global aphi, r, h, ph, print_fieldlines,notebook, do_box, do_save
    fig = plt.figure(figsize=(64, 32))

    X = r*np.sin(h)
    Y = r*np.cos(h)
    if(nb==1 and do_box==0):
        X[:,:,0]=0.0*X[:,:,0]
        X[:,:,bs2new-1]=0.0*X[:,:,bs2new-1]

    plotmax = int(20*rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > np.sqrt(2)*plotmax:
            ilim = i
            break

    levels_ch = np.linspace(min, max, 300)
    #levels_ch = np.arange(min, max, (max-min)/300.0)

    # full left figure
    plt.subplot(1, 2, 1)
    plc_new((var)[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
    res = plc_new((var)[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    if (print_fieldlines == 1):
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=-1.0 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$", fontsize=90)
    plt.ylabel(r"$z / R_g$", fontsize=90)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=plt.colorbar(res, cax=cax)
    #cb.ax.tick_params(labelsize=50)

    # zoomed right figure
    factor = 20
    plt.subplot(1, 2, 2)
    plc_new((var)[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * factor, ymax=rmax * factor)
    res = plc_new((var)[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * factor, ymax=rmax * factor)
    if (print_fieldlines == 1):
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * factor, ymax=rmax * factor)
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=-1.0 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * factor, ymax=rmax * factor)

    plt.xlabel(r"$x / R_g$", fontsize=90)
    #plt.ylabel(r"$z / R_g$", fontsize=60)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=plt.colorbar(res, cax=cax)
    #cb.ax.tick_params(labelsize=50)
    plt.tight_layout()
    if (do_save==1):
        plt.savefig(name, dpi=100)
    # if (notebook==0):
    # NOTE always close
    plt.close('all')


 
    # dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'
    # os.chdir(dumps_path)


    # set_mpi(0)
    # import pp_c

    # path_to_check = os.environ['HOME']+'/bh/harm2d/models/cnn/saves/b3_v0.1.0.pth'
    # if os.path.exists(path_to_check):
    #     model_path = path_to_check
        
    # # otherwise no model, random init
    # else:
    #     model_path = None
        
    # model_path = None
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train(model_path=model_path, device=device)


    # world_size = torch.cuda.device_count()
    # if world_size > 1:
    #     print(f"Starting distributed training on {world_size} GPUs...")
    #     mp.spawn(main_worker, args=(world_size, model_path,), nprocs=world_size, join=True)
    # else:
    #     print(f"Starting single GPU training...")
    #     train()
    
