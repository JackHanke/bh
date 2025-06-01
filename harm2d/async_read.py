## async_read.py is a minimal pp.py clone rewritten for read performance of dumps
## cli command > python async_read.py build_ext --inplace


# system imports
import os
import sys
import subprocess
import logging
import time
import pickle
import yaml
import asyncio

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

## setup package and compile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "pp_c", 
            sources=["pp_c.pyx", "functions.c"], 
            include_dirs=[np.get_include()], 
            extra_compile_args=["-fopenmp"], 
            extra_link_args=["-O2 -fopenmp"]
        )
    ]
)


# rblock_new_ml() rewrite
def get_grid_data(dumps_path: str):
    # replacement of global variable settings
    AMR_ACTIVE, AMR_LEVEL, AMR_REFINED = 0,1,2
    AMR_LEVEL1, AMR_LEVEL2, AMR_LEVEL3 = 110,111,112

    grid_path = dumps_path + "/gdumps/grid"

    if(os.path.isfile(grid_path)):
        fin = open(grid_path, "rb")
        size = os.path.getsize(grid_path)
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4

    else:
        print("Cannot find grid file!")
        return

    with open(grid_path, "rb") as fin:
        size = os.path.getsize(grid_path)
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

    return block, nmax, n_ord

# rdump_griddata rewrite for performance
async def read_dump_from_disk(
        dump_dir: str, 
        dump: int,
        block: np.array,
        n_ord: np.array,
    ):
    import pp_c

    ## hardcode unchaging global variables
    # set_mpi 
    rank = 0
    # 
    AMR_ACTIVE, AMR_LEVEL, AMR_REFINED = 0,1,2
    AMR_LEVEL1, AMR_LEVEL2, AMR_LEVEL3 = 110,111,112
    # others
    r_min, r_max = 1.0, 100.0
    theta_min, theta_max = 0.0, 9
    phi_min, phi_max = -1, 9
    do_box=0
    set_cart=0
    axisym=1
    print_fieldlines=0
    export_raytracing_GRTRANS=0
    export_raytracing_RAZIEH=0
    kerr_schild=0
    DISK_THICKNESS=0.03
    check_files=1
    interpolate_var=0
    AMR = 0 # get all data in grid
    
    gridsizex1 = 224
    gridsizex2 = 48
    gridsizex3 = 96
    mytype = np.float32
    flag = 0
    interpolate_var = 0
    RAD_M1 = np.int32(0)
    RESISTIVE = np.int32(0)
    TWO_T = 0
    P_NUM = 0
    n_active_total = 456
    lowres1 = 1
    lowres2 = 1
    lowres3 = 1
    nb = 1
    bs1 = 14
    bs2 = 12
    bs3 = 12
    axisym = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1
    REF_1 = 1
    REF_2 = 1
    REF_3 = 1

    a = 0.94
    gam = 1.6666666666666667
    startx1 = 0.09509474077300727
    startx2 = -0.9791666666666666
    startx3 = 0.0
    _dx1 = 0.06125185632674673
    _dx2 = 0.04079861111111111
    _dx3 = 0.06544984694978735
    i_min = 0
    i_max = 224
    j_min = 0
    j_max = 48
    z_min = 0
    z_max = 96

    # NOTE this may cause problems, these are non-zero after c functions are called
    gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
    gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
    x1 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    x2 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    x3 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    r = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')


    # Allocate memory
    rho = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    ug = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    uu = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    B = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if(export_raytracing_RAZIEH):
        Rdot = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        Rdot = np.zeros((1, 1, 1, 1), dtype=mytype, order='C')
    bsq = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if(RAD_M1):
        E_rad = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        uu_rad = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        E_rad=np.copy(ug)
        uu_rad=np.copy(uu)

    if (RESISTIVE):
        E = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        E = B

    if (TWO_T):
        TE = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        TI = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        TE = rho
        TI = rho

    if (P_NUM):
        photon_number = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        photon_number = rho
        
    if (os.path.isfile("dumps%d/new_dump" % dump)):
        flag = 1
    else:
        flag = 0

    pp_c.rdump_griddata(
        flag, 
        interpolate_var, 
        np.int32(RAD_M1),
        np.int32(RESISTIVE), 
        TWO_T, 
        P_NUM, 
        dump_dir, 
        dump, 
        n_active_total, 
        lowres1, 
        lowres2, 
        lowres3, 
        nb,
        bs1,
        bs2,
        bs3, 
        rho,
        ug, 
        uu, 
        B, 
        E, 
        E_rad, 
        uu_rad, 
        TE, 
        TI, 
        photon_number, 
        gcov,
        gcon,
        axisym,
        n_ord,
        block, 
        nb1,
        nb2,
        nb3,
        REF_1, 
        REF_2,
        REF_3, 
        np.max(block[n_ord, AMR_LEVEL1]),
        np.max(block[n_ord, AMR_LEVEL2]), 
        np.max(block[n_ord, AMR_LEVEL3]),
        export_raytracing_RAZIEH, 
        DISK_THICKNESS, 
        a, 
        gam, 
        Rdot, 
        bsq, 
        r, 
        startx1,
        startx2,
        startx3,
        _dx1,
        _dx2,
        _dx3,
        x1,
        x2,
        x3, 
        i_min, 
        i_max, 
        j_min, 
        j_max, 
        z_min, 
        z_max,
    )

    bs1new = gridsizex1
    bs2new = gridsizex2
    bs3new = gridsizex3

    if (do_box == 1):
        startx1 = startx1 + (i_min) * _dx1
        startx2 = startx2 + (j_min) * _dx2
        startx3 = startx3 + (z_min) * _dx3

    nb2d = nb
    nb = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1

    return rho, ug, uu, B

# read data from disk for data and label batch, transform and tensorize
async def make_batch(
        indexes: list[int], 
        dumps_path: str,
        block: np.array,
        n_ord: np.array,
    ):
    batch_size = len(indexes)
    batch_and_label_indexes = indexes + [i+1 for i in indexes]
    
    tasks = [
        read_dump_from_disk(
            dump_dir=dumps_path, 
            dump=idx, 
            block=block, 
            n_ord=n_ord
        ) for idx in batch_and_label_indexes
    ]
    results = await asyncio.gather(*tasks)
    
    batch_tensor, label_tensor = [], []
    for result_num, (rho, ug, uu, B) in enumerate(results):
        # transform and stack results
        rho = rho.reshape((1,224,48,96))
        ug = ug.reshape((1,224,48,96))
        uu = uu[1:4].reshape((3,224,48,96))
        B = B[1:4].reshape((3,224,48,96))
        
        # stacked_arr = np.concat((np.log10(rho), np.log10(ug), uu, B), axis=0)
        stacked_arr = np.concat((rho, ug, uu, B), axis=0)
        # NOTE this assumes all files come in in order, idk if this is reasonable
        if result_num < batch_size:
            batch_tensor.append(stacked_arr)
        elif result_num >= batch_size:
            label_tensor.append(stacked_arr)
    
    # tensorize
    batch_tensor = torch.Tensor(np.array(batch_tensor))
    label_tensor = torch.Tensor(np.array(label_tensor))
    
    return batch_tensor, label_tensor

# make data and label batch from batch_indexes
def construct_batch(
        batch_indexes: list[int], 
        dumps_path: str, 
        block: np.array,
        n_ord: np.array,
    ):
    batch_data, label_data = asyncio.run(make_batch(
        indexes=batch_indexes, 
        dumps_path=dumps_path, 
        block=block, 
        n_ord=n_ord,
    ))

    return batch_data, label_data

# training script
def train(device):
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
    # rblock_new_ml()
    # initial grid data read
    block, nmax, n_ord = get_grid_data(dumps_path=dumps_path)

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
            # zero gradients
            optim.zero_grad()
            # construct batch of data manually
            batch_data, label_data = construct_batch(
                batch_indexes=batch_indexes.tolist(), 
                dumps_path=dumps_path,
                block=block,
                n_ord=n_ord,
            )
            # send tensors to device
            batch_data, label_data = batch_data.to(device), label_data.to(device)

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
        model.eval()
        epoch_valid_loss = []

        prog_bar = tqdm(enumerate(validation_indexes.reshape(-1, batch_size)), total=num_valid_batches)
        for batch_num, batch_indexes in prog_bar:
            start = time.time()
            # construct batch of data manually
            batch_data, label_data = construct_batch(
                batch_indexes=batch_indexes.tolist(), 
                dumps_path=dumps_path,
                block=block,
                n_ord=n_ord,
            )

            # make prediction
            pred = model.forward(batch_data)

            # compute loss
            loss_value = loss_fn(pred, label_data)
            epoch_valid_loss.append(loss_value)
            
            # validation batch logging
            validation_str = f'Epoch {epoch+1} validation batch {batch_num+1} completed with loss {loss_value.item():.4f} in {time.time()-start:.2f}s.'
            prog_bar.set_description(validation_str)
            
        avg_vloss_after_epoch = sum(epoch_valid_loss)/len(epoch_valid_loss)
        valid_losses.append(avg_vloss_after_epoch)

        # validation logging
        validation_loss_str = f"Epoch {epoch+1} valid loss value: {avg_loss_after_epoch:.4f}"
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
def distributed_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 
def cleanup():
    dist.destroy_process_group()

## main training function for multi GPU training
def main_worker(rank, world_size, model_path: str = None):
    # setup environment
    distributed_setup(rank, world_size)
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

    # initial grid data read
    block, nmax, n_ord = get_grid_data(dumps_path=dumps_path)
    
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
                batch_indexes=batch_indexes.tolist(), 
                dumps_path=dumps_path,
                device=device,
                block=block,
                n_ord=n_ord,
            )
            # zero gradients
            optimizer.zero_grad()
            # compute prediction
            pred = model.forward(batch_data)
            # compute loss
            loss = loss_fn(pred, label_data)
            # backprop and update gradients
            loss.backward()
            optimizer.step()
            # add loss to tracking
            epoch_train_loss.append(loss.item())
            
            # increment batch number
            train_batch_num += 1

            # training batch logging
            if rank == 0: 
                batch_str = f'Train loss for epoch {epoch+1}, batch {valid_batch_num}: {loss.item():.4f} in {time.time()-start:.2f}s'
                prog_bar.set_description(batch_str)
                logger.info(batch_str)
                print(batch_str)

        train_loss_avg = sum(epoch_train_loss)/len(epoch_train_loss)
        train_losses.append(train_loss_avg)
        if rank == 0:
            train_str = f"Completed train loss for epoch {epoch+1}: {train_loss_avg:.4f} in {time.time()-train_start_time:.2f} s"
            prog_bar.set_description(train_str)
            logger.info(train_str)
            print(train_str)


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
                device=device,
                block=block,
                n_ord=n_ord,
            )
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
                print(batch_str)

        if rank == 0:
            val_loss_avg = sum(epoch_valid_loss)/len(epoch_valid_loss)

            valid_str = f"Completed train loss for epoch {epoch+1}: {val_loss_avg:.4f} in {time.time()-valid_start_time:.2f} s"
            prog_bar.set_description(train_str)
            logger.info(valid_str)
            print(valid_str)

            valid_losses.append(val_loss_avg)

            # save best model on rank 0
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
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


if __name__ == '__main__':
    import pp_c

    dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(device=device)

    # if 'do_test' in sys.argv: do_test = True
    # else: do_test = False
    # # do_test = True
    
    # if do_test:
    #     num_dumps = 10
        
    #     # initial grid data read
    #     block, nmax, n_ord = get_grid_data(dumps_path=dumps_path)
        
    #     start = time.time()
    #     batch_indexes = [i for i in range(1,num_dumps+1)]
    #     batch_data, label_data = construct_batch(batch_indexes=batch_indexes, dumps_path=dumps_path, device=device)
        
    #     print(batch_data.shape)
    #     print(label_data.shape)
        
    #     print(f'async_read.py created tensorized batch of {num_dumps} dumps in: {time.time()-start:.4f}s')
        
    # elif not do_test:
        
    #     print('No testing!')

    # # if saved b3 model, continue training
    # path_to_check = os.environ['HOME']+'/bh/harm2d/models/cnn/saves/b3_v0.1.1.pth'
    # if os.path.exists(path_to_check):
    #     model_path = path_to_check
        
    # # otherwise no model, random init
    # else:
    #     model_path = None

    # world_size = torch.cuda.device_count()
    # if world_size >= 1:
    #     print(f"Starting distributed training on {world_size} GPUs...")
    #     mp.spawn(main_worker, args=(world_size, model_path,), nprocs=world_size, join=True)
    
