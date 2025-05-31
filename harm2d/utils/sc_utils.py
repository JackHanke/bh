import numpy as np
import torch
import os

# returns random selection of dump incides
def custom_batcher(
        batch_size: int, 
        num_dumps: int, 
        split: float = 0.8, 
        seed: int = None,
        start: int = None,
        end: int = None,
    ):
    # randomize what data is trained on
    if seed is not None: np.random.seed(seed)
    # randomize data
    if start is None and end is None:
        indexes = np.arange(num_dumps) # 0 to num_dumps
    else:
        # if only training on some portion of dumps, use line below:
        indexes = np.arange(start=start, stop=end) # start to end
    
    np.random.shuffle(indexes)
    # get split
    split_idx = round(len(indexes)*(split))
    # split indexes and return
    train_indexes = indexes[:split_idx]
    validation_indexes = indexes[split_idx:]
    return train_indexes, validation_indexes

# turn relevant global variables into single tensor
# dim for reduced data is (batch_size=1, channels=8, depth=224, width=48, height=96)
def tensorize_globals(rho: np.array, ug: np.array, uu: np.array, B: np.array):
    rho_tensor = torch.tensor(rho)[0].unsqueeze(0)
    ug_tensor = torch.tensor(ug)[0].unsqueeze(0)
    uu_tensor = torch.tensor(uu[1:4]).squeeze(1)
    B_tensor = torch.tensor(B[1:4]).squeeze(1)
    # tensorize
    data_tensor = torch.cat((rho_tensor, ug_tensor, uu_tensor, B_tensor), dim=0)
    return data_tensor.unsqueeze(0)


# rewrite of rblock_new for grid read performance
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

    AMR_ACTIVE = 0
    AMR_LEVEL = 1
    AMR_REFINED = 2
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    AMR_PARENT = 6
    AMR_CHILD1 = 7
    AMR_CHILD2 = 8
    AMR_CHILD3 = 9
    AMR_CHILD4 = 10
    AMR_CHILD5 = 11
    AMR_CHILD6 = 12
    AMR_CHILD7 = 13
    AMR_CHILD8 = 14
    AMR_NBR1 = 15
    AMR_NBR2 = 16
    AMR_NBR3 = 17
    AMR_NBR4 = 18
    AMR_NBR5 = 19
    AMR_NBR6 = 20
    AMR_NODE = 21
    AMR_POLE = 22
    AMR_GROUP = 23
    AMR_CORN1 = 24
    AMR_CORN2 = 25
    AMR_CORN3 = 26
    AMR_CORN4 = 27
    AMR_CORN5 = 28
    AMR_CORN6 = 29
    AMR_CORN7 = 30
    AMR_CORN8 = 31
    AMR_CORN9 = 32
    AMR_CORN10 = 33
    AMR_CORN11 = 34
    AMR_CORN12 = 35
    AMR_LEVEL1=  110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112  
    AMR_NBR1_3=113
    AMR_NBR1_4=114
    AMR_NBR1_7=115
    AMR_NBR1_8=116
    AMR_NBR2_1=117
    AMR_NBR2_2=118
    AMR_NBR2_3=119
    AMR_NBR2_4=120
    AMR_NBR3_1=121
    AMR_NBR3_2=122
    AMR_NBR3_5=123
    AMR_NBR3_6=124
    AMR_NBR4_5=125
    AMR_NBR4_6=126
    AMR_NBR4_7=127
    AMR_NBR4_8=128
    AMR_NBR5_1=129
    AMR_NBR5_3=130
    AMR_NBR5_5=131
    AMR_NBR5_7=132
    AMR_NBR6_2=133
    AMR_NBR6_4=134
    AMR_NBR6_6=135
    AMR_NBR6_8=136
    AMR_NBR1P=161
    AMR_NBR2P=162
    AMR_NBR3P=163
    AMR_NBR4P=164
    AMR_NBR5P=165
    AMR_NBR6P=166
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


# original training script before multi GPU was written

# training script
def train(model: torch.nn.module, model_path: str = None):
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
    # model = JACK_CNN_3D().to(device)
    model = CNN_DEPTH().to(device)
    
    # model = CNN_DEPTH().to(device)
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
            ## fetch and tensorize data
            # NOTE everything is a global variable so it has to be this way. im sorry
            batch_data, label_data = [], []
            # batch_idx is the dump number
            for batch_idx in batch_indexes:
                start = time.time()

                # at every batch of size batch_size, we need to read in 2 * batch_size dumps
                
                ## get data frame
                # get data into global context NOTE this is really slow
                # rblock_new(batch_idx)
                rpar_new(batch_idx)
                # get grid data
                rgdump_griddata(dumps_path)
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

            # memory save maybe idk
            batch_data = None
            label_data = None

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
            ## fetch and tensorize data
            # NOTE everything is a global variable so it has to be this way. im sorry
            batch_data, label_data = [], []
            # batch_idx is the dump number
            for batch_idx in batch_indexes:
                start = time.time()
                ## get data frame
                # get data into global context
                rpar_new(batch_idx)
                rgdump_griddata(dumps_path)
                rdump_griddata(dumps_path, batch_idx)
                # format data as tensor
                data_tensor = tensorize_globals(rho=rho, ug=ug, uu=uu, B=B)
                # add to batch
                batch_data.append(data_tensor)

                ## get label frame
                # get data into global context
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
