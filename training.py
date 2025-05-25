# system imports
import os
import sys
import subprocess
import logging

# training imports
import torch
import numpy as np
from tqdm import tqdm

# training utilities
from utils.sc_utils import custom_batcher, tensorize_globals
from models.cnn.cnn import CNN_3D

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='training.log', 
    filemode='w', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# logger = logging.getLogger(__name__)

# 
def run_build_command(script_name: str, command_args: list):
    """
    Executes a Python script with the given command-line arguments using subprocess.

    Args:
        script_name (str): The name of the Python script to run (e.g., 'setup.py').
        command_args (list): A list of arguments to pass to the script (e.g., ['build_ext', '--inplace']).
    """
    print(f"--- Running '{script_name} {' '.join(command_args)}' ---")
    try:
        # Construct the command: python <script_name> <arg1> <arg2> ...
        # sys.executable ensures we use the same python interpreter that's running this script
        command = [sys.executable, script_name] + command_args
        
        # Run the command. capture_output=False means output goes directly to console.
        # check=True raises CalledProcessError if the command returns a non-zero exit code.
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"--- Successfully ran '{script_name}' ---")
    except FileNotFoundError:
        print(f"Error: The script '{script_name}' was not found. Make sure it's in the current directory or accessible via PATH.")
        sys.exit(1) # Exit with an error code
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}.")
        # If capture_output was True, you'd print e.stdout and e.stderr here.
        sys.exit(1) # Exit with an error code
    except Exception as e:
        print(f"An unexpected error occurred while running '{script_name}': {e}")
        sys.exit(1) # Exit with an error code

def setup():
    # global variables
    global notebook
    global axisym,set_cart,axisym,REF_1,REF_2,REF_3,set_cart,D,print_fieldlines
    global lowres1,lowres2,lowres3, RAD_M1, RESISTIVE, export_raytracing_GRTRANS, export_raytracing_RAZIEH,r1,r2,r3
    global r_min, r_max, theta_min, theta_max, phi_min,phi_max, do_griddata, do_box, check_files, kerr_schild
    global set_mpi

    """
    Main function to execute the build commands.
    """
    # change to home
    harm_directory = os.environ['HOME'] + '/bh/harm2d'
    os.chdir(harm_directory)

    # Define the arguments for the build command
    build_ext_args = ['build_ext', '--inplace']

    # Ensure the scripts exist before attempting to run them
    if not os.path.exists('setup.py'):
        print("Error: 'setup.py' not found in the current directory. Cannot proceed.")
        sys.exit(1)
    if not os.path.exists('pp.py'):
        print("Error: 'pp.py' not found in the current directory. Cannot proceed.")
        sys.exit(1)

    # Run the setup.py command
    run_build_command('setup.py', build_ext_args)

    # Run the pp.py command
    run_build_command('pp.py', build_ext_args)

    # set params
    lowres1 = 1 # 
    lowres2 = 1 # 
    lowres3 = 1 # 
    r_min, r_max = 1.0, 100.0
    theta_min, theta_max = 0.0, 9
    phi_min, phi_max = -1, 9
    do_box=0
    set_cart=0
    # set_mpi(0)
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

    print('Imports and setup done.')

    print("\nAll build commands completed successfully.")

# training script
def train():
    # path to dumps
    dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'
    os.chdir(dumps_path)

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

    rgdump_griddata(dumps_path)
    with open("gdumps/grid", "rb") as fin:
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
        gd = np.fromfile(fin, dtype=np.int32, count=NV * nmax, sep='')
        gd = gd.reshape((NV, nmax), order='F').T
        start = time.time()
        block[:,0:NV] = gd
        if(NV<170):
            block[:, AMR_LEVEL1] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL2] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL3] = gd[:, AMR_LEVEL]

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


if __name__ == "__main__":
    # setup environment
    setup()

    # train model
    train()

    # 


