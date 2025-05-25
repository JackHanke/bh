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


if __name__ == "__main__":
    # setup environment
    setup()

    # train model
    # train()

    # 


