# Black Hole Emulation

![](./assets/bh.jpg)

## Project Guide

Our work for training on the Perlmutter supercomputer can be found within `harmd2d`:

`harmd2d/`

├── `models` includes model definitions for feed forward neural networks (`ffnn`), convolutional nerual networks (`cnn`), and UNet style encoders (`unet`)

├── `pp.py` adds thre training scripts for these models, including external logging and multi-GPU support. Our current implementation needs to be in this file because the variables are stored as globals within the context, and it is a work-in-progress to isolate these processes.

├── `async_read.py` is a work-in-progress implementation for asynchronous data reading to improve 

For running a trial training run, run the following

```bash
module load conda
conda env create -f scenv.yaml
conda activate scenv
python pp.py
```

For launching a true training sessions with no hangup training, run

```bash
./trainsh
```


## Developer Notes

- [Perlmutter Docs](https://docs.nersc.gov/getting-started/)
    - To login: `ssh user@saul.nersc.gov`, then password + 6 digit Google Authenticator code
- Real 3 dimensional data can be found at: `'/pscratch/sd/l/lalakos/ml_data_rc300/reduced'`
- The [BitBucket link](https://bitbucket.org/atchekho/harm2d/src/master/)=

## Setup

To setup environment on Perlmutter, run the following:
```bash
module load conda
conda env create -f scenv.yaml
conda activate scenv
python -m ipykernel install --user --name scenv --display-name scenvkernel
```

Then refresh your browser window, and then click on the `scenvkernel` kernel to run the `sc_workspace.ipynb`.

For `ffmpeg` rendering on Perlmutter, follow [this blogpost](https://xiaocanli.github.io/blog/2023/ffmpeg-perlmutter/https://xiaocanli.github.io/blog/2023/ffmpeg-perlmutter/)

