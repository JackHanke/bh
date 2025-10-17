# Black Hole Emulation

We seek to emulate the black hole simulations created by 

## Project TODOs

- Rewrite `plc_cart` to render frames directly from disk
- Async/ multithreaded/ multiprocess file read for constant batch size read
- Standardize data
- Write custom batcher

## Project Guide

`'/pscratch/sd/l/lalakos/ml_data_rc300/reduced'`

The data is dimension `(8,224,48,96)`, where the the first `8` channels are density, internal energy, the 3 components of velocity, and the 3 components of flux. This works out to `8.26` million FLs, which is naively `66MB` a frame uncompressed.

Selecting the high flux region of the 

## Project Layout and Developer Notes

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

The project layout is as follows.

```
bh/
├── assets/
├── harm2d/
├── harmpi/
├── movies/
├── utils/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

- [Perlmutter Docs](https://docs.nersc.gov/getting-started/)
    - To login: `ssh user@saul.nersc.gov`, then password + 6 digit Google Authenticator code
- Real 3 dimensional data can be found at: `'/pscratch/sd/l/lalakos/ml_data_rc300/reduced'`
- The [BitBucket link](https://bitbucket.org/atchekho/harm2d/src/master/)

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

