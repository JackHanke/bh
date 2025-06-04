# Black Hole Emulation

![](./assets/bh.jpg)

## Project Guide

TODO

## Developer Notes

- [Perlmutter Docs](https://docs.nersc.gov/getting-started/)
    - To login: `ssh user@saul.nersc.gov`, then password + 6 digit Google Authenticator code
- Real 3 dimensional data can be found at: `'/pscratch/sd/l/lalakos/ml_data_rc300/reduced'`
- The [BitBucket link](https://bitbucket.org/atchekho/harm2d/src/master/)=

## Setup

To setup environment, run the following:
```
module load conda
conda env create -f scenv.yaml
conda activate scenv
python -m ipykernel install --user --name scenv --display-name scenvkernel
```

Then refresh your browser window, and then click on the `scenvkernel` kernel to run the `sc_workspace.ipynb`.

For `ffmpeg` rendering on Perlmutter, follow [this blogpost](https://xiaocanli.github.io/blog/2023/ffmpeg-perlmutter/https://xiaocanli.github.io/blog/2023/ffmpeg-perlmutter/)

## Project TODOs

TODO
