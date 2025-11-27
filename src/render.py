import os
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.standardize_inputs import standardize, destandardize

import torch

with open('config.yaml', 'r') as file: config = yaml.safe_load(file)

# 
def plot_frame(
        data_path: str,
        variable_name: str, 
        dump_number: int,
        save_path: str = None
    ):

    variable_axis_dictionary = {
        'log(rho)': 0,
        'ug': 1,
        'uu_x': 2,
        'uu_y': 3,
        'uu_z': 4,
        'B_x': 5,
        'B_y': 6,
        'B_z': 7,
    }

    var_idx = variable_axis_dictionary[variable_name]

    with h5py.File(data_path, "r") as f:
        # get the axis the specific dump index is stored at on disk
        idx = dump_number - f['dump_index'][0]
        # get data
        var = f['data'][idx][0][var_idx, :, :]
    var = np.expand_dims(var, axis=0)
    
    _plc_cart(
        var = var,
        min = -2, 
        max = 2, 
        rmax = 100, 
        offset = 0, 
        name = f"{variable_name}_{dump_number}", 
        label = f"Frame {dump_number}"
    )

# view frame and forward pass of autoencoder
def view_reconstruction(data_path:str, model_path:str, variable_name: str, dump_number: int):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    input_shape = (8,224,48,96)
    
    variable_axis_dictionary = {
        'log(rho)': 0,
        'ug': 1,
        'uu_x': 2,
        'uu_y': 3,
        'uu_z': 4,
        'B_x': 5,
        'B_y': 6,
        'B_z': 7,
    }
    var_idx = variable_axis_dictionary[variable_name]
    
    with h5py.File(data_path, "r") as f:
        # get the axis the specific dump index is stored at on disk
        idx = dump_number - f['dump_index'][0]
        # get data
        var = f['data'][idx][0][var_idx, :, :]
    # var = np.expand_dims(var, axis=0)

    # res, ax = _preprocess_var_for_plotting(var)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(var[:,:,0])

    # 
    avg_save_path = 'src/'+config['avg_save_path']
    variance_save_path = 'src/'+config['variance_save_path']
    avg_array = torch.load(avg_save_path)
    variance_array = torch.load(variance_save_path)
    # load model
    model = torch.load(model_path)

    # preprocess
    standardized_var = standardize(var, avg_array, variance_array).to(DEVICE)

    # inference
    prediction = model(standardized_var)
    
    # postprocess
    postprocessed_prediction = destandardize(prediction, avg_array, variance_array)

    ax2.imshow(postprocessed_prediction)

    plt.show()
    

    
    

# view predictions over time in latent space
def view_latent_predictions():
    pass

# transform raw data to viewable data
def _preprocess_var_for_plotting(var, xcoord=None, ycoord=None, ax=None):
    r = np.load('utils/'+config['r_path'])
    h = np.load('utils/'+config['h_path'])
    ph = np.load('utils/'+config['ph_path'])
    block = np.load('utils/'+config['block_path'])
    n_ord = np.load('utils/'+config['n_ord_path'])
    rmax = 100
    offset = 0
    nb, nb2d, nb1, nb2, nb3 = 1,1,1,1,1
    AMR_COORD3 = 5
    AMR_LEVEL3 = 112
    REF_3 = 1
    cb = True
    xy = True
    notebook = True
    print_fieldlines = False
    do_box = False
    do_save = False
    
    plotmax = int(20*rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > np.sqrt(2)*plotmax:
            ilim = i
            break
            
    myvar = var[:, 0:ilim]
    _, bs1new, bs2new, bs3new = myvar.shape
    
    X = r*np.sin(h)
    Y = r*np.cos(h)
    if(nb==1 and do_box==0):
        X[:,:,0]=0.0*X[:,:,0]
        X[:,:,bs2new-1]=0.0*X[:,:,bs2new-1]
        
    l = [None] * nb2d

    if (np.min(myvar) == np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = False
    nc = 15
    k = 0
    mirrory = 0
    # cmap = kwargs.pop('cmap',cm.jet)
    isfilled = False
    xy = 0
    xmax = 10
    ymax = 5
    z = 0

    min = -2
    max = 2
    levels_ch = np.linspace(min, max, 300)
    levels=levels_ch
    nc=100
    cb=0
    isfilled=1
    xcoord=X[:, 0:ilim]
    ycoord=Y[:, 0:ilim]
    xy=1
    z=offset
    factor = 20
    xmax=rmax * factor
    ymax=rmax * factor

    if ax is None:
        ax = plt.gca()
    if isfilled:
        for i in range(0, nb):
            index_z_block=int((z-int((z/360))*360.0)/360.0*bs3new*nb3*(1+REF_3)**(block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block/bs3new)):
                offset=index_z_block-block[n_ord[i], AMR_COORD3]*bs3new
                res = ax.contourf(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, extend='both')
    else:
        for i in range(0, nb):
            index_z_block=int(z/360.0*bs3new*nb3*(1+REF_3)**(block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block/bs3new)):
                offset=index_z_block-block[n_ord[i], AMR_COORD3]*bs3new
                res = ax.contour(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, linewidths=4, extend='both')
    plt.colorbar(res, ax=ax)
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)
    return res, ax
    

# helper function _plc_cart from harm codebase
def _plc_cart(
        var, 
        min, 
        max, 
        rmax, 
        offset, 
        name, 
        label,
        notebook: bool = True,
        print_fieldlines: bool = False,
        do_box: bool = False,
        do_save: bool = False,
    ):

    # bring in relevant globals
    r = np.load('utils/'+config['r_path'])
    h = np.load('utils/'+config['h_path'])
    ph = np.load('utils/'+config['ph_path'])
    _, bs1new, bs2new, bs3new = var.shape
    nb = 1
                 
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
    plt.subplot(1, 2, 1)
    _plc_new(var[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
    res = _plc_new(var[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(res, cax=cax)
    #cb.ax.tick_params(labelsize=50)
    
    factor = 20
    plt.subplot(1, 2, 2)
    _plc_new(var[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * factor, ymax=rmax * factor)
    res = _plc_new(var[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * factor, ymax=rmax * factor)

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
    if (notebook==0):
        plt.close('all')

def _plc_new(myvar, xcoord=None, ycoord=None, ax=None, **kwargs):
    # global r, h, ph

    ## NOTE hardcoded to avoid globals
    r = np.load('utils/'+config['r_path'])
    h = np.load('utils/'+config['h_path'])
    ph = np.load('utils/'+config['ph_path'])
    block = np.load('utils/'+config['block_path'])
    n_ord = np.load('utils/'+config['n_ord_path'])
    nb, nb2d, nb1, nb2, nb3 = 1,1,1,1,1
    _, bs1new, bs2new, bs3new = myvar.shape
    AMR_COORD3 = 5
    AMR_LEVEL3 = 112
    REF_3 = 1
    cb = True
    xy = True
        
    l = [None] * nb2d

    if (np.min(myvar) == np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k', 0)
    mirrory = kwargs.pop('mirrory', 0)
    # cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled', False)
    xy = kwargs.pop('xy', 0)
    xmax = kwargs.pop('xmax', 10)
    ymax = kwargs.pop('ymax', 5)
    z = kwargs.pop('z', 0)

    if ax is None:
        ax = plt.gca()
    if isfilled:
        for i in range(0, nb):
            index_z_block=int((z-int((z/360))*360.0)/360.0*bs3new*nb3*(1+REF_3)**(block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block/bs3new)):
                offset=index_z_block-block[n_ord[i], AMR_COORD3]*bs3new
                res = ax.contourf(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, extend='both',**kwargs)
    else:
        for i in range(0, nb):
            index_z_block=int(z/360.0*bs3new*nb3*(1+REF_3)**(block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block/bs3new)):
                offset=index_z_block-block[n_ord[i], AMR_COORD3]*bs3new
                res = ax.contour(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, linewidths=4, extend='both', **kwargs)
    if (cb == True):  # use color bar
        plt.colorbar(res, ax=ax)
    if xy:
        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)
    return res
