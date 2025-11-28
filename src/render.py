import os
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.standardize_inputs import standardize, destandardize

import torch

HOME_DIR = os.getenv('HOME')
with open(f'{HOME_DIR}/bh/config.yaml', 'r') as file: config = yaml.safe_load(file)

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

    print(f'Max value: {np.max(var)} Min value: {np.min(var)}')
    
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
def view_reconstruction(data_path:str, model, variable_name: str, dump_number: int, device):
    
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
    # add batch index
    var = np.expand_dims(var, axis=0)

    # res, ax = _preprocess_var_for_plotting(var)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # 
    avg_save_path = 'src/'+config['avg_save_path']
    variance_save_path = 'src/'+config['variance_save_path']
    avg_array = torch.load(avg_save_path).to(device)
    variance_array = torch.load(variance_save_path).to(device)
    
    var_tensor = torch.from_numpy(var.astype(np.float32)).to(device)

    # preprocess
    standardized_var = standardize(var_tensor, avg_array, variance_array, device=device)
    # inference
    prediction, _, _ = model(standardized_var)
    # postprocess
    postprocessed_prediction = destandardize(prediction, avg_array, variance_array, device=device)
    
    ## plotting
    cmap = 'jet'
    xmax, ymax = 50, 50
    # true data
    x,y,z = _preprocess_var_for_plotting(var, ax=ax1, side='left')
    res = ax1.contourf(x, y, z, 100, extend='both', cmap=cmap)
    x,y,z = _preprocess_var_for_plotting(var, ax=ax1, side='right')
    res = ax1.contourf(x, y, z, 100, extend='both', cmap=cmap)
    ax1.set_xlim(-xmax, xmax)
    ax1.set_ylim(-ymax, ymax)
    ax1.set_title(f'Dump {dump_number}')
    ax1.set_aspect('equal', adjustable='box')
    # mesh1 = ax1.pcolormesh(x, y, z, shading='auto', cmap=cmap)
    # cbar1 = fig.colorbar(mesh1, ax=ax1)

    # reconstruction 
    x,y,z = _preprocess_var_for_plotting(standardized_var.cpu().detach().numpy()[:,var_idx], ax=ax2, side='left')
    res = ax2.contourf(x, y, z, 100, extend='both', cmap=cmap)
    x,y,z = _preprocess_var_for_plotting(standardized_var.cpu().detach().numpy()[:,var_idx], ax=ax2, side='right')
    res = ax2.contourf(x, y, z, 100, extend='both', cmap=cmap)
    ax2.set_xlim(-xmax, xmax)
    ax2.set_ylim(-ymax, ymax)
    ax2.set_title(f'Recon')
    ax2.set_aspect('equal', adjustable='box')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    # mesh2 = ax1.pcolormesh(x, y, z, shading='auto', cmap=cmap)
    # cbar2 = fig.colorbar(mesh2, ax=ax2)

    plt.tight_layout()
    plt.show()

    # TODO save fig
    
    

# view predictions over time in latent space
def view_latent_predictions():
    pass

# transform raw data to viewable data
def _preprocess_var_for_plotting(var, ax, side='right'):
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
    z = 0
    if side == 'left':
        X = -1.0 * X
        z = 180 + z
        
    
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
                # res = ax.contourf(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, extend='both')
    else:
        for i in range(0, nb):
            index_z_block=int(z/360.0*bs3new*nb3*(1+REF_3)**(block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block/bs3new)):
                offset=index_z_block-block[n_ord[i], AMR_COORD3]*bs3new
                # res = ax.contour(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, linewidths=4, extend='both')
    
    return xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset]
    

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
