# 
def _plc_cart(
        var, 
        min, 
        max, 
        rmax, 
        offset, 
        name, 
        label,
        notebook: bool = False,
        print_fieldlines: bool = False,
        do_box: bool = False,
        do_save: bool = False,
    ):

    global r, h, ph
    global aphi
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

    plt.subplot(1, 2, 1)
    _plc_new(np.log10((var))[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
    res = _plc_new(np.log10((var))[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    if (print_fieldlines == 1):
        _plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
        _plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=-1.0 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
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

    factor = 20
    plt.subplot(1, 2, 2)
    _plc_new(np.log10((var))[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * factor, ymax=rmax * factor)
    res = _plc_new(np.log10((var))[:, 0:ilim], levels=levels_ch, nc=100, cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * factor, ymax=rmax * factor)
    if (print_fieldlines == 1):
        _plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * factor, ymax=rmax * factor)
        _plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=-1.0 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * factor, ymax=rmax * factor)

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
    if (notebook==0):
        plt.close('all')

def _plc_new(myvar, xcoord=None, ycoord=None, ax=None, **kwargs):
    global r, h, ph

    ## NOTE hardcoded to avoid globals
    AMR_COORD3 = 5
    AMR_LEVEL3 = 112
    REF_3 = 1
        
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

def plot_frame(
        variable_name: int, 
        dump_number: int,
        save_path: str
    ):

    variable_axis_dicitonary = {
        'log(rho)': 0,
        'ug': 1,
        'uu_x': 2,
        'uu_y': 3,
        'uu_z': 4,
        'B_x': 5,
        'B_y': 6,
        'B_z': 7,
    }

    _plc_cart(
        var = rho, 
        min = -2, 
        max = 2, 
        rmax = 100, 
        offset = 0, 
        name = f"{variable_name}_{dump_number}", 
        label = f"Frame {dump_number}"
    )

if __name__ == '__main__':
    plot_frame(variable='log(rho)', dump_number=4000)
