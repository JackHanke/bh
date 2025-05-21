from harmpi.harm_script import *

# utils for communicating with harmpi 

# rewrites read_header to get globals it uses into context
def read_header(dump,issilent=True,returnheaderline=False):
    global t,nx,ny,nz,N1,N2,N3,N1G,N2G,N3G,starti,startj,startk,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp, games, startx1, startx2, startx3, x10, x20, tf, NPR, DOKTOT, BL
    global fractheta
    global fracphi
    global rbr
    global npow2
    global cpow2
    #read image
    fin = open( dump, "rb" )
    headerline = fin.readline()
    header = headerline.split()
    nheadertot = len(header)
    fin.close()
    if not dump.startswith("dumps/rdump"):
        if not issilent: print( "dump header: len(header) = %d" % len(header) )
        nheader = 57
        n = 0
        t = myfloat(np.float64(header[n])); n+=1
        #per tile resolution
        N1 = int(header[n]); n+=1
        N2 = int(header[n]); n+=1
        N3 = int(header[n]); n+=1
        #total resolution
        nx = int(header[n]); n+=1
        ny = int(header[n]); n+=1
        nz = int(header[n]); n+=1
        #numbers of ghost cells
        N1G = int(header[n]); n+=1
        N2G = int(header[n]); n+=1
        N3G = int(header[n]); n+=1
        startx1 = myfloat(float(header[n])); n+=1
        startx2 = myfloat(float(header[n])); n+=1
        startx3 = myfloat(float(header[n])); n+=1
        _dx1=myfloat(float(header[n])); n+=1
        _dx2=myfloat(float(header[n])); n+=1
        _dx3=myfloat(float(header[n])); n+=1
        tf=myfloat(float(header[n])); n+=1
        nstep=myfloat(float(header[n])); n+=1
        a=myfloat(float(header[n])); n+=1
        gam=myfloat(float(header[n])); n+=1
        cour=myfloat(float(header[n])); n+=1
        DTd=myfloat(float(header[n])); n+=1
        DTl=myfloat(float(header[n])); n+=1
        DTi=myfloat(float(header[n])); n+=1
        DTr=myfloat(float(header[n])); n+=1
        DTr01=myfloat(float(header[n])); n+=1
        dump_cnt=myfloat(float(header[n])); n+=1
        image_cnt=myfloat(float(header[n])); n+=1
        rdump_cnt=myfloat(float(header[n])); n+=1
        rdump01_cnt=myfloat(float(header[n])); n+=1
        dt=myfloat(float(header[n])); n+=1
        lim=myfloat(float(header[n])); n+=1
        failed=myfloat(float(header[n])); n+=1
        Rin=myfloat(float(header[n])); n+=1
        Rout=myfloat(float(header[n])); n+=1
        hslope=myfloat(float(header[n])); n+=1
        R0=myfloat(float(header[n])); n+=1
        NPR=int(header[n]); n+=1
        DOKTOT=int(header[n]); n+=1
        DOCYLINDRIFYCOORDS=int(header[n]); n+=1
        fractheta = myfloat(header[n]); n+=1
        fracphi   = myfloat(header[n]); n+=1
        rbr       = myfloat(header[n]); n+=1
        npow2     = myfloat(header[n]); n+=1
        cpow2     = myfloat(header[n]); n+=1
        x10 = myfloat(header[n]); n+=1
        x20 = myfloat(header[n]); n+=1
        fracdisk = myfloat(header[n]); n+=1
        fracjet = myfloat(header[n]); n+=1
        r0disk = myfloat(header[n]); n+=1
        rdiskend = myfloat(header[n]); n+=1
        r0jet = myfloat(header[n]); n+=1
        rjetend = myfloat(header[n]); n+=1
        jetnu = myfloat(header[n]); n+=1
        rsjet = myfloat(header[n]); n+=1
        r0grid = myfloat(header[n]); n+=1
        BL = myfloat(header[n]); n+=1
    else:
        print("rdump header")
        nheader = 48
        n = 0
        #per tile resolution
        N1 = int(header[n]); n+=1
        N2 = int(header[n]); n+=1
        N3 = int(header[n]); n+=1
        #total resolution
        nx = int(header[n]); n+=1
        ny = int(header[n]); n+=1
        nz = int(header[n]); n+=1
        #numbers of ghost cells
        N1G = int(header[n]); n+=1
        N2G = int(header[n]); n+=1
        N3G = int(header[n]); n+=1
        #starting indices
        starti = int(header[n]); n+=1
        startj = int(header[n]); n+=1
        startk = int(header[n]); n+=1
        t = myfloat(header[n]); n+=1
        tf = myfloat(header[n]); n+=1
        nstep = int(header[n]); n+=1
        a = myfloat(header[n]); n+=1
        gam = myfloat(header[n]); n+=1
        game = myfloat(header[n]); n+=1
        game4 = myfloat(header[n]); n+=1
        game5 = myfloat(header[n]); n+=1
        cour = myfloat(header[n]); n+=1
        DTd = myfloat(header[n]); n+=1
        DTl = myfloat(header[n]); n+=1
        DTi = myfloat(header[n]); n+=1
        DTr = myfloat(header[n]); n+=1
        DTr01 = myfloat(header[n]); n+=1
        dump_cnt = myfloat(header[n]); n+=1
        image_cnt = myfloat(header[n]); n+=1
        rdump_cnt = myfloat(header[n]); n+=1
        rdump01_cnt=myfloat(float(header[n])); n+=1
        dt = myfloat(header[n]); n+=1
        lim = myfloat(header[n]); n+=1
        failed = myfloat(header[n]); n+=1
        Rin = myfloat(header[n]); n+=1
        Rout = myfloat(header[n]); n+=1
        hslope = myfloat(header[n]); n+=1
        R0 = myfloat(header[n]); n+=1
        fractheta = myfloat(header[n]); n+=1
        fracphi = myfloat(header[n]); n+=1
        rbr = myfloat(header[n]); n+=1
        npow2 = myfloat(header[n]); n+=1
        cpow2 = myfloat(header[n]); n+=1
        x10 = myfloat(header[n]); n+=1
        x20 = myfloat(header[n]); n+=1
        mrat = myfloat(header[n]); n+=1
        fel0 = myfloat(header[n]); n+=1
        felfloor = myfloat(header[n]); n+=1
        tdump = myfloat(header[n]); n+=1
        trdump = myfloat(header[n]); n+=1
        timage = myfloat(header[n]); n+=1
        tlog  = myfloat(header[n]); n+=1
    if n < len(header):
        nheader = 60
        global_fracdisk   = myfloat(header[n]); n+=1
        global_fracjet    = myfloat(header[n]); n+=1
        global_r0disk     = myfloat(header[n]); n+=1
        global_rdiskend   = myfloat(header[n]); n+=1
        global_r0jet      = myfloat(header[n]); n+=1
        global_rjetend    = myfloat(header[n]); n+=1
        global_jetnu      = myfloat(header[n]); n+=1
        global_rsjet      = myfloat(header[n]); n+=1
        global_r0grid     = myfloat(header[n]); n+=1
    if n != nheader or n != nheadertot:
        print("Wrong number of elements in header: nread = %d, nexpected = %d, nototal = %d: incorrect format?"
              % (n, nheader, nheadertot) )
        return headerline
    if returnheaderline:
        return headerline
    else:
        return header

# rewrites data_assign 
def data_assign(gd,**kwargs):
    # global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor, ktot, pg
    global t,nx,ny,nz,N1,N2,N3,N1G,N2G,N3G,starti,startj,startk,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp, games, startx1, startx2, startx3, x10, x20, tf, NPR, DOKTOT, BL
    nx = kwargs.pop("nx",nx)
    ny = kwargs.pop("ny",ny)
    nz = kwargs.pop("nz",nz)
    ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug = gd[0:11,:,:].view(); n = 11
    pg = (gam-1)*ug
    lrho=np.log10(rho)
    vu=np.zeros_like(gd[0:4])
    B=np.zeros_like(gd[0:4])
    vu[1:4] = gd[n:n+3]; n+=3
    B[1:4] = gd[n:n+3]; n+=3
    #if total entropy equation is evolved (on by default)
    if DOKTOT == 1:
      ktot = gd[n]; n+=1
    divb = gd[n]; n+=1
    uu = gd[n:n+4]; n+=4
    ud = gd[n:n+4]; n+=4
    bu = gd[n:n+4]; n+=4
    bd = gd[n:n+4]; n+=4
    bsq = mdot(bu,bd)
    v1m,v1p,v2m,v2p,v3m,v3p=gd[n:n+6]; n+=6
    gdet=gd[n]; n+=1
    rhor = 1+(1-a**2)**0.5
    if "guu" in globals():
        #lapse
        alpha = (-guu[0,0])**(-0.5)

    dump_dict = {
        't':t,
        'nx':nx,
        'ny':ny,
        'nz':nz,
        '_dx1':_dx1,
        '_dx2':_dx2,
        '_dx3':_dx3,
        'gam':gam,
        'hslope':hslope,
        'a':a,
        'R0':R0,
        'Rin':Rin,
        'Rout':Rout,
        'ti':ti,
        'tj':tj,
        'tk':tk,
        'x1':x1,
        'x2':x2,
        'x3':x3,
        'r':r,
        'h':h,
        'ph':ph,
        'rho':rho,
        'ug':ug,
        'vu':vu,
        'B':B,
        'pg':pg,
        # 'cs2':cs2,
        # 'Sden':Sden,
        # 'U':U,
        # 'gdetB':gdetB,
        'divb':divb,
        'uu':uu,
        'ud':ud,
        'bu':bu,
        'bd':bd,
        'v1m':v1m,
        'v1p':v1p,
        'v2m':v2m,
        'v2p':v2p,
        'gdet':gdet,
        'bsq':bsq,
        'gdet':gdet,
        # 'alpha':alpha,
        # 'rhor':rhor, 
        # 'ktot':ktot, 
        # 'pg':pg
    }
    

    if n != gd.shape[0]:
        print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
        return 1, dump_dict
    return 0, dump_dict

# reads in normal dump, returns dictionary of variables instead of injecting globals
def read_dump_util(dump: str, path_to_dumps:str = 'harmpi/dumps/',noround: int = 0):
    headerline = read_header(path_to_dumps + dump, returnheaderline = True)

    gd = read_body(path_to_dumps + dump,nx=N1+2*N1G,ny=N2+2*N2G,nz=N3+2*N3G,noround=1)
    if noround:
        return_code, dump_dict = data_assign(         gd,type=type,nx=N1+2*N1G,ny=N2+2*N2G,nz=N3+2*N3G)
    else:
        return_code, dump_dict = data_assign(myfloat(gd),type=type,nx=N1+2*N1G,ny=N2+2*N2G,nz=N3+2*N3G)
    
    return return_code, dump_dict

# reading dump, 
def read_dump_util_sc(
        dump: int, 
        set_mpi: callable,
        rblock_new: callable,
        rpar_new: callable,
        rgdump_new: callable,
        rdump_new: callable,
        rgdump_griddata: callable,
        rdump_griddata: callable,
        dumps_path: str = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced',
    ):

    os.chdir(dumps_path)
    
    global notebook, axisym,set_cart,axisym,REF_1,REF_2,REF_3,set_cart,D,print_fieldlines
    global lowres1,lowres2,lowres3, RAD_M1, RESISTIVE, export_raytracing_GRTRANS, export_raytracing_RAZIEH,r1,r2,r3
    global r_min, r_max, theta_min, theta_max, phi_min,phi_max, do_griddata, do_box, check_files, kerr_schild

    # set params
    lowres1 = 1
    lowres2 = 1
    lowres3 = 1
    
    do_box=0
    r_min=1.0
    r_max=100.0
    theta_min=0.0
    theta_max=9
    phi_min=-1
    phi_max=9
    axisym=1
    print_fieldlines=0
    export_raytracing_GRTRANS=0
    export_raytracing_RAZIEH=0
    kerr_schild=0
    DISK_THICKNESS=0.03
    set_cart=0
    set_mpi(0)
    check_files=1
    notebook=1
    
    interpolate_var=0
    
    AMR = 0 # get all data in grid

    rblock_new(dump)
    rpar_new(dump)

    if AMR:
        rgdump_new(dumps_path)
        rdump_new(dumps_path, dump)
    else:
        rgdump_griddata(dumps_path)
        rdump_griddata(dumps_path, dump)

    os.chdir(os.environ['HOME']+'/bh')

    dump_dict = {
        'r': r,
        'ug': ug,
        'uu': uu,
        'B': B,
        't': t,
    }

    return dump_dict

if __name__ == '__main__':
    # _, dump_dict = read_dump_util(dump='dump000')
    _, dump_dict = read_dump_util(path_to_dumps='/pscratch/sd/l/lalakos/ml_data_rc300',dump='dump000')

    # rho_val = dump_dict['rho']
    # print(f'rho variable for this dump: {rho_val}')
