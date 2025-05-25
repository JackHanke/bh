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

# arjun's
def rblock_new_opt(dump):
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
    
    # Read in data for every block
    # print("_" * 20)
    # start = time.time()
    if (os.path.isfile("dumps%d/grid" % dump)):
        fin = open("dumps%d/grid" % dump, "rb")
        size = os.path.getsize("dumps%d/grid" % dump)
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = 36
        # end = time.time()
        # print(f"End of if: {end - start}")
        
    elif(os.path.isfile("gdumps/grid")):
        fin = open("gdumps/grid", "rb")
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
        # end = time.time()
        # print(f"End of elif: {end - start}")
        
    else:
        print("Cannot find grid file in dump %d !" %dump)

    # Allocate memory
    # start_mem = time.time()
    block = np.zeros((nmax, 200), dtype=np.int32, order='C')
    n_ord = np.zeros((nmax), dtype=np.int32, order='C')
    # end_mem = time.time()
    # print(f"end of memory allocation: {end_mem - start_mem}")

    # print(f"NV * nmax: {NV * nmax}")
    
    # start_gd_memmap = time.time()
    gd_mem = np.memmap(fin, dtype=np.int32, mode='r')[1:]
    # end_load_gd_mem = time.time()
    # print(f"end of loading gd mem: {end_load_gd_mem - start_gd_memmap} Shape: {gd_mem.shape}")

    gd_mem = gd_mem.reshape((NV, nmax), order='F').T
    # end_gd_mem_reshape = time.time()
    # print(f"end of reshape gd mem: {end_gd_mem_reshape - end_load_gd_mem}")

    # start_process_gd_mem = time.time()
    block[:,0:NV] = gd_mem
    if(NV<170):
        block[:, AMR_LEVEL1] = gd_mem[:, AMR_LEVEL]
        block[:, AMR_LEVEL2] = gd_mem[:, AMR_LEVEL]
        block[:, AMR_LEVEL3] = gd_mem[:, AMR_LEVEL]
    
    # intermediate_process_gd_mem = time.time()
    # print(f"intermediate processing gd mem: {intermediate_process_gd_mem - start_process_gd_mem}")
    i = 0
    if (os.path.isfile("dumps%d/grid" % dump)):
        for n in range(0, nmax):
            if block[n, AMR_ACTIVE] == 1:
                n_ord[i] = n
                i += 1
    # print(f"end of procesing grid data mem: {time.time() - intermediate_process_gd_mem}")
    fin.close()
    # print("_" * 20)

if __name__ == '__main__':
    # _, dump_dict = read_dump_util(dump='dump000')
    _, dump_dict = read_dump_util(path_to_dumps='/pscratch/sd/l/lalakos/ml_data_rc300',dump='dump000')
    # rho_val = dump_dict['rho']
    # print(f'rho variable for this dump: {rho_val}')



    ## NOTE compare rblock_new with custom grid read
    tot_start = time.time()
    for dump in [0, 500, 43, 762, 1001]:
        print(f'-----dump {dump}-----')
        start = time.time()
        rblock_new(dump=dump)
        print(f'Total rblock_new: {time.time()-start:.3f} s')
    print(f'Total of 5 dumps with : {time.time()-tot_start:.3f} s')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
    tot_start = time.time()
    ## NOTE replaces rblock_new call entirely
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

        i = 0
    for dump in [1, 501, 44, 763, 1002]:
        print(f'-----dump {dump}-----')
        start = time.time()
        print(f'Total rblock_new: {time.time()-start:.3f} s')

    print(f'Total of 5 dumps with : {time.time()-tot_start:.3f} s')
