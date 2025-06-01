
import time
import numpy as np

# 



# rblock_new_ml()
def rblock_new_ml():
    global AMR_ACTIVE, AMR_LEVEL,AMR_LEVEL1,AMR_LEVEL2,AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    global AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8
    global AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4, AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP
    global AMR_CORN1, AMR_CORN2, AMR_CORN3, AMR_CORN4, AMR_CORN5, AMR_CORN6
    global AMR_CORN7, AMR_CORN8, AMR_CORN9, AMR_CORN10, AMR_CORN11, AMR_CORN12
    global AMR_NBR1_3, AMR_NBR1_4, AMR_NBR1_7, AMR_NBR1_8, AMR_NBR2_1, AMR_NBR2_2, AMR_NBR2_3, AMR_NBR2_4, AMR_NBR3_1, AMR_NBR3_2, AMR_NBR3_5, AMR_NBR3_6, AMR_NBR4_5, AMR_NBR4_6, AMR_NBR4_7, AMR_NBR4_8
    global AMR_NBR5_1, AMR_NBR5_3, AMR_NBR5_5, AMR_NBR5_7, AMR_NBR6_2, AMR_NBR6_4, AMR_NBR6_6, AMR_NBR6_8
    global AMR_NBR1P, AMR_NBR2P, AMR_NBR3P, AMR_NBR4P, AMR_NBR5P, AMR_NBR6P
    global block, nmax, n_ord, AMR_TIMELEVEL

    AMR_ACTIVE, AMR_LEVEL, AMR_REFINED = 0,1,2
    AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT = 3,4,5,6
    AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8 = 7, 8, 9, 10, 11, 12, 13, 14
    AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4, AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP = 15,16,17,18,19,20,21,22,23
    AMR_CORN1, AMR_CORN2, AMR_CORN3, AMR_CORN4, AMR_CORN5, AMR_CORN6, AMR_CORN7, AMR_CORN8, AMR_CORN9, AMR_CORN10, AMR_CORN11, AMR_CORN12 = 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
    AMR_LEVEL1, AMR_LEVEL2, AMR_LEVEL3 = 110,111,112
    AMR_NBR1_3, AMR_NBR1_4, AMR_NBR1_7, AMR_NBR1_8, AMR_NBR2_1, AMR_NBR2_2, AMR_NBR2_3, AMR_NBR2_4, AMR_NBR3_1, AMR_NBR3_2, AMR_NBR3_5, AMR_NBR3_6, AMR_NBR4_5, AMR_NBR4_6, AMR_NBR4_7, AMR_NBR4_8, AMR_NBR5_1, AMR_NBR5_3, AMR_NBR5_5, AMR_NBR5_7, AMR_NBR6_2, AMR_NBR6_4, AMR_NBR6_6, AMR_NBR6_8=113, 114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136
    AMR_NBR1P, AMR_NBR2P, AMR_NBR3P, AMR_NBR4P, AMR_NBR5P, AMR_NBR6P=161,162,163,164,165,166
    AMR_TIMELEVEL=36

    if(os.path.isfile("gdumps/grid")):
        fin = open("gdumps/grid", "rb")
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
        # end = time.time()
        # print(f"End of elif: {end - start}")

    else:
        print("Cannot find grid file!")
        return

    with open("gdumps/grid", "rb") as fin:
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
        block = np.zeros((nmax, 200), dtype=np.int32, order='C')
        n_ord = np.zeros((nmax), dtype=np.int32, order='C')
        gd = np.fromfile(fin, dtype=np.int32, count=NV * nmax, sep='')
        gd = gd.reshape((NV, nmax), order='F').T
        # start = time.time()
        block[:,0:NV] = gd
        if(NV<170):
            block[:, AMR_LEVEL1] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL2] = gd[:, AMR_LEVEL]
            block[:, AMR_LEVEL3] = gd[:, AMR_LEVEL]

# rpar_new
def rpar_new(dump):
    global t, n_active, n_active_total, nstep, Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
    global bs1, bs2, bs3, nb1, nb2, nb3, startx1, startx2, startx3, _dx1, _dx2, _dx3
    global tf, a, gam, cour, Rin, Rout, R0, density_scale,REF_1,REF_2,REF_3, RAD_M1, RESISTIVE, TWO_T, P_NUM
    global nx, ny, nz, nb, rhor,temp_array, gd1_temp,gd2_temp, NODE, TIMELEVEL,flag_restore,r1,r2,r3, export_raytracing_RAZIEH, interpolate_var, rank

    if (os.path.isfile("dumps%d/parameters" % dump)):
        fin = open("dumps%d/parameters" % dump, "rb")
    else:
        print("Rpar error!")

    t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    Dtd = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtl = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtr = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    dump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    rdump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    dt = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    failed = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    bs1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    startx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    tf = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    a = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    gam = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    cour = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rin = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rout = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    R0 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    density_scale = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    for n in range(0,13):
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    if(trash >= 1000):
        P_NUM=1
        trash=trash-1000
    else:
        P_NUM=0
    if (trash >= 100):
        TWO_T = 1
        trash = trash - 100
    else:
        TWO_T = 0
    if (trash >= 10):
        RESISTIVE = 1
        trash = trash - 10
    else:
        RESISTIVE = 0
    if (trash >= 1):
        RAD_M1 = 1
        trash = trash - 1
    else:
        RAD_M1 = 0
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    #Set grid spacing
    _dx1=(np.log(Rout)-np.log(Rin))/(bs1*nb1)
    fractheta=-startx2
    #fractheta = 1.0 - 2.0 / (bs2*nb2) * (bs3*nb3>2.0)
    _dx2=2.0*fractheta/(bs2*nb2)
    _dx3=2.0*np.pi/(bs3*nb3)

    nb = n_active_total
    rhor = 1 + (1 - a ** 2) ** 0.5

    NODE=np.copy(n_ord)
    TIMELEVEL=np.copy(n_ord)

    REF_1=1
    REF_2=1
    REF_3=1
    flag_restore = 0
    size = os.path.getsize("dumps%d/parameters" % dump)
    if(size>=66*4+3*n_active_total*4):
        n=0
        while n<n_active_total:
            n_ord[n]=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            NODE[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n=n+1
    elif(size >= 66 * 4 + 2 * n_active_total * 4):
        n = 0
        flag_restore=1
        while n < n_active_total:
            n_ord[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n = n + 1

    if(export_raytracing_RAZIEH==1 and (bs1%lowres1!=0 or bs2%lowres2!=0 or bs3%lowres3!=0 or ((lowres1 & (lowres1-1) == 0) and lowres1 != 0)!=1 or ((lowres2 & (lowres2-1) == 0) and lowres2 != 0)!=1 or ((lowres3 & (lowres3-1) == 0) and lowres3 != 0)!=1)):
        if(rank==0):
            print("For raytracing block size needs to be divisable by lowres!")
    if(export_raytracing_RAZIEH==1 and interpolate_var==0):
        if (rank == 0):
            print("Warning: Variable interpolation is highly recommended for raytracing!")
    fin.close()

# rgdump_griddata
def rgdump_griddata(dir):
    global ti, tj, tk, x1, x2, x3, r, h, ph, gcov, gcon, gdet, drdx, dxdxp, alpha, axisym, interpolate_var
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new, set_cart, set_xc, lowres1,lowres2,lowres3
    global nb1, nb2, nb3, REF_1, REF_2, REF_3
    global startx1,startx2,startx3,_dx1,_dx2,_dx3, export_raytracing_RAZIEH
    global r_min, r_max, theta_min, theta_max, phi_min, phi_max, i_min, i_max, j_min, j_max, z_min, z_max, do_box, rank, gridsizex1, gridsizex2, gridsizex3, check_files
    # global block, nmax, n_ord, AMR_TIMELEVEL
    import pp_c

    set_cart=0
    set_xc=0

    ACTIVE1 = np.max(block[n_ord, AMR_LEVEL1])*REF_1
    ACTIVE2 = np.max(block[n_ord, AMR_LEVEL2])*REF_2
    ACTIVE3 = np.max(block[n_ord, AMR_LEVEL3])*REF_3

    if ((int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1) % lowres1) != 0 or (int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2) % lowres2) != 0 or (int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3) % lowres3) != 0):
        print("Incompatible lowres settings in rgdump_griddata")

    gridsizex1 = int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1/lowres1)
    gridsizex2 = int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2/lowres2)
    gridsizex3 = int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3/lowres3)

    _dx1 = _dx1 * lowres1 * (1.0 / (1.0 + REF_1) ** ACTIVE1)
    _dx2 = _dx2 * lowres2 * (1.0 / (1.0 + REF_2) ** ACTIVE2)
    _dx3 = _dx3 * lowres3 * (1.0 / (1.0 + REF_3) ** ACTIVE3)

    #Calculate inner and outer boundaries of selection box after upscaling and downscaling; Assumes uniform grid x1=log(r) etc
    if(do_box==1):
        i_min = max(np.int32((np.log(r_min)-(startx1+0.5*_dx1)) / _dx1) + 1, 0)
        i_max = min(np.int32((np.log(r_max)-(startx1+0.5*_dx1)) / _dx1) + 1, gridsizex1)
        j_min=max(np.int32(((2.0/np.pi*(theta_min)-1.0)-(startx2+0.5*_dx2))/_dx2) + 1,0)
        j_max=min(np.int32(((2.0/np.pi*(theta_max)-1.0)-(startx2+0.5*_dx2))/_dx2) + 1,gridsizex2)
        z_min=max(np.int32((phi_min-(startx3+0.5*_dx3))/_dx3) + 1,0)
        z_max=min(np.int32((phi_max-(startx3+0.5*_dx3))/_dx3) + 1,gridsizex3)

        gridsizex1 = i_max-i_min
        gridsizex2 = j_max-j_min
        gridsizex3 = z_max-z_min

        if((j_max<j_min or i_max<i_min or z_max<z_min) and rank==0):
            print("Bad box selection")
    else:
        i_min=0
        i_max=gridsizex1
        j_min=0
        j_max=gridsizex2
        z_min=0
        z_max=gridsizex3

    nx = gridsizex1
    ny = gridsizex2
    nz = gridsizex3

    # Allocate memory
    x1 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    x2 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    x3 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    r = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    h = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    ph = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if axisym:
        gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
        gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
        gdet = np.zeros((1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
        dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
    else:
        gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        gdet = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if(rank==0 and check_files==1):
        for n in range(0,n_active_total):
            if(os.path.isfile('gdumps/gdump%d' %n_ord[n])==0 or os.path.getsize('gdumps/gdump%d' %n_ord[n])!=(9*bs1*bs2*bs3+(bs1*bs2*49)*(axisym)+(bs1*bs2*bs3*49)*(axisym==0))*8):
                print("Gdump file %d doesn't exist" %n_ord[n])
    size = os.path.getsize('gdumps/gdump%d' %n_ord[0])
    if(size==58*bs3*bs2*bs1*8):
        flag=1
    else:
        flag=0

    pp_c.rgdump_griddata(flag, interpolate_var, dir, axisym, n_ord,lowres1, lowres2, lowres3 ,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet,block, nb1, nb2, nb3, REF_1, REF_2, REF_3, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]), startx1,startx2,startx3,_dx1,_dx2,_dx3, export_raytracing_RAZIEH, i_min, i_max, j_min, j_max, z_min, z_max)

# rdump_griddata
def rdump_griddata(dir, dump):
    # global rho, ug, uu, B
    global uu_rad, E_rad, E,  TE, TI, photon_number, RAD_M1, RESISTIVE, TWO_T, P_NUM, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, gcov,gcon,axisym,_dx1,_dx2,_dx3, nb, nb1, nb2, nb3, REF_1, REF_2, REF_3, n_ord, interpolate_var, export_raytracing_GRTRANS,export_raytracing_RAZIEH, DISK_THICKNESS, a, gam, bsq, Rdot
    global startx1,startx2,startx3,_dx1,_dx2,_dx3,x1,x2,x3
    global r_min, r_max, theta_min, theta_max, phi_min, phi_max, i_min, i_max, j_min, j_max, z_min, z_max, do_box, check_files
    import pp_c

    # Allocate memory
    rho = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    ug = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    uu = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    B = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if(export_raytracing_RAZIEH):
        Rdot = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        Rdot = np.zeros((1, 1, 1, 1), dtype=mytype, order='C')
    bsq = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if(RAD_M1):
        E_rad = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        uu_rad = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        E_rad=np.copy(ug)
        uu_rad=np.copy(uu)

    if (RESISTIVE):
        E = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        E = B

    if (TWO_T):
        TE = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        TI = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        TE = rho
        TI = rho

    if (P_NUM):
        photon_number = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        photon_number = rho
        
    if (os.path.isfile("dumps%d/new_dump" % dump)):
        flag = 1
    else:
        if (rank == 0 and check_files == 10):
            for count in range(0, 5400):
                if (os.path.isfile("dumps%d/new_dump%d" %(dump, count))==0):
                    print("Dump file %d in folder %d doesn't exist" %(count,dump))
        flag = 0

    pp_c.rdump_griddata(
        flag, 
        interpolate_var, 
        np.int32(RAD_M1),
        np.int32(RESISTIVE), 
        TWO_T, 
        P_NUM, 
        dir, 
        dump, 
        n_active_total, 
        lowres1, 
        lowres2, 
        lowres3, 
        nb,
        bs1,
        bs2,
        bs3, 
        rho,
        ug, 
        uu, 
        B, 
        E, 
        E_rad, 
        uu_rad, 
        TE, 
        TI, 
        photon_number, 
        gcov,
        gcon,
        axisym,
        n_ord,
        block, 
        nb1,
        nb2,
        nb3,
        REF_1, 
        REF_2,
        REF_3, 
        np.max(block[n_ord, AMR_LEVEL1]),
        np.max(block[n_ord, AMR_LEVEL2]), 
        np.max(block[n_ord, AMR_LEVEL3]),
        export_raytracing_RAZIEH, 
        DISK_THICKNESS, 
        a, 
        gam, 
        Rdot, 
        bsq, 
        r, 
        startx1,
        startx2,
        startx3,
        _dx1,
        _dx2,
        _dx3,
        x1,
        x2,
        x3, 
        i_min, 
        i_max, 
        j_min, 
        j_max, 
        z_min, 
        z_max,
    )

    print(flag)
    print(interpolate_var)
    print(npint32(RAD_M1))
    print(npint32(RESISTIVE))
    print(TWO_T)
    print(P_NUM)
    print(dir)
    print(dump)
    print(n_active_total)
    print(lowres1)
    print(lowres2)
    print(lowres3)
    print(nb)
    print(bs1)
    print(bs2)
    print(bs3)
    print(rho)
    print(ug)
    print(uu)
    print(B)
    print(E)
    print(E_rad)
    print(uu_rad)
    print(TE)
    print(TI)
    print(photon_number)
    print(gcov)
    print(gcon)
    print(axisym)
    print(n_ord)
    print(block)
    print(nb1)
    print(nb2)
    print(nb3)
    print(REF_1)
    print(REF_2)
    print(REF_3)
    print(np.max(block[n_ord, AMR_LEVEL1]))
    print(np.max(block[n_ord, AMR_LEVEL2]))
    print(np.max(block[n_ord, AMR_LEVEL3]))
    print(export_raytracing_RAZIEH)
    print(DISK_THICKNESS)
    print(a)
    print(gam)
    print(Rdot)
    print(bsq)
    print(r)
    print(startx1)
    print(startx2)
    print(startx3)
    print(_dx1)
    print(_dx2)
    print(_dx3)
    print(x1)
    print(x2)
    print(x3)
    print(i_min)
    print(i_max)
    print(j_min)
    print(j_max)
    print(z_min)
    print(z_max)

    bs1new = gridsizex1
    bs2new = gridsizex2
    bs3new = gridsizex3

    if (do_box == 1):
        startx1 = startx1 + (i_min) * _dx1
        startx2 = startx2 + (j_min) * _dx2
        startx3 = startx3 + (z_min) * _dx3

    nb2d = nb
    nb = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1

    return rho, ug, uu, B

if __name__ == '__main__':

    dump_index = 5
    dumps_path = '/pscratch/sd/l/lalakos/ml_data_rc300/reduced'

    start = time.time()
    rho, ug, uu, B = rdump_griddata(dir=dumps_path, dump=dump_index)
    print(f'rho shape: {rho.shape()}')
    f'Read time of dump {dump} {time.time()-start:.4f}s'


    
