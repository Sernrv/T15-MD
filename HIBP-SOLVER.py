# -*- coding: utf-8 -*-
'''
T-15MD tokamak, HIBP

Program calculates trajectories and selects voltages on
primary beamline (B2 plates) and secondary beamline (A3, B3, A4 plates)
'''
import os
import sys
import time
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import hibpcalc.geomfunc as gf
import hibpcalc.fields as fields
from hibpcalc.misc import SecBeamlineData
import hibplib as hb
import hibpplotlib as hbplot
import define_geometry as defgeom


# %% additional functions
def sign_eps(x, eps):
    if abs(x) < eps:
        return 0.0
    else:
        return np.sign(x)


def linear_at_zero(k=0.3):
    def f(x):
        # return 0.01*np.sign(x)
        x = x * k  # 2.0
        if abs(x) > 0.5:
            return 0.5 * np.sign(x)
        else:
            return x

    return f


'''
plA3 = geomT15.plates_dict["A3"]

'''


# def calc_shift(V, plates, adaptive_aim, eps=0.1):
def calc_shift(V, plates, adaptive_aim, func=linear_at_zero(2.0)):
    if not adaptive_aim:
        return np.array([0., 0., 0.])

    basis = plates.front_basis(norm=False)
    Vvert = V.dot(basis[0]) / np.linalg.norm(basis[0])
    Vhorz = V.dot(basis[1]) / np.linalg.norm(basis[1])

    sin_vert = Vvert / np.linalg.norm(V)
    sin_horz = Vhorz / np.linalg.norm(V)

    # dv = -basis[0] * sign_eps(sin_vert, eps)*0.5
    # dh = -basis[1] * sign_eps(sin_horz, eps)*0.5
    dv = -basis[0] * func(sin_vert)
    dh = -basis[1] * func(sin_horz)

    return dv + dh


# %% set flags
optimizeB2 = True
optimizeA3B3 = True
calculate_zones = False
pass2AN = True
save_radref = False
save_primary = False
save_sec = False
pass2aim_only = False
load_traj_from_file = False
save_grids_and_angles = True
adaptive_aim = False
debag = False
multiple_beamlines = False

# %% set up main parameters
regime = 'work'

analyzer = 1
beamline_num = 23

beamlines_1an = [SecBeamlineData(2.5, -0.3, 0.0, 14.0, 14.5, -20.0),  # 0
                 SecBeamlineData(2.5, -0.2, 0.0, 16.5, 15.0, -20.0),  # 1
                 SecBeamlineData(2.5, -0.1, 0.0, 19.0, 15.5, -20.0),  # 2
                 SecBeamlineData(2.5, 0.0, 0.0, 23.5, 16.0, -20.0),  # 3
                 SecBeamlineData(2.5, 0.1, 0.0, 26.5, 15.5, -20.0),  # 4
                 SecBeamlineData(2.5, 0.2, 0.0, 31.5, 17.0, -20.0),  # 5
                 SecBeamlineData(2.5, 0.3, 0.0, 36.0, 18.5, -20.0),  # 6

                 SecBeamlineData(2.6, -0.3, 0.0, 17.5, 17.5, -20.0),  # 7
                 SecBeamlineData(2.6, -0.2, 0.0, 24.5, 18.0, -20.0),  # 8
                 SecBeamlineData(2.6, -0.1, 0.0, 25.5, 18.0, -20.0),  # 9
                 SecBeamlineData(2.6, 0.0, 0.0, 29.5, 19.0, -20.0),  # 10
                 SecBeamlineData(2.6, 0.1, 0.0, 34.0, 21.0, -20.0),  # 11
                 SecBeamlineData(2.6, 0.2, 0.0, 40.5, 23.0, -20.0),  # 12
                 SecBeamlineData(2.6, 0.3, 0.0, 43.0, 25.5, -20.0),  # 13

                 SecBeamlineData(2.7, -0.3, 0.0, 17.0, 18.0, -20.0),  # 14
                 SecBeamlineData(2.7, -0.2, 0.0, 25.5, 20.0, -20.0),  # 15
                 SecBeamlineData(2.7, -0.1, 0.0, 29.5, 21.5, -20.0),  # 16
                 SecBeamlineData(2.7, 0.0, 0.0, 33.5, 23.0, -20.0),  # 17
                 SecBeamlineData(2.7, 0.1, 0.0, 38.0, 24.5, -20.0),  # 18
                 SecBeamlineData(2.7, 0.2, 0.0, 41.5, 27.5, -20.0),  # 19
                 SecBeamlineData(2.7, 0.3, 0.0, 49.0, 32.5, -20.0),  # 20

                 SecBeamlineData(2.8, -0.3, 0.0, 18.0, 19.5, -20.0),  # 21
                 SecBeamlineData(2.8, -0.2, 0.0, 24.0, 21.0, -20.0),  # 22
                 SecBeamlineData(2.8, -0.1, 0.0, 32.0, 23.5, -20.0),  # 23
                 SecBeamlineData(2.8, 0.0, 0.0, 37.0, 24.5, -20.0),  # 24
                 SecBeamlineData(2.8, 0.1, 0.0, 42.5, 27.5, -20.0),  # 25
                 SecBeamlineData(2.8, 0.2, 0.0, 45.0, 30.5, -20.0),  # 26
                 SecBeamlineData(2.8, 0.3, 0.0, 51.0, 35.0, -20.0),  # 27

                 SecBeamlineData(2.9, -0.1, 0.0, 20.0, 15.0, -20.0),  # 28
                 SecBeamlineData(2.9, 0.0, 0.0, 20.0, 15.0, -20.0),  # 29
                 SecBeamlineData(2.9, 0.1, 0.0, 20.0, 15.0, -20.0),  # 30
                 SecBeamlineData(2.9, 0.2, 0.0, 20.0, 15.0, -20.0),  # 31

                 SecBeamlineData(3.0, -0.1, 0.0, 20.0, 15.0, -20.0),  # 32
                 SecBeamlineData(3.0, 0.0, 0.0, 20.0, 15.0, -20.0),  # 33
                 SecBeamlineData(3.0, 0.1, 0.0, 20.0, 15.0, -20.0),  # 34
                 SecBeamlineData(3.0, 0.2, 0.0, 20.0, 15.0, -20.0),  # 35
                 ]

if multiple_beamlines:
    start = 24
    diapason = 4
    beamline_indexes = range(start, start + diapason, 1)
else:
    beamline_indexes = [beamline_num]

# toroidal field on the axis
Btor = 1.0  # [T]
Ipl = 1.0  # Plasma current [MA]
print('\nShot parameters: Btor = {} T, Ipl = {} MA'.format(Btor, Ipl))

# timestep [sec]
dt = 0.2e-7  # 0.7e-7

# probing ion charge and mass
q = 1.602176634e-19  # electron charge [Co]
m_ion = 204.3833 * 1.6605e-27  # Tl ion mass [kg]

# beam energy
energies = {'work': (40., 380., 20.), 'test': (300., 300., 20.)}

Emin, Emax, dEbeam = energies[regime]

# %% beamline loop
primary_grids = {}

for beamline_num in beamline_indexes:

    # %%set paths
    results_folder = r"D:\T15_new\output\Оптимизация точки пристрелки\Отчёт\x=" + str(beamlines_1an[beamline_num].xaim)
    traj2load = ['beamline {}'.format(beamline_num)]
    # %% set voltages
    # UA2 voltages
    UU_A2 = {'work': (-10., 50., 2.), 'test': (-10., 50., 2.)}
    UA2min, UA2max, dUA2 = UU_A2[regime]  # 32., 32., 2. #12., 12., 2.  #0., 34., 2.  # -3, 33., 3.  # -3., 30., 3.
    NA2_points = 10

    # B2 plates voltage
    UB2, dUB2 = 0.0, 5.0  # 10.  # [kV], [kV/m]

    # B3 voltages
    UB3, dUB3 = 0.0, 10  # [kV], [kV/m]

    # A3 voltages
    UA3, dUA3 = 0.0, 7.0  # [kV], [kV/m]
    if analyzer == 2:
        dUA3 = -dUA3

    # A4 voltages
    UA4, dUA4 = 0.0, 2.0  # [kV], [kV/m]

    # %% Define Geometry
    geomT15 = defgeom.define_geometry(analyzer=analyzer, beamline_num=beamline_num)
    r0 = geomT15.r_dict['r0']  # trajectory starting point

    # angles of aim plane normal [deg]
    alpha_aim = 0.
    beta_aim = 0.
    stop_plane_n = gf.calc_vector(1.0, alpha_aim, beta_aim)

    # %% Load Electric Field
    E_slow = {}
    E_fast = {}
    # load E for primary beamline
    try:
        fields._read_plates('prim', geomT15, E_slow, E_fast, hb.createplates)
        print('\n Primary Beamline loaded')
    except FileNotFoundError:
        print('\n Primary Beamline NOT FOUND')

    # load E for secondary beamline
    try:
        fields._read_plates('sec', geomT15, E_slow, E_fast, hb.createplates)
        # add diafragm for A3 plates to Geometry
        # hb.add_diafragm(geomT15, 'A3', 'A3d', diaf_width=0.05)
        hb.add_diafragm(geomT15, 'A4', 'A4d', diaf_width=0.05)
        print('\n Secondary Beamline loaded')
    except FileNotFoundError:
        print('\n Secondary Beamline NOT FOUND')

    # E = E_slow
    E = E_fast

    # %% Analyzer parameters
    if 'an' in geomT15.plates_dict.keys():
        # Analyzer G
        G = geomT15.plates_dict['an'].G
        # add detector coords to dictionary
        edges = geomT15.plates_dict['an'].det_edges
        geomT15.r_dict['det'] = edges[edges.shape[0] // 2][0]
    else:
        G = 1.
        print('\nNO Analyzer')

    # %% Load Magnetic Field
    pf_coils = fields.import_PFcoils('PFCoils.dat')
    PF_dict = fields.import_PFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)
    if 'B' not in locals():
        dirname = 'magfield'
        B = fields.read_B_new(Btor, Ipl, PF_dict, dirname=dirname)
    else:
        print('B already loaded')

    # %% Optimize Primary Beamline   
    if not load_traj_from_file:
        # define list of trajectories that hit r_aim
        traj_list_B2 = []
        if debag:
            prim_intersect = []
            sec_intersect = []
            smth_is_wrong = []
        # initial beam energy range
        Ebeam_range = np.arange(Emin, Emax + dEbeam, dEbeam)  # [keV]

        for Ebeam in Ebeam_range:
            # set shifted aim
            shift = np.zeros(3)
            geomT15.r_dict['aim_zshift'] = geomT15.r_dict['aim']

            dUB2 = Ebeam / 16.
            t1 = time.time()
            shot = ''
            input_fname = ''
            print('\n>>INPUT FILE: ', input_fname)
            if input_fname != '':
                exp_voltages = np.loadtxt(input_fname)
                indexes = np.linspace(1, exp_voltages.shape[0] - 1,
                                      NA2_points, dtype=int)
            if optimizeB2:
                optimizeA3B3 = True
                target = 'aim_zshift'  # 'aim'  # 'aimB3'
                # A2 plates voltage
                UA2_range = np.arange(UA2min, UA2max + dUA2, dUA2)
                # UA2_range = np.linspace(UA2min, UA2max, NA2_points)  # [kV]
                eps_xy, eps_z = 1e-3, 2.5e-2
                print('\n Primary beamline optimization')
            else:
                target = 'aim'
                UA2_range = exp_voltages[indexes, 1]
                UB2_range = exp_voltages[indexes, 2]
                eps_xy, eps_z = 1e-3, 1.
                print('\n Calculating primary beamline')
            if not optimizeA3B3:
                target = 'aim'
                UA3_range = exp_voltages[indexes, 3]
                UB3_range = exp_voltages[indexes, 4]
                eps_xy, eps_z = 1e-3, 1.

            # UA2 loop
            for i in range(UA2_range.shape[0]):
                UA2 = UA2_range[i]
                if not optimizeB2:
                    UB2 = UB2_range[i]
                if not optimizeA3B3:
                    UA3, UB3 = UA3_range[i], UB3_range[i]

                # reset aim point
                shift = np.zeros(3)
                geomT15.r_dict['aim_zshift'] = geomT15.r_dict['aim']

                print('\n\nE = {} keV; UA2 = {:.2f} kV\n'.format(Ebeam, UA2))
                # dict of starting voltages
                U_dict = {'A2': UA2, 'B2': UB2,
                          'A3': UA3, 'B3': UB3, 'A4': UA4, 'an': Ebeam / (2 * G)}
                # create new trajectory object
                tr = hb.Traj(q, m_ion, Ebeam, r0, geomT15.angles_dict['r0'][0],
                             geomT15.angles_dict['r0'][1], U_dict, dt)

                # optimize B2 voltage
                # here the trajectories calculated !!!
                tr = hb.optimize_B2(tr, geomT15, UB2, dUB2, E, B, dt, stop_plane_n,
                                    target, optimizeB2, eps_xy=eps_xy, eps_z=eps_z)

                # check geometry intersection
                if True in tr.IntersectGeometry.values():
                    if debag:
                        prim_intersect.append(tr)
                    print('NOT saved, primary intersected geometry')
                    continue
                if True in tr.IntersectGeometrySec.values():
                    if debag:
                        sec_intersect.append(tr)
                    print('NOT saved, secondary intersected geometry')
                    continue
                # if no intersections, upldate UB2 values
                UB2 = tr.U['B2']
                # check aim
                if tr.IsAimXY and tr.IsAimZ:

                    # calc shift
                    shift = calc_shift(tr.RV_sec[-1, 3:6], geomT15.plates_dict['A3'], adaptive_aim)

                    # if not shifted - save
                    if np.all(np.isclose(shift, np.zeros(3))):
                        traj_list_B2.append(tr)
                        print('\n Trajectory saved, UB2={:.2f} kV'.format(tr.U['B2']))

                    # else - update aim and optimize again
                    else:
                        geomT15.r_dict['aim_zshift'] = geomT15.r_dict['aim'] + shift
                        tr_shifted = copy.deepcopy(tr)
                        tr_shifted = hb.optimize_B2(tr_shifted, geomT15, UB2, dUB2, E, B, dt, stop_plane_n,
                                                    target, optimizeB2, eps_xy=eps_xy, eps_z=eps_z)

                        if (True in tr_shifted.IntersectGeometry.values()) or (
                                True in tr_shifted.IntersectGeometrySec.values()):
                            traj_list_B2.append(tr)
                            print('\n Trajectory saved without aim shift, UB2={:.2f} kV'.format(tr.U['B2']))
                            continue
                        if tr_shifted.IsAimXY and tr_shifted.IsAimZ:
                            traj_list_B2.append(tr_shifted)
                            print('\n Trajectory saved, UB2={:.2f} kV'.format(tr.U['B2']))
                            continue
                        else:
                            traj_list_B2.append(tr)
                            print('\n Trajectory saved without aim shift, UB2={:.2f} kV'.format(tr.U['B2']))
                            continue

                else:
                    if debag:
                        smth_is_wrong.append(tr)
                    print('NOT saved, sth is wrong')
                # traj_list_B2.append(tr)

        t2 = time.time()
        if optimizeB2:
            print('\n B2 voltage optimized, t = {:.1f} s\n'.format(t2 - t1))
        else:
            print('\n Trajectories to r_aim calculated, t = {:.1f} s\n'.format(t2 - t1))

        # %%

        traj_list_passed = copy.deepcopy(traj_list_B2)
        primary_grids[beamline_num] = traj_list_passed

        # %% Save traj list
        if save_primary:
            hb.save_traj_list(traj_list_passed, Btor, Ipl, beamline_num)

        # %% Additional plots
        if save_grids_and_angles:
            hbplot.plot_grid(traj_list_passed, geomT15, Btor, Ipl,
                             onlyE=True, marker_A2='')
            # hbplot.plot_fan(traj_list_passed, geomT15, Ebeam, UA2, Btor, Ipl,
            #                 plot_analyzer=False, plot_traj=True, plot_all=False)

            # hbplot.plot_scan(traj_list_passed, geomT15, Ebeam, Btor, Ipl,
            #                   full_primary=False, plot_analyzer=True,
            #                   plot_det_line=True, subplots_vertical=True, scale=4)
            anglesdict = hbplot.plot_sec_angles(traj_list_passed, Btor, Ipl,
                                                linestyle='-o', Ebeam='all')
            # hbplot.plot_fan(traj_list_passed, geomT15, 240., 40., Btor, Ipl)

            # get data to create path name
            zport_in = 0 if geomT15.r_dict['port_in'][2] == 0 else geomT15.r_dict['port_in'][2]
            beta_prim = int(geomT15.angles_dict['B2'][1])
            y_aim = int(geomT15.r_dict['aim'][1] * 1000)
            z_aim = int(geomT15.r_dict['aim'][2] * 1000)

            # path to create folder and save plots and log.txt
            path = os.path.join(results_folder,
                                f"B_tor{(Btor)}", f"Ipl{(Ipl)}",
                                f"prim_z{zport_in}_beta{beta_prim}",
                                f"y_aim{y_aim}_z_aim{z_aim}")

            # create new directory
            os.makedirs(path, exist_ok=True)

            """ save plots to path """

            if os.path.exists(path):
                # get info about plots
                fig_nums = plt.get_fignums()
                figs = [plt.figure(n) for n in fig_nums]

                # resize and save plots
                figs[0].set_size_inches(20, 12.5)
                figs[0].axes[0].set_xlim(1.0, 2.6)
                figs[0].axes[0].set_ylim(-0.5, 1.5)
                figs[0].savefig(os.path.join(path, "grid.png"), dpi=300)

                figs[1].set_size_inches(20, 12.5)
                figs[1].savefig(os.path.join(path, "exit_alpha.png"), dpi=300)

                figs[2].set_size_inches(20, 12.5)
                figs[2].savefig(os.path.join(path, "exit_beta.png"), dpi=300)

                # close opened plots

                plt.close(figs[0])
                plt.close(figs[1])
                plt.close(figs[2])

            """ get min max of exit alpha and beta """

            # create two arrays with all exit alphas and betas
            array = list(anglesdict.items())
            alphas = []
            betas = []

            # add all alphas and betas from anglesdict to arrays
            for i in range(len(array)):
                for j in range(len(array[i][1])):
                    alphas.append(array[i][1][j][2])
                    betas.append(array[i][1][j][3])

            # find min max in exit alphas and betas and create formatted string
            # example "0 : 48 / -17 : 54"
            diapason = f"{math.floor(min(alphas))} : {math.ceil(max(alphas))} / {math.floor(min(betas))} \
: {math.ceil(max(betas))}"

            """save file log.txt with initital parameters to folder"""

            # create list with main parameters
            logfile = [f"Path: {path}",
                       f"B_tor: {Btor}",
                       f"Ipl: {Ipl}",
                       f"prim_z: {geomT15.r_dict['port_in'][2]}", f"beta: {geomT15.angles_dict['B2'][1]}",
                       f"y_aim: {geomT15.r_dict['aim'][1]}", f"z_aim: {geomT15.r_dict['aim'][2]}",
                       diapason]

            # save log.txt to path
            np.savetxt(os.path.join(path, "log.txt"), logfile, fmt='%s')
            # print log.txt to console
            print(*logfile, sep='\n')

    # %% load trajectory list for further optimization
    if load_traj_from_file:
        traj_list = []
        for name in traj2load:
            traj_list += hb.read_traj_list(name, dirname=r'output/sec/B1_I1/')
        traj_list_passed = copy.deepcopy(traj_list)
        eps_xy, eps_z = 1e-3, 1e-3

    # %%
    # if pass2aim_only:
    #     sys.exit()

    # %% Optimize Secondary Beamline

    t1 = time.time()
    if optimizeA3B3:
        traj_list = copy.deepcopy(traj_list_passed)
        traj_list_a3b3 = hb.optimize_sec_fine(E, B, geomT15, traj_list)
        t2 = time.time()
        print('\n A3, B3 & A4 voltages optimized, t = {:.1f} s\n'.format(t2 - t1))
    else:
        print('\n Calculating secondary beamline')
        traj_list_a3b3_calc = []
        for tr in copy.deepcopy(traj_list_passed):
            print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A2']))
            RV0 = np.array([tr.RV_sec[0]])
            tr.pass_sec(RV0, geomT15.r_dict['slit'], E, B, geomT15,
                        stop_plane_n=geomT15.plates_dict['an'].det_plane_n,
                        tmax=9e-5, eps_xy=eps_xy, eps_z=eps_z)
            traj_list_a3b3_calc.append(tr)
        t2 = time.time()
        print('\n Secondary beamline calculated, t = {:.1f} s\n'.format(t2 - t1))

    # %% Save traj list
    if save_sec:
        hb.save_traj_list(traj_list_a3b3, Btor, Ipl, beamline_num, dirname='output/sec')

# %% Calculate ionization zones
if calculate_zones:
    t1 = time.time()
    slits = [2]
    traj_list_zones = []
    print('\n Ionization zones calculation')
    for tr in copy.deepcopy(traj_list_a3b3):
        print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A2']))
        tr = hb.calc_zones(tr, dt, E, B, geomT15, slits=slits,
                           timestep_divider=6,
                           eps_xy=1e-4, eps_z=1, dt_min=1e-12,
                           no_intersect=True, no_out_of_bounds=True)
        traj_list_zones.append(tr)
        print('\n Trajectory saved')
    t2 = time.time()
    print('\n Ionization zones calculated, t = {:.1f} s\n'.format(t2 - t1))

    hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl,
                             slits=slits, plot_fan=False)

# %% Pass to ANALYZER
if pass2AN:
    print('\n Passing to ANALYZER {}'.format(analyzer))
    # define list of trajectories that hit detector
    traj_list_an = []
    for tr in copy.deepcopy(traj_list_a3b3):
        print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A3']))
        RV0 = np.array([tr.RV_sec[0]])
        # pass secondary trajectory to detector
        tr.pass_sec(RV0, geomT15.r_dict['det'], E, B, geomT15,
                    stop_plane_n=geomT15.plates_dict['an'].det_plane_n,
                    tmax=9e-5, eps_xy=eps_xy, eps_z=eps_z)
        traj_list_an.append(tr)

# %% Additional plots
# hbplot.plot_grid_a3b3(traj_list_a3b3, geomT15, Btor, Ipl,
#                       marker_E='p')
# hbplot.plot_traj(traj_list_a3b3, geomT15, 500, 0.0, Btor, Ipl,
#                   full_primary=False, plot_analyzer=True,
#                   subplots_vertical=True, scale=3.5)
hbplot.plot_grid_simple(traj_list_a3b3, geomT15, Btor, Ipl)
hbplot.plot_sec_angles(traj_list_a3b3, Btor, Ipl,
                       linestyle='-o', Ebeam='all')
# for E in Ebeam: #  не работает
#     hbplot.plot_scan_one_subplot(traj_list_a3b3, geomT15, Ebeam, Btor, Ipl, plot_analyzer=True)
