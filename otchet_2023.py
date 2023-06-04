# -*- coding: utf-8 -*-
'''
T-15MD tokamak, HIBP

Program calculates trajectories and selects voltages on
primary beamline (B2 plates) and secondary beamline (A3, B3, A4 plates)
'''
# %% imports
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


# def linear_at_zero(x):
#    x = x*2.0
#    if abs(x) > 0.5: 
#        return 0.5*np.sign(x)
#    else: 
#        return x

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


# !!! For otchet only
cut_grid_E = {0: 380, 1: 380, 2: 380, 3: 380, 4: 380, 5: 380, 6: 380,
              7: 380, 8: 380, 9: 380, 10: 320, 11: 280, 12: 220, 13: 200,
              14: 380, 15: 380, 16: 380, 17: 340, 18: 300, 19: 260, 20: 220,
              21: 380, 22: 380, 23: 380, 24: 360, 25: 320, 26: 280, 27: 240,
              28: 380, 29: 320, 30: 320, 31: 280,
              32: 340, 33: 320, 34: 300, 35: 260
              }  # cut bad high E

UA2_cut_needed = {0: False, 1: False, 2: True, 3: False, 4: False, 5: False, 6: False,
                  7: False, 8: False, 9: True, 10: False, 11: False, 12: False, 13: True,
                  14: False, 15: False, 16: True, 17: True, 18: True, 19: True, 20: True,
                  21: False, 22: False, 23: True, 24: True, 25: True, 26: True, 27: True
                  }

cut_grid_UA2 = {2: {380.: [42.]},
                9: {380.: [40.]},
                13: {180.: [20.], 200.: [24.]},
                16: {380.: [34., 36.]},
                17: {320.: [30.], 340.: [34., 36.]}, 18: {280.: [28], 300.: [32., 34.]},
                19: {220.: [20.]},
                20: {200.: [20., 22.], 220.: [24., 26.]},
                23: {380.: [30., 32.]},
                24: {280.: [16.], 320.: [26.], 340.: [30., 32.], 360.: [32., 34., 36.]},
                25: {280.: [24.], 300.: [28., 30.], 320.: [30., 32., 34.]},
                26: {240.: [20., 22.], 260.: [24., 26.], 280.: [28., 30., 32.]},
                27: {200.: [18.], 220: [20., 22.], 240: [24., 26., 28.]},
                28: {360.: [22.], 380.: [26., 28.]},
                30: {260.: [16.], 280.: [20.], 300.: [24., 26.], 320.: [26., 28., 30.]},
                31: {240.: [16.], 260.: [20., 22.], 280.: [24., 26., 28.]},
                34: {260.: [12.], 280.: [16.], 300.: [20.]},
                35: {240.: [14.], 260.: [16., 18.]}
                }


def cut_UA2(traj_list, cut_grid_UA2=cut_grid_UA2, beamline_num=0):
    # mask_UA2=[]
    # prev_E = 0
    # prev_UA2 = 0
    # traj_list_cut = []
    # for tr in traj_list:
    #     if not np.isclose(tr.Ebeam, prev_E):
    #         prev_E = tr.Ebeam
    #         prev_UA2=tr.U['A2']
    #     if abs(tr.U['A2'] - prev_UA2) > 2.0:
    #         # mask_UA2.append(False)
    #         pass
    #     else:
    #         # mask_UA2.append(True)
    #         traj_list_cut.append(tr)
    #         prev_E = tr.Ebeam
    #         prev_UA2=tr.U['A2']    

    traj_list_cut = []
    for tr in traj_list:
        if beamline_num in cut_grid_UA2.keys():
            if tr.Ebeam not in cut_grid_UA2[beamline_num].keys():
                traj_list_cut.append(tr)
            else:
                if any(np.isclose(tr.U['A2'], cut_grid_UA2[beamline_num][tr.Ebeam])):
                    pass
                else:
                    traj_list_cut.append(tr)
        else:
            traj_list_cut.append(tr)

    return traj_list_cut


def cut_E(traj_list, cut_grid_E=cut_grid_E, beamline_num=0):
    mask_E = [False if tr.Ebeam > cut_grid_E[beamline_num] else True for tr in traj_list]
    traj_list = copy.deepcopy(np.array(traj_list)[mask_E])
    return traj_list


def cut(traj_list, beamline_num=0):
    traj_list = cut_E(traj_list, beamline_num=beamline_num)
    traj_list = cut_UA2(traj_list, beamline_num=beamline_num)
    return traj_list


# !!!
# %% set flags
optimizeB2 = True
optimizeA3B3 = True
calculate_zones = False
pass2AN = False
save_radref = False
save_primary = False
pass2aim_only = True
load_traj_from_file = True
save_grids_and_angles = False
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

                 SecBeamlineData(2.9, -0.1, 0.0, 29.5, 23.5, -20.0),  # 28
                 SecBeamlineData(2.9, 0.0, 0.0, 37.5, 25.5, -20.0),  # 29
                 SecBeamlineData(2.9, 0.1, 0.0, 42.5, 28.5, -20.0),  # 30
                 SecBeamlineData(2.9, 0.2, 0.0, 47.5, 31.5, -20.0),  # 31

                 SecBeamlineData(3.0, -0.1, 0.0, 29.5, 23.5, -20.0),  # 32
                 SecBeamlineData(3.0, 0.0, 0.0, 37.0, 25.0, -20.0),  # 33
                 SecBeamlineData(3.0, 0.1, 0.0, 40.0, 28.0, -20.0),  # 34
                 SecBeamlineData(3.0, 0.2, 0.0, 44.5, 31.0, -20.0),  # 35
                 ]

if multiple_beamlines:
    start = 0
    diapason = 36
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
energies = {'work': (40., 500., 20.), 'test': (300., 300., 20.)}

Emin, Emax, dEbeam = energies[regime]

# %% beamline loop
primary_grids = {}
radrefs = {}

for beamline_num in beamline_indexes:

    # %%set paths
    results_folder = r"D:\YandexDisk\Курчатовский институт\Мои работы\Поворот первичного бимлайна на Т-15МД\Оптимизация точки пристрелки\Отчёт\sec\x=" + str(
        beamlines_1an[beamline_num].xaim)
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
        hb.add_diafragm(geomT15, 'A3', 'A3d', diaf_width=0.05)
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

    # %% load trajectory list for further optimization
    if load_traj_from_file:
        traj_list = []
        for name in traj2load:
            traj_list += hb.read_traj_list(name, dirname=r'output/sec/B1_I1/')
        traj_list_passed = copy.deepcopy(traj_list)
        eps_xy, eps_z = 1e-3, 1e-3

    # %% calc radrefs
    hb.calc_radrefs_2d_separatrix(traj_list_passed, geomT15)

    for tr in traj_list_passed:
        if tr.theta > np.pi:
            tr.theta -= 2 * np.pi

    rhos = [tr.rho for tr in traj_list_passed]
    thetas = np.array([tr.theta for tr in traj_list_passed])
    mask_07 = [np.isclose(tr.rho, 0.7, atol=0.1) for tr in traj_list_passed]
    d_rho = (min(rhos), max(rhos))
    d_theta = (min(thetas), max(thetas))
    d_theta_07 = (min(thetas[mask_07]), max(thetas[mask_07]))

    radrefs[beamline_num] = (d_rho, d_theta, d_theta_07)

    # %% cut
    # traj_list_passed = cut(traj_list_passed, beamline_num=beamline_num)
    # %% Additional plots
    if save_primary:
        hb.save_traj_list(traj_list_passed, Btor, Ipl, beamline_num)

    if save_grids_and_angles:
        hbplot.plot_grid_simple(traj_list_passed, geomT15, Btor, Ipl,
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
            figs[0].axes[0].set_xlim(1.0, 4.6)
            figs[0].axes[0].set_ylim(-0.5, 2.0)
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

    # !!!
    # hbplot.plot_grid_simple(traj_list_passed, geomT15, Btor, Ipl,
    #                   onlyE=False, marker_A2='*')
