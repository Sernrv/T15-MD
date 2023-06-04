'''
Define main parameters of T-15MD HIBP geometry
'''
import numpy as np
import hibplib as hb
import hibpcalc.fields as fields
from hibpcalc.misc import SecBeamlineData


# %%
def define_geometry(analyzer=1, beamline_num=0):
    '''

    Parameters
    ----------
    analyzer : int, optional
        number of the HIBP Analyzer. The default is 1.

    Returns
    -------
    geom : Geometry
        Geometry object with a proper configuration.

    '''
    geom = hb.Geometry()

    # plasma parameters
    geom.R = 1.5  # tokamak major radius [m]
    geom.r_plasma = 0.7  # plasma minor radius [m]
    geom.elon = 1.8  # plasma elongation

    # PRIMARY beamline geometry
    # alpha and beta angles of the PRIMARY beamline [deg]
    alpha_prim = 34.  # 20.  # 30.
    beta_prim = -10.  # -11
    gamma_prim = 0.
    prim_angles = {'r0': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'B2': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'A2': np.array([alpha_prim, beta_prim, gamma_prim])}
    geom.angles_dict.update(prim_angles)

    # coordinates of the injection port [m]
    xport_in = 2.136  # 1.5 + 0.726
    yport_in = 1.189  # 1.064
    zport_in = 0.0  # 0.019
    geom.r_dict['port_in'] = np.array([xport_in, yport_in, zport_in])

    # distance from the injection port to the Alpha2 plates
    dist_A2 = 0.3  # [m]
    # distance from Alpha2 plates to the Beta2 plates
    dist_B2 = 0.4  # [m]
    # distance from Beta2 plates to the initial point of the traj [m]
    dist_r0 = 0.2

    # coordinates of the center of the ALPHA2 plates
    geom.add_coords('A2', 'port_in', dist_A2, geom.angles_dict['A2'])
    # coordinates of the center of the BETA2 plates
    geom.add_coords('B2', 'A2', dist_B2, geom.angles_dict['B2'])
    # coordinates of the initial point of the trajectory [m]
    geom.add_coords('r0', 'B2', dist_r0, geom.angles_dict['r0'])

    # %% beamlines
    #                            xaim   yaim    zaim    alpha  beta   gamma
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

    beamline1 = beamlines_1an[beamline_num]

    #                           xaim  yaim   zaim    alpha beta  gamma 
    # beamline1 = SecBeamlineData(2.5,  -0.2,   0.0,    15.0, 15.0, -20.0)
    beamline2 = SecBeamlineData(2.6, 0.0, zport_in, 30.0, 20.0, -20.0)

    # !!!
    # beamline1 = SecBeamlineData(2.5,  0.0,   0.0,    20.0, 15.0, -20.0)
    # beamline_debug = SecBeamlineData(2.5,  0.0, 0.0, 15.0, 15.0, -20.0)
    # beamline1 = beamline_debug

    # %%
    # AIM position (BEFORE the Secondary beamline) [m]
    if analyzer == 1:
        beamline = beamline1
        xaim, yaim, zaim, alpha_sec, beta_sec, gamma_sec = beamline
        A3_angles = np.array([alpha_sec, beta_sec, gamma_sec])
    elif analyzer == 2:
        beamline = beamline2
        xaim, yaim, zaim, alpha_sec, beta_sec, gamma_sec = beamline
        # in the second line U_lower_plate=0
        A3_angles = np.array([alpha_sec, beta_sec, gamma_sec + 180.])

    r_aim = np.array([xaim, yaim, zaim])
    geom.r_dict['aim'] = r_aim
    geom.r_dict['aim_zshift'] = r_aim  # + np.array([0., 0., 0.03])

    # SECONDARY beamline geometry
    sec_angles = {'A3': A3_angles,
                  'B3': np.array([alpha_sec, beta_sec, gamma_sec]),
                  'A4': np.array([alpha_sec, beta_sec, gamma_sec]),
                  'an': np.array([alpha_sec, beta_sec, gamma_sec])}
    geom.angles_dict.update(sec_angles)

    # distance from r_aim to the Alpha3 center
    dist_A3 = 0.2  # 0.3  # 1/2 of plates length
    # distance from Alpha3 to the Beta3 center
    dist_B3 = 0.5  # + 0.6
    # from Beta3 to Alpha4
    dist_A4 = 0.5
    # distance from Alpha4 to the entrance slit of the analyzer
    dist_s = 0.5

    # coordinates of the center of the ALPHA3 plates
    geom.add_coords('A3', 'aim', dist_A3, geom.angles_dict['A3'])
    # coordinates of the center of the BETA3 plates
    geom.add_coords('B3', 'A3', dist_B3, geom.angles_dict['B3'])
    geom.add_coords('aimB3', 'A3', dist_B3 - 0.2, geom.angles_dict['A3'])
    # coordinates of the center of the ALPHA4 plates
    geom.add_coords('A4', 'B3', dist_A4, geom.angles_dict['A4'])
    geom.add_coords('aimA4', 'B3', dist_A4 - 0.2, geom.angles_dict['A4'])
    # Coordinates of the CENTRAL slit
    geom.add_coords('slit', 'A4', dist_s, geom.angles_dict['an'])
    # Coordinates of the ANALYZER
    geom.add_coords('an', 'A4', dist_s, geom.angles_dict['an'])

    # print info
    print('\nDefining geometry for Analyzer #{}'.format(analyzer))
    print('\nPrimary beamline angles: ', geom.angles_dict['r0'])
    print('Secondary beamline angles: ', geom.angles_dict['A3'])
    print('r0 = ', np.round(geom.r_dict['r0'], 3))
    print('r_aim = ', np.round(geom.r_dict['aim'], 3))
    print('r_slit = ', np.round(geom.r_dict['slit'], 3))

    # TOKAMAK GEOMETRY
    # chamber entrance and exit coordinates
    geom.chamb_ent = [(1.87, 1.152), (1.675, 1.434), (2.358, 0.445), (1.995, 0.970)]
    geom.chamb_ext = [(2.39, -0.44), (2.39, -2.0), (2.39, 0.44), (2.39, 0.8)]

    # Toroidal Field coil
    geom.coil = np.loadtxt('TFCoil.dat') / 1000  # [m]
    # Poloidal Field coils
    geom.pf_coils = fields.import_PFcoils('PFCoils.dat')
    # Camera contour
    geom.camera = np.loadtxt('T15_vessel.txt') / 1000
    # Separatrix contour
    geom.sep = np.loadtxt('T15_sep.txt') / 1000
    # First wall innner and outer contours
    geom.in_fw = np.loadtxt('infw.txt') / 1000  # [m]
    geom.out_fw = np.loadtxt('outfw.txt') / 1000  # [m]

    return geom
