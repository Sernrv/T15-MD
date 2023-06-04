# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:58:09 2023

@author: Krohalev_OD
"""

'''
primary hit invisible wall, r = [ 2.887 -0.43  -0.031]


# r
[ 2.898  0.658 -0.103], invisible_wall_x = 2.9, r_aim = [ 2.5  0.3 -1e-03]
[ 2.897  0.635 -0.21 ], invisible_wall_x = 2.9, r_aim = [ 2.5  0.3 -1e-03]
[ 2.899 -0.128 -0.124], invisible_wall_x = 2.9, r_aim = [ 2.5  0.3 -1e-03]

[ 2.898  0.764 -0.176], invisible_wall_x = 2.9, r_aim = [ 2.5e+00  3.0e-01 -1.0e-03], stop_n = [ 1.  0. -0.]
'''

# %%
xx = plt.gca().get_children()[0].get_xdata()
yy = plt.gca().get_children()[0].get_ydata()
zz = plt.gca().get_children()[1].get_ydata()

xx_pr = plt.gca().get_children()[2].get_xdata()
yy_pr = plt.gca().get_children()[2].get_ydata()
zz_pr = plt.gca().get_children()[3].get_ydata()

geom = geomT15

rv_prim = np.vstack((xx_pr, yy_pr, zz_pr)).T
mask = np.sqrt((rv_prim[:, 0] - geom.R) ** 2 +
               (rv_prim[:, 1] / geom.elon) ** 2) <= geom.r_plasma

# %%

rv0 = rrvv[200] + np.array([0.0001, 0.0, 0.0, 0.00, 0.0, 0.0])

# x = np.where(np.isclose(rrvv, rv0))
y = np.isclose(rrvv, rv0)
z = np.all(y, axis=1)
ii = np.where(z)
i = ii[0][0] if len(ii[0]) > 0 else None
print(i)


def argfind_rv(rrvv, rv):
    # x = np.where(np.isclose(rrvv, rv0))
    bb = np.all(np.isclose(rrvv, rv), axis=1)
    ii = np.where(bb)
    i = ii[0][0] if len(ii[0]) > 0 else None
    return i


# %%

plt.figure(123)
plt.plot(xx_pr, yy_pr)
plt.plot(xx, yy)

# %%

import hibpcalc.geomfunc as gf


def _plane_segment_intersect(planeNormal, planePoint, segmPoint0, segmPoint1, eps=1e-6):  # !!! to be tested
    '''
    function returns intersection segment between plane and ray
    '''
    segmVector = segmPoint1 - segmPoint0
    planeC = - planeNormal.dot(planePoint)

    up = planeNormal.dot(segmPoint0) + planeC
    dn = planeNormal.dot(segmVector)

    print(up, dn)
    if abs(dn) < eps:
        # return np.full_like(segmPoint0, np.nan) 
        return None, None

    t = - up / dn
    intersectLinePlane = segmVector * t + segmPoint0

    if 0.0 <= t <= 1.0:
        return intersectLinePlane, t
    else:
        return None, t


def _is_between(A, B, C, eps=1e-6):
    '''
    function returns True if point C is on the segment AB (between A and B)
    '''
    if np.isnan(C).any():
        return False
    # check if the points are on the same line
    crossprod = np.cross(B - A, C - A)
    if np.linalg.norm(crossprod) > eps:
        print("1")
        print(np.linalg.norm(crossprod))
        return False
    # check if the point is between
    dotprod = np.dot(B - A, C - A)
    if dotprod < 0 or dotprod > np.linalg.norm(B - A) ** 2:
        print("2")
        return False
    return True


stop_plane_n = np.array([1., 0., -0.])
r_aim = np.array([2.5e+00, 3.0e-01, -1.0e-03])

plt.plot(xx, yy)
plt.plot([r_aim[0], r_aim[0]], [r_aim[1] - 0.2, r_aim[1] + 0.2])

for i in range(len(xx) - 1):
    rv = np.array([xx[i], yy[i], zz[i]])
    rv_next = np.array([xx[i + 1], yy[i + 1], zz[i + 1]])

    r_intersect = gf.line_plane_intersect(stop_plane_n, r_aim, rv, rv_next)
    if gf.is_between(rv, rv_next, r_intersect):
        print(i)
        print("BINGO!")
        # break

    if gf.plane_segment_intersect(stop_plane_n, r_aim, rv, rv_next) is not None:
        print("bingo!")
        print(i)

        break

# 73-74
i = 73
rv = np.array([xx[i], yy[i], zz[i]])
rv_next = np.array([xx[i + 1], yy[i + 1], zz[i + 1]])

# rv = rv - np.array([0.000113, 0, 0])
# rv_next = rv_next - np.array([0.000113, 0, 0])

print(_plane_segment_intersect(stop_plane_n, r_aim, rv, rv_next, eps=1e-19))


# %% tozhe szhech'

def _rect(edges, indexes):
    ii0, ii1, ii2, ii3 = indexes
    return [edges[ii0], edges[ii1], edges[ii2], edges[ii3]]


if False:
    anzt = geomT15.plates_dict["an"]
    plA3 = geomT15.plates_dict["A3"]
    plB3 = geomT15.plates_dict["B3"]

    test_pl = anzt
    # test_pl = plA3
    # test_pl = plB3

    plt.figure(111)
    test_pl.plot(plt.gca())
    rct = test_pl.front_rect()

    c = gf.rect_center(rct)
    _v, _h = test_pl.front_basis()
    along = np.cross(_v, _h)
    gf.plot_segm(c, c + along)
    _a = along / np.linalg.norm(along)

    N = 100
    xx = np.zeros(N)
    EE_h = np.zeros_like(xx)
    EE_v = np.zeros_like(xx)
    EE_a = np.zeros_like(xx)
    for i in range(N):
        r = c + along * i / N * 2.0
        xx[i] = r[0]
        E_loc = fields.return_E(r, E, tr.U, geomT15)
        EE_v[i] = E_loc.dot(_v)
        EE_h[i] = E_loc.dot(_h)
        EE_a[i] = E_loc.dot(_a)
    plt.figure(222)
    plt.plot(xx, EE_v)
    plt.plot(xx, EE_h)
    # plt.plot(xx, EE_a)

# %% xlam
if False:
    _anzt = copy.deepcopy(anzt)

    dr = geomT15.r_dict['an']
    # get angles
    angles = copy.deepcopy(geomT15.plates_dict['an'].angles)
    beamline_angles = copy.deepcopy(geomT15.plates_dict['an'].beamline_angles)

    _anzt.shift(-dr)

    # rotate point to the coord system of plates
    _anzt.rotate(angles, beamline_angles, inverse=True)
    plt.figure(444)
    _anzt.plot(plt.gca(), 'ZY')

    xx0 = np.zeros(N)
    EE_h0 = np.zeros_like(xx0)
    EE_v0 = np.zeros_like(xx0)
    EE_a0 = np.zeros_like(xx0)
    for i in range(N):
        r = np.array([i / 100 - 0.1, 0.11, 0.0])
        xx0[i] = r[0]
        try:
            E_loc = np.array([E['an'][0](r), E['an'][1](r), E['an'][2](r)])
            EE_v0[i] = E_loc[1]
            EE_h0[i] = E_loc[2]
            EE_a0[i] = E_loc[0]
        except:
            break

    # plt.figure(333)
    plt.plot(xx0, EE_v0 * 0.0001)
    plt.plot(xx0, EE_h0 * 0.0001)

# %%
if False:
    r = geomT15.r_dict['aim']

    r_new = r - geomT15.r_dict['A3']
    # get angles
    angles = geomT15.plates_dict['A3'].angles
    beamline_angles = geomT15.plates_dict['A3'].beamline_angles
    # rotate point to the coord system of plates
    r_new = gf.rotate3(r_new, angles, beamline_angles, inverse=True)
    # interpolate Electric field

    print([interp(r_new) for interp in E['A3']])
    print(E_fast['A3'](r_new))

# %%
if False:
    domain = np.array([  # analyzer
        -0.42286543, 1.02463457,  # xmin, xmax
        -0.09185417, 0.30564583,  # ymin, ymax
        -0.2, 0.1975,  # zmin, zmax
        0.005  # delta
    ])

    x = np.arange(domain[0], domain[1], domain[6])  # + r_new[0]
    y = np.arange(domain[2], domain[3], domain[6])  # + r_new[1]
    z = np.arange(domain[4], domain[5], domain[6])  # + r_new[2]

    _dirname = r'D:\radrefs\T-15MD\current\elecfield\sec'
    _plts_name = 'an'
    _Ex = np.load(_dirname + '\\' + _plts_name + '_Ex.npy')
    _Ey = np.load(_dirname + '\\' + _plts_name + '_Ey.npy')
    _Ez = np.load(_dirname + '\\' + _plts_name + '_Ez.npy')

    _i = 144
    _k = 39
    _j = 40

    xx = np.zeros(80)
    yy = np.zeros(80)
    for j in range(80):
        xx[j] = y[j]
        yy[j] = _Ey[_i, j, _k] * 0.00001

    plt.figure(999)
    plt.plot(yy, xx)

    _i = 144
    _k = 40
    _j = 40

    xx = np.zeros(80)
    yy = np.zeros(80)
    for j in range(80):
        xx[j] = y[j]
        yy[j] = _Ey[_i, j, _k] * 0.00001

    plt.plot(yy, xx)

    _anzt.plot(plt.gca(), 'ZY')

# %% test posle prochteniya sjech'

# rect = front_rect(plA3.edges)
# #plot_rect(rect)
# v1 = rect[1] - rect[0]
# v2 = rect[-1] - rect[0]
# n = np.cross(v1, v2)
# _a, _b, _g = (plA3.angles * np.pi / 180.0)
# _n = np.array([np.cos(_a)*np.cos(_b), np.sin(_a), -np.cos(_a)*np.sin(_b)])
# print (np.cross(n, _n))
# plot_front_rect(plA3.edges); plA3.plot(plt.gca());
# for k in np.linspace(0.1, 0.9, 9): 
#    irect = inner_rect(rect, k) 
#    plot_rect(irect)

# aimA3 = geomT15.r_dict["aim"]
# _center = 0.25*(rect[0] + rect[1] + rect[2] + rect[3] )
# print(_center - aimA3) 
# geomT15.plot(plt.gca())


# %%
r = np.array([3.64718001, 2.10776035, -0.83657758])
r = np.array([1.65156143, 0.28965075, 0.50376969])
r = np.array([2.97347764, 2.10238695, -0.3984648])

print(B(r))

xx = [r[0] - 0.0001, r[0] + 0.0001]
yy = [r[1] - 0.0001, r[1] + 0.0001]
geomT15.plot(plt.gca())
plt.plot(xx, yy, 'o')

r_slit = geomT15.r_dict['slit']
slit_plane_n = geomT15.plates_dict['an'].slit_plane_n

(r - r_slit).dot(slit_plane_n)

rs = geomT15.r_dict['slit']
r4 = geomT15.r_dict['A4']
print(B(r4))
print(B(rs))

# %%

tr.plot_prim(plt.gca())
tr.plot_sec(plt.gca())

r_slit = geomT15.r_dict['slit']
slit_plane_n = geomT15.plates_dict['an'].slit_plane_n


def xRotateMx(ang):
    s = np.sin(ang)
    c = np.cos(ang)
    return np.matrix([[1.0, 0.0, 0.0],
                      [0.0, c, -s],
                      [0.0, s, c]], dtype=np.float64)


def yRotateMx(ang):
    s = np.sin(ang)
    c = np.cos(ang)
    return np.matrix([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]], dtype=np.float64)


def zRotateMx(ang):
    s = np.sin(ang)
    c = np.cos(ang)
    return np.matrix([[c, -s, 0.0],
                      [s, c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)


def transformPt(pt, mx):  # 2D, 3D
    return mx.dot(pt).A1


def zwardRotateMx(vec):  # vec -> (0, 0, L)
    # to transform any plane to XY plane
    v = vec
    a1 = np.arctan2(v[0], v[2])
    mx1 = yRotateMx(-a1)

    v = transformPt(v, mx1)
    a2 = np.arctan2(v[1], v[2])
    mx2 = xRotateMx(a2)

    return mx2.dot(mx1)


def _rect(r, plane_n, size):
    rect = [np.array([-size, -size, 0.0]),
            np.array([-size, size, 0.0]),
            np.array([size, size, 0.0]),
            np.array([size, -size, 0.0]),
            np.array([-size, -size, 0.0])]

    mx = zwardRotateMx(plane_n).I
    rect = [mx.dot(_r).A1 for _r in rect]
    rect = [_r + r for _r in rect]
    return rect


import hibpcalc.geomfunc as gf

r = _rect(r_slit, slit_plane_n, 0.1)
gf.plot_rect(r)
r = _rect(r_slit, slit_plane_n, 0.2)
gf.plot_rect(r)
r = _rect(r_slit, slit_plane_n, 0.3)
gf.plot_rect(r)

# %%

import hibpcalc.misc as misc

uu = np.linspace(-40.0, 40.0, 41)
dd = uu - 20.5
i, i1, t = misc.find_fork(dd)

print(i, t, dd[i], dd[i + 1])

# %%

r = geomT15.r_dict["A3"]
U_dict["A3"] = 22.0
E_ = fields.return_E(r, E, U_dict, geomT15)
print(E_)
E_ = fields._return_E(r, E_fast, U_dict, geomT15)
print(E_)

# %%

plt.figure(225);
geomT15.plot(plt.gca(), 'XY');
plt.figure(229);
geomT15.plot(plt.gca(), 'XZ')

# %%

# import hibpcalc.fields as fields
xx = np.linspace(0., 5., 1000)
rr = [np.array([x, 0., 0.]) for x in xx]

BB = [B(r) for r in rr]
BBx = np.zeros_like(xx)
for i in range(len(BB)):
    b = B(rr[i])
    if np.any(np.isnan(b)):
        BBx[i] = np.nan
    else:
        # BBx[i] = np.linalg.norm(b[0]) # b[0][1]
        BBx[i] = b[0][2]

# BBx = np.array([b[0] for b in BB])
plt.figure(200)
plt.plot(xx, BBx)
geomT15.plot(plt.gca())
plt.plot([0.0, 3.5], [0.0, 0.0])

rr = [np.array([x, 0.4, -0.2]) for x in xx]
BB = [B(r) for r in rr]
BBx = np.zeros_like(xx)
for i in range(len(BB)):
    b = B(rr[i])
    if np.any(np.isnan(b)):
        BBx[i] = np.nan
    else:
        # BBx[i] = np.linalg.norm(b[0]) # b[0][1]
        BBx[i] = b[0][2]

# BBx = np.array([b[0] for b in BB])
plt.figure(200)
plt.plot(xx, BBx)
geomT15.plot(plt.gca())
plt.plot([0.0, 3.5], [0.0, 0.0])


# yy = np.linspace(-1., 2., 1000)

# %%
def plot_B_stream(B, color='r', dens=1.0, plot_sep=True):
    '''
    stream plot of magnetic field
    '''

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    plot_geometry(ax1, plot_sep=plot_sep)

    xx = np.arange(B.volume_corner1[0], B.volume_corner2[0], B.resolution)
    yy = np.arange(B.volume_corner1[1], B.volume_corner2[1], B.resolution)
    zz = np.arange(B.volume_corner1[2], B.volume_corner2[2], B.resolution)

    BBxy =
    for x in xx:
        for y in yy:
            r = np.array([x, y, 0.])
            b = B(r)
            if np.any(np.isnan(b)):
                BBx[i] = np.nan
            else:
                BBx[i] = b[0][1]

    ax1.streamplot(x, y, Bx[:, :, z_cut].swapaxes(0, 1),
                   By[:, :, z_cut].swapaxes(0, 1), color=color, density=dens)
    ax2.streamplot(x, z, Bx[:, y_cut, :].swapaxes(0, 1),
                   Bz[:, y_cut, :].swapaxes(0, 1), color=color, density=dens)
    plt.show()


# %%

B.volume_corner1
Out[225]: [1.1, -1.1, -1.2]

B.volume_corner2
Out[226]: [5.42, 2.12, 0.52]


# %%

def plot_point(r, **kwargs):
    xx = [r[0]]
    yy = [r[1]]
    plt.plot(xx, yy, 'o', **kwargs)


def mark_E(z, color, thr):
    xx = np.linspace(2.0, 3.5, 100)
    yy = np.linspace(-0.5, 1.0, 100)
    for x in xx:
        for y in yy:
            z = 0.0
            r = np.array([x, y, z])
            E_ = return_E(r, E, U_dict_, geomT15)
            if np.linalg.norm(E_[0]) > thr:
                plot_point(r, color=color)


plt.figure(999)
geomT15.plot(plt.gca())
xx = np.linspace(2.0, 3.5, 100)
yy = np.linspace(-0.5, 1.0, 100)

# U_dict = {'A2': 50.0,
# 'B2': 9.247043072458366,
# 'A3': 22.0,
# 'B3': 0.0,
# 'A4': 0.0,
# 'an': 107.48837083315956}

U_dict_ = {'A2': 50.0,
           'B2': 9.247043072458366,
           'A3': 22.0,
           'B3': 0.0,
           'A4': 0.0,
           'an': 107.48837083315956}

mark_E(-0.1, 'red', 1000.0)
mark_E(0.0, 'blue', 1000.0)
mark_E(0.1, 'green', 1000.0)

# %% runcells
runcell('imports', 'D:/radrefs/T-15MD/current/HIBP-SOLVER.py')
runcell('Define Geometry', 'D:/radrefs/T-15MD/current/HIBP-SOLVER.py')
runcell('Load Electric Field', 'D:/radrefs/T-15MD/current/HIBP-SOLVER.py')
runcell('Analyzer parameters', 'D:/radrefs/T-15MD/current/HIBP-SOLVER.py')
runcell('Optimize Secondary Beamline', 'D:/radrefs/T-15MD/current/HIBP-SOLVER.py')

# %%
import hibpcalc.misc as misc

#                           xaim  yaim  zaim  alpha beta  gamma 

beamline1 = misc.SecBeamlineData(2.5, 0.2, 0.0, 15.0, 15.0, -20.0)
beamline2 = misc.SecBeamlineData(2.6, 0.0, zport_in, 30.0, 20.0, -20.0)

xaim, yaim, zaim, alpha_sec, beta_sec, gamma_sec = beamline1
print(xaim)
print(yaim)
print(zaim)
print(alpha_sec)
print(beta_sec)
print(gamma_sec)

# %%
viewer_path = r'C:\reonid\reonid-packages\hibp_lib'

from copy import deepcopy
import hibpcalc.misc as misc

if viewer_path not in sys.path: sys.path.append(viewer_path)
from myviewer import Viewer, ViewerAdapter
import optimizers

trajectories = deepcopy(traj_list_passed)

# opt_A3 = optimizers.Optimizer('zone',   2.0, 40.0, 'slit', geomT15, 'A3')
opt_A3 = optimizers.Optimizer('zone', 2.0, 40.0, 'A4', geomT15, 'A3')
opt_B3 = optimizers.Optimizer('center', 2.0, 40.0, 'slit', geomT15, 'B3')
opt_A4 = optimizers.Optimizer('center', 2.0, 40.0, 'slit', geomT15, 'A4')


# def _calc_

def _optimize(tr, optimizer, E, B, geom, RV0, plot_ok=False):
    uu = np.linspace(-40.0, 40.0, 41)
    dd = np.zeros_like(uu)
    # dd_v, dd_h, dd_x = np.zeros_like(uu), np.zeros_like(uu), np.zeros_like(uu)
    for i, u in enumerate(uu):
        print("------PASS SEC---------", i, u)
        optimizer.set_voltage(tr, u)
        # tr._pass_sec(RV0, optimizer.r_target, E, B, geom, stop_plane_n=optimizer.stop_plane_n,
        #             break_at_intersection=True)
        tr._pass_sec(RV0, optimizer.r_target, E, B, geom, stop_plane_n=optimizer.stop_plane_n)
        if True in tr.IntersectGeometrySec.values():
            dd[i] = np.nan  # optimizer.calc_delta(tr.RV_sec[-1]) # np.nan

        else:
            dd[i] = optimizer.calc_delta(tr.RV_sec[-1])

        if plot_ok:
            plt.figure(345)
            tr.plot_prim(plt.gca())
            tr.plot_sec(plt.gca())

            plt.figure(346)
            tr.plot_prim(plt.gca(), 'XZ')
            tr.plot_sec(plt.gca(), 'XZ')

            # dd[i] = _calc_delta(tr.RV_sec[-1])

    mask = np.isnan(dd)
    uu = uu[~mask]
    dd = dd[~mask]
    tr.uuu = uu
    tr.ddd = dd
    i, i1, t = misc.find_fork(dd, 0.01)
    if i is not None:
        u = uu[i] + t * (uu[i + 1] - uu[i])

        optimizer.set_voltage(tr, u)
        tr._pass_sec(RV0, optimizer.r_target, E, B, geom, stop_plane_n=optimizer.stop_plane_n)

        plt.figure(345)
        tr.plot_prim(plt.gca())
        tr.plot_sec(plt.gca())

        plt.figure(346)
        tr.plot_prim(plt.gca(), 'XZ')
        tr.plot_sec(plt.gca(), 'XZ')
    else:
        print('fail: ', tr.Ebeam, tr.U['A2'])


for traj in trajectories[10:14]:  # [65:66]:
    RV0_ = np.array([traj.RV_sec[0]])
    _optimize(traj, opt_A3, E, B, geomT15, RV0_, True)
    _optimize(traj, opt_B3, E, B, geomT15, RV0_, True)
    _optimize(traj, opt_A4, E, B, geomT15, RV0_, True)

plt.figure(345);
geomT15.plot(plt.gca())
# traj.plot_prim(plt.gca())
# traj.plot_sec(plt.gca())

plt.figure(347)
plt.plot(traj.uuu, traj.ddd, "o")

plt.figure(346);
geomT15.plot(plt.gca(), 'XZ')
# traj.plot_prim(plt.gca(), 'XZ')
# traj.plot_sec(plt.gca(), 'XZ')


_line_in_stop_plane = np.cross(opt_A3.stop_plane_n, np.array([0.0, 0.0, 1.0]))
_pt_in_stop_plane = opt_A3.r_target

xx = [_pt_in_stop_plane[0] - _line_in_stop_plane[0], _pt_in_stop_plane[0] + _line_in_stop_plane[0]]
yy = [_pt_in_stop_plane[1] - _line_in_stop_plane[1], _pt_in_stop_plane[1] + _line_in_stop_plane[1]]

plt.figure(345)
plt.plot(xx, yy)


# %%

def segm_rect_intersect(segm_p0, segm_p1, rect):
    center = gf.rect_center(rect)
    normal = np.cross(rect[1] - rect[0], rect[-1] - rect[0])
    ints = gf.plane_segment_intersect(center, normal, segm_p0, segm_p1)


class SecondaryBeamLineBox():
    def __init__(self, geom):
        self.geom = geom
        a3 = geom.plates_dict['A3']
        a3_front_rect = a3.front_rect()
        a3_basis = a3.front_basis()
        a3_axis = np.cross(a3_basis[0], a3_basis[1])

        self.axis = a3_axis
        self.basis = a3_basis
        self.front_rect = (a3_front_rect, k=3.0)
        self.back_rect = self.front_rect + a3_axis * 4.0  # ???

        self.side_rects = [_side_rect(0, 1), _side_rect(1, 2), _side_rect(2, 3), _side_rect(3, 4)]

    def intersect(self, r0, r1):
        return False

    def _side_rect(self, i0, i1):
        return [self.front_rect[i0], self.front_rect[i1], self.back_rect[i1], self.back_rect[i0]]

    def plot(self):
        gf.plot_rect(self.front_rect)
        gf.plot_rect(self.back_rect)


# %%

class B_ignore_plates:
    def __init__(self, B, E, geom, plates_to_ignore):
        self.B = B
        self.E = E
        self.geom = geom
        self.plates = plates_to_ignore

    def __call__(self, r):
        result = B(r)
        if self.inside_plates(r):
            return result * 0.0
        else:
            return result

    def inside_plates(self, r):
        for key in self.plates:
            # shift the center of coord system
            r_new = r - self.geom.r_dict[key]
            # get angles
            angles = copy.deepcopy(self.geom.plates_dict[key].angles)
            beamline_angles = copy.deepcopy(self.geom.plates_dict[key].beamline_angles)
            # rotate point to the coord system of plates
            r_new = gf.rotate3(r_new, angles, beamline_angles, inverse=True)
            # interpolate Electric field
            Etemp = np.zeros(3)
            try:
                Etemp[0] = self.E[key][0](r_new)
                return True
            except (ValueError, IndexError):
                continue

        else:
            return False


# %%

B_cut = B_ignore_plates(B, E, geomT15, ['A3', 'B3'])

for traj in trajectories[65:66]:
    RV0_ = np.array([traj.RV_sec[0]])
    _optimize(traj, opt_A3, E, B_cut, geomT15, RV0_, True)
    _optimize(traj, opt_B3, E, B_cut, geomT15, RV0_, False)
    _optimize(traj, opt_A4, E, B_cut, geomT15, RV0_, False)

# %%
plt.figure(345);
geomT15.plot(plt.gca())
plt.figure(346);
geomT15.plot(plt.gca(), 'XZ')

# %%
plt.figure(225);
geomT15.plot(plt.gca())
plt.figure(229);
geomT15.plot(plt.gca(), 'XZ')

# %%
# !!! HERE YOU CAN CATCH TRAJ HITTING FILAMENT
# Btor = 1., Ipl = 1., Ebeam = 220, UA2 = 26.5, r_aim = [2.5, 0., 0.]
# UB2 = 6.08 ?probably wrong?

import hibplib as hb
import hibpcalc.fields as fields
import hibpcalc.geomfunc as gf
import hibpcalc.misc as misc

# timestep [sec]
dt = 0.2e-7  # 0.7e-7
q = 1.602176634e-19  # electron charge [Co]
m_ion = 204.3833 * 1.6605e-27  # Tl ion mass [kg]
r0 = geomT15.r_dict['r0']  # trajectory starting point
Ebeam = 160.

# ???
alpha_aim = 0.
beta_aim = 0.
stop_plane_n = gf.calc_vector(1.0, alpha_aim, beta_aim)

U_dict = {'A2': UA2, 'B2': UB2,
          'A3': UA3, 'B3': UB3, 'A4': UA4, 'an': Ebeam / (2 * G)}
# create new trajectory object
tr = hb.Traj(q, m_ion, Ebeam, r0, geomT15.angles_dict['r0'][0],
             geomT15.angles_dict['r0'][1], U_dict, dt)

deltas = []
voltages = []

uu = np.linspace(-40., 40., 51)
uu = np.linspace(4., 8., 51)

# tr.U['A2'] = 26.5
tr.U['A2'] = 24.
tr.U['B2'] = 0.

# tr = hb._optimize_B2(tr, geomT15, 0., 0., E_slow, B_, dt, stop_plane_n,
#                    'aim_zshift', True, eps_xy=1e-3, eps_z=2.5e-2)

# 4%%

for UB2 in uu:
    tr.U['B2'] = UB2
    target = 'aim'
    tr.pass_prim(E_slow, B_, geomT15)
    r_aim = geomT15.r_dict['aim']
    tr.pass_fan(r_aim, E_slow, B_, geomT15, no_intersect=True, no_out_of_bounds=True,
                invisible_wall_x=geomT15.r_dict[target][0] + 0.1)
    tr._pass_to_target(r_aim, E_slow, B_, geomT15, no_intersect=True, no_out_of_bounds=True,
                       invisible_wall_x=geomT15.r_dict[target][0] + 0.1)

    if tr.IsAimXY:
        deltas.append(tr.RV_sec[-1, 2])
        voltages.append(UB2)

        # xx, yy, _ = tr.RV_prim[:, 0:3].T
        # plt.plot(xx, yy)
        try:
            plt.figure(111)
            tr.plot(color='b')
        except Exception as e:
            print(e)

i, i1, t = misc.find_fork(deltas, 0.02)
if i is not None:
    u_ok = voltages[i] + t * (voltages[i1] - voltages[i])
    tr.U['B2'] = u_ok
    tr.pass_prim(E_slow, B_, geomT15)
    tr.pass_fan(r_aim, E_slow, B_, geomT15, no_intersect=True, no_out_of_bounds=True,
                invisible_wall_x=geomT15.r_dict[target][0] + 0.1)
    tr._pass_to_target(r_aim, E_slow, B_, geomT15, no_intersect=True, no_out_of_bounds=True,
                       invisible_wall_x=geomT15.r_dict[target][0] + 0.1)
    tr.plot()

geomT15.plot(plt.gca())

plt.figure(21)
plt.plot(voltages, deltas)

# %% Cure the Magfield

# (216, 161, 86)
Fx, Fy, Fz = B_.list_Fx_Fy_Fz

# indexes_float = (point - self.volume_corner1)/self.res // 1

for j in range(161):
    # j = 161 // 2
    k = 20
    xx = np.arange(216) * B_.res + B_.volume_corner1[0]
    yy = Fx[:, j, k]
    plt.plot(xx, yy)

geomT15.plot(plt.gca())

xmask = np.abs(Fx) > 1.0
# Fx[xmask] = 0.0

ymask = np.abs(Fy) > 1.0
# Fy[ymask] = 0.0


# %%

wtf_r = np.array([1.81762, -0.09373, -0.23387])
xx = np.linspace(wtf_r[0] - 1, wtf_r[0] + 1, 10000)

xx = np.linspace(-5.0, 5.0, 10000)
y = wtf_r[1]
z = wtf_r[2]

bb = [B(np.array([x, y, z]))[0][0] for x in xx]
plt.plot(xx, bb)

bb = [B(np.array([x, y, z]))[0][1] for x in xx]
plt.plot(xx, bb)

bb = [B(np.array([x, y, z]))[0][2] for x in xx]
plt.plot(xx, bb)

np.sum(B.list_Fx_Fy_Fz[1] > 5.0)

# %%
dz2 = None
dz1 = 1.2

if (dz2 is None) or np.abs(dz1) < np.abs(dz2):
    print('OK')


# %%

def single_shot(tr, geom, E, B, dt, stop_plane_n, target='aim', eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10):
    r_aim = geom.r_dict[target]

    tr.dt1, tr.dt2 = dt, dt
    # pass fan of secondaries
    # !!! no_intersect=True, no_out_of_bounds=True
    tr.pass_fan(r_aim, E, B, geom, stop_plane_n=stop_plane_n, eps_xy=eps_xy, eps_z=eps_z, no_intersect=False,
                no_out_of_bounds=False, invisible_wall_x=geom.r_dict[target][0] + 0.1)
    # pass trajectory to the target
    tr._pass_to_target(r_aim, E, B, geom, stop_plane_n=stop_plane_n, eps_xy=eps_xy, eps_z=eps_z, dt_min=dt_min,
                       no_intersect=False, no_out_of_bounds=False, invisible_wall_x=geom.r_dict[target][0] + 0.1)

    if True in tr.IntersectGeometry.values():
        return None

    # if tr.IsAimXY
    dz = r_aim[2] - tr.RV_sec[-1, 2]  # ???
    return dz


def double_shot(tr, geom, UB2_pair, E, B, dt, stop_plane_n, target='aim', eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10):
    _u1, _u2 = UB2_pair
    tr.U['B2'] = _u1
    dz1 = single_shot(tr, geom, E, B, dt, stop_plane_n, target='aim', eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10)
    tr.U['B2'] = _u2
    dz2 = single_shot(tr, geom, E, B, dt, stop_plane_n, target='aim', eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10)

    if dz1 is None:
        return False, None
    if dz2 is None:
        return False, None

    if abs(dz1 - dz2) < 0.02:
        if abs(min(dz1, dz2)) < 0.05:
            return True, _u1
        else:
            return False, None

    ok = abs(min(dz1, dz2)) < 0.2

    # u_prognosis = _u1 + (_u2 - _u1)*dU_dz(u1, u2, dz1, dz2)
    u_prognosis = _u1 - dz1 / (dz2 - dz1) * (_u2 - _u1)
    return ok, u_prognosis


def _optimize_B2(tr, geom, UB2, dUB2, E, B, dt, stop_plane_n, target='aim',
                 optimize=True, eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10):
    '''
    get voltages on B2 plates and choose secondary trajectory
    which goes into target
    '''
    # set up target
    print('Target: ' + target)
    # r_aim = geom.r_dict[target]

    ok, u_progn = double_shot(tr, geom, (UB2 - 4.0, UB2 + 4.0), E, B, dt, stop_plane_n, target, eps_xy, eps_z, dt_min)

    if ok:
        tr.U['B2'] = u_progn
        single_shot(tr, geom, E, B, dt, stop_plane_n, target, eps_xy, eps_z, dt_min)
        return tr
    else:
        if u_progn is not None:
            ok, u_progn = double_shot(tr, geom, (u_progn - 4.0, u_progn + 4.0), E, B, dt, stop_plane_n, target, eps_xy,
                                      eps_z, dt_min)
            tr.U['B2'] = u_progn if u_progn is not None else UB2
            single_shot(tr, geom, E, B, dt, stop_plane_n, target, eps_xy, eps_z, dt_min)
            return tr
        else:
            # Brute force
            print('      !!! ----------------- u_progn is None --------------- !!!  ')
            u1, u2 = UB2 - 14.0, UB2 + 14.0
            ok1, u_progn1 = double_shot(tr, geom, (u1 - 4.0, u1 + 4.0), E, B, dt, stop_plane_n, target, eps_xy, eps_z,
                                        dt_min)
            ok2, u_progn2 = double_shot(tr, geom, (u2 - 4.0, u2 + 4.0), E, B, dt, stop_plane_n, target, eps_xy, eps_z,
                                        dt_min)

            tr.U['B2'] = u_progn1 if u_progn1 is not None else UB2
            dz1 = single_shot(tr, geom, E, B, dt, stop_plane_n, target, eps_xy, eps_z, dt_min)

            tr.U['B2'] = u_progn2 if u_progn is not None else UB2
            dz2 = single_shot(tr, geom, E, B, dt, stop_plane_n, target, eps_xy, eps_z, dt_min)

            if dz1 is None:
                return tr

            if (dz2 is None) or np.abs(dz1) < np.abs(dz2):
                tr.U['B2'] = u_progn1 if u_progn1 is not None else UB2
                dz1 = single_shot(tr, geom, E, B, dt, stop_plane_n, target, eps_xy, eps_z, dt_min)
            return tr


# %%
secondary
hit
invisible
wall, r = [2.7    1.961 - 0.575],
invisible_wall_x = 2.7, r_aim = [2.6 - 0.2  0.], stop_n = [1.  0. - 0.]

# %% #plot grid moving with aim

import hibpplotlib as hbplot
import hibplib as hb
import define_geometry as defgeom
import hibpcalc.fields as fields

indexes = np.array([0, 7, 14, 21])
markers = ['o', '<', 'p', '*']
geomT15 = defgeom.define_geometry(analyzer=analyzer, beamline_num=indexes[1])

E = {}
E_fast = {}
try:
    fields.read_plates('prim', geomT15, E, E_fast, hb.createplates)
    print('\n Primary Beamline loaded')
except FileNotFoundError:
    print('\n Primary Beamline NOT FOUND')

try:
    fields.read_plates('sec', geomT15, E, E_fast, hb.createplates)
    # add diafragm for A3 plates to Geometry
    hb.add_diafragm(geomT15, 'A3', 'A3d', diaf_width=0.05)
    hb.add_diafragm(geomT15, 'A4', 'A4d', diaf_width=0.05)
    print('\n Secondary Beamline loaded')
except FileNotFoundError:
    print('\n Secondary Beamline NOT FOUND')

traj_lists = {}
plt.figure(101)

for index, marker in zip(indexes, markers):
    name = 'beamline {}'.format(index)
    traj_list = []
    traj_list += hb.read_traj_list(name, dirname=r'output/B1_I1/1st attempt')
    traj_lists[index] = copy.deepcopy(traj_list)
    mask = [True if (tr.Ebeam == 140) or (tr.Ebeam == 260) or (tr.Ebeam == 400) else False for tr in traj_list]
    traj_list_cut = np.array(traj_list)[mask]
    hbplot.plot_grid_simple(traj_list_cut, geomT15, Btor, Ipl, onlyE=True, marker_A2='', marker_E=marker, ax=plt.gca())

plt.figure(101).set_size_inches(20, 12.5)
plt.figure(101).axes[0].set_xlim(1.0, 2.6)
plt.figure(101).axes[0].set_ylim(-0.5, 1.5)
# plt.figure(101).axes[0].set_label()
plt.figure(101).savefig(
    r'D:\YandexDisk\Курчатовский институт\Мои работы\Поворот первичного бимлайна на Т-15МД\Оптимизация точки пристрелки\Отчёт\evolution',
    dpi=300)

# %%

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# from colorspacious import cspace_converter

cmap = mpl.colormaps['jet']

resolution = 0.02  # 0.05  # 0.01    # [m]
disc_len = 0.1  # discretisation length for wire [m]
# xmin ymin zmin [m]
volume_corner1 = (1.1, -1.1, -1.2)
# volume_corner1 = (0.5, -3.0, -0.1)
# volume_corner1 = (0.9, -3.0, -0.02)
# xmax ymax zmax [m]
volume_corner2 = (5.4 + resolution, 2.1 + resolution, 0.5 + resolution)

_grid = np.mgrid[volume_corner1[0]:volume_corner2[0]:resolution,
        volume_corner1[1]:volume_corner2[1]:resolution,
        volume_corner1[2]:volume_corner2[2]:resolution]

_points = np.vstack(map(np.ravel, _grid)).T

# create list of grid points

# %%

import numpy as np
import T15_magfieldcalc as mc
import hibpcalc.geomfunc as gf

J_vals, x_vals, y_vals = mc.import_Jplasm(r'D:\radrefs\T-15MD\current\2MA_sn.txt')
J_max = np.max(J_vals)
_Jtot = np.sum(J_vals)  # total J, used for normalisation

if True:
    step_x, step_y = 3, 3
    J_vals = J_vals[step_x:J_vals.shape[0]:step_x, step_y:J_vals.shape[1]:step_y]
    x_vals = x_vals[step_x:x_vals.shape[0]:step_x]
    y_vals = y_vals[step_y:y_vals.shape[0]:step_y]

Jtot = np.sum(J_vals)
print('Jtot: ', _Jtot, Jtot)

for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        if J_vals[j, i] > 0.00001:
            ms = int(J_vals[j, i] / J_max * 10)
            c = cmap(J_vals[j, i] / J_max * 0.999)
            gf.plot_point([x, y], ms=ms, color=c)

geomT15.plot(plt.gca())
gf.plot_point([1.5, 0.0], marker='+', ms=15, color='r')

# %%

import wire

nfi = 40  # number of steps along toroidal angle
dfi = 2 * np.pi / nfi
fi = np.arange(0, 2 * np.pi + dfi, dfi)

disc_len = 0.1
CurrTot = 1.0  # MA

wires = []
strong_wire = None
# -> x cycle start
for i in range(x_vals.shape[0]):
    # single wire is a circle in xz plane
    # set wire radius as a value from x_vals
    x = np.sin(fi) * x_vals[i]
    z = np.cos(fi) * x_vals[i]
    # -> y cycle start
    for j in range(y_vals.shape[0]):
        if J_vals[j, i] != 0.0:
            # set y value
            y = np.full_like(x, y_vals[j])
            # concatenate in one np array
            new_coil = np.c_[x, y, z]
            # create new wire object for calculation by Biotsavart script
            new_w = wire.Wire(path=new_coil, discretization_length=disc_len,
                              current=1e6 * CurrTot * J_vals[j, i] / Jtot)
            if J_vals[j, i] > 1.725:
                strong_wire = new_w

            wires.append(new_w)

# for pt in wires[0].path: 
#     pt[1] = pt[2]
#     gf.plot_point(pt, ms=1)

# for pt in wires[0].discretized_path: 
#     pt[1] = pt[2] + 0.1
#     gf.plot_point(pt, ms=1)

# plt.plot(x, z)
# B = biot_savart(points, wires)
# return B, wires

# %%

# calc_Bpoint(r, IdL, r1):
# biot_savart(points, wires):

import T15_magfieldcalc as mc

mc.EPS_WIRE = 0.0006

ref = geomT15.r_dict['A3']
_rr = np.zeros((100, 3))
for i, r in enumerate(_rr):
    _rr[i, 0] = ref[0] + i * 0.1 - 3.0

w = wires[177]
w = strong_wire
_IdL, _r1 = w.IdL_r1
_BB = [mc.calc_Bpoint(r, _IdL, _r1) for r in _rr]

plt.plot(_rr[:, 0], [b[0] * 1.0e-6 for b in _BB])


def plot_circle(r0, radius):
    angles = np.linspace(0, 2.0 * np.pi, 100)

    xx = r0[0] + radius * np.sin(angles)
    yy = r0[1] + radius * np.cos(angles)
    plt.plot(xx, yy)


for pt in w.path:
    # pt[1] = pt[2]
    gf.plot_point(pt)
    # plot_circle(pt, 0.6)

plot_circle(w.path[10], mc.EPS_WIRE)
geomT15.plot(plt.gca())

# %% B on axis

_rr = np.zeros((100, 3))
for i, r in enumerate(_rr):
    _rr[i, 0] = ref[0] + i * 0.1 - 4.0

w = strong_wire
w_center = np.mean(w.path[0:-1], axis=0)
print(w_center)
_rr = np.zeros((200, 3))
for i, r in enumerate(_rr):
    j = i - 100
    _rr[i] = w_center + np.array([0.0, j * 0.1, 0.0])

_IdL, _r1 = w.IdL_r1
_BB = [mc.calc_Bpoint(r, _IdL, _r1) for r in _rr]

SI_mu0 = 4 * np.pi * 1e-7

norm = np.linalg.norm
# plt.plot(_rr[:, 1], [b[1]*1.0e-6 for b in _BB] )
# plt.plot(_rr[:, 1], [b[1]*1.0e-6 / 10 for b in _BB] )
plt.plot(_rr[:, 1], [b[1] * SI_mu0 / 4.0 / np.pi for b in _BB])

I_A = w.current
w_R = w.path[0].copy()
w_R[1] = 0
w_R = np.linalg.norm(w_R)
B_on_axis = SI_mu0 * I_A / 2.0 / w_R
print(B_on_axis)
plt.plot([-1, 1], [B_on_axis, B_on_axis])

print(w_center)


# %%

def plot_point_(pt, fig=None, ms=4, **kwargs):
    if fig is not None:
        plt.figure(fig)
    xx = [pt[0]]
    yy = [pt[1]]
    plt.plot(xx, yy, 'o', ms=ms, **kwargs)  # markersize


plot_point_(w_center, marker='+', ms=14)
for pt in w.path:
    # pt[1] = pt[2]
    plot_point_(pt)

for _r in _rr:
    plot_point_(_r)

# %%
B.plot(dens=5)
geomT15.plot(plt.figure(1).axes[0])


# %%

# _Bx_ = B_pl_[:, 0].reshape(grid.shape[1:])
# _By_ = B_pl_[:, 1].reshape(grid.shape[1:])
# _Bz_ = B_pl_[:, 2].reshape(grid.shape[1:])

def test_B_data(B, grid, coord=1, yshift=0.0, z_idx=None):
    coord_idx = {0: 0, 1: 1, 2: 2, 'x': 0, 'y': 1, 'z': 2}[coord]

    _Bc_ = B[:, coord_idx].reshape(grid.shape[1:])

    i = 108
    j = 80
    k = 43 if z_idx is None else z_idx

    for j in range(grid.shape[2]):  # !!! SIC! 2 -> y
        # for j in range(55, 56): # !!! SIC! 2 -> y
        yy = _Bc_[:, j, k]
        xx = grid[0, :, j, k]
        if yshift == 0.0:
            plt.plot(xx, yy)
        else:
            y = grid[1, i, j, k]  # y [-1, 2.1]
            plt.plot(xx, yy + y * yshift)

        # test_B_data(B_pl_, grid, 'y', 2)


r'D:\radrefs\T-15MD\current\magfield'
B_pl_old = np.load(r'D:\radrefs\T-15MD\current\magfield\magfieldPlasm_1MA_sn.npy')
B_pl_new = np.load(r'D:\radrefs\T-15MD\current\extra_magfields\magfield_new\magfieldPlasm_1MA_sn.npy')

# B_pl_005 = np.load(r'D:\radrefs\T-15MD\current\extra_magfields\magfield_new\magfieldPlasm_1MA_sn_005.npy')
# B_pl_06 = np.load(r'D:\radrefs\T-15MD\current\extra_magfields\magfield_new\magfieldPlasm_1MA_sn_06.npy')

# %%

with open(r'D:\radrefs\T-15MD\current\extra_magfields\magfield_new\geometry.dat', 'r') as f:
    volume_corner1 = [float(i) for i in f.readline().split()[0:3]]
    volume_corner2 = [float(i) for i in f.readline().split()[0:3]]
    resolution = float(f.readline().split()[0])

# create grid of points
grid = np.mgrid[volume_corner1[0]:volume_corner2[0]:resolution,
       volume_corner1[1]:volume_corner2[1]:resolution,
       volume_corner1[2]:volume_corner2[2]:resolution]


# %%

def interp_from_array(B, grid):
    Bx = B[:, 0].reshape(grid.shape[1:])
    By = B[:, 1].reshape(grid.shape[1:])
    Bz = B[:, 2].reshape(grid.shape[1:])
    return f.FieldInterpolator(grid, [Bx, By, Bz])


probing_line = gf.line_array(gf.pt3D(0.0, 0.050, 0.0), gf.pt3D(5.0, 0.050, 0.0), 1000)

B_old = interp_from_array(B_pl_old, grid)
B_new = interp_from_array(B_pl_new, grid)

B_005 = interp_from_array(B_pl_005, grid)
B_06 = interp_from_array(B_pl_06, grid)

bb_old = np.array([B_old(r)[0] for r in probing_line])
bb_new = np.array([B_new(r)[0] for r in probing_line])

bb_005 = np.array([B_005(r)[0] for r in probing_line])
bb_06 = np.array([B_06(r)[0] for r in probing_line])

plt.figure(243)

plt.plot(probing_line[:, 0], bb_old[:, 1])
plt.plot(probing_line[:, 0], bb_new[:, 1])

# plt.plot( probing_line[:, 0], bb_005[:, 1] )
# plt.plot( probing_line[:, 0], bb_06[:, 1] )

# %%
plt.figure(1)
test_B_data(B_pl_, grid, 'y', 0.0)
plt.figure(2)
test_B_data(B_pl_old, grid, 'y', 0.0)

# %%
I_A = 1e6
R_tok = 1.5
B_on_axis = SI_mu0 * I_A / 2.0 / R_tok
print(B_on_axis)


def Bfield_of_wire(r, wire):
    _IdL, _r1 = w.IdL_r1
    _b = calc_Bpoint(r, _IdL, _r1)
    return _b * 1e-7


def Bfield_of_wire_list(r, wire_list):
    _b = np.zeros(3)

    for w in wires_pl:
        _IdL, _r1 = w.IdL_r1
        _b += calc_Bpoint(r, _IdL, _r1)
    return _b * 1e-7


def Bfield_of_wire_list(r, wire_list):
    return biot_savart(r.reshape(1, 3), wire_list)


print(_BB[1] * 1e-7)

__BB = biot_savart(r.reshape(1, 3), wires_pl)
print(__BB[0][1])

# %%
from hibpcalc.geomfunc import ray_segm_intersect_2d
import matplotlib as mpl

cmap = mpl.colormaps['jet']
sep = copy.deepcopy(geomT15.sep)
sep[:, 0] += geomT15.R
xx = np.linspace(0., 3., 10)
yy = np.linspace(-1., 2., 10)
rhos = []
for x in xx:
    for y in yy:
        for segm_0, segm_1 in zip(sep[:-1], sep[1:]):
            rho = ray_segm_intersect_2d((np.array([1.5, 0.]), np.array([x, y])), (segm_0, segm_1))
            if rho is not None:
                rhos.append(rho)
                break
        c = cmap(rho * 0.5)
        gf.plot_point([x, y], color=c)

geomT15.plot(plt.gca())

# %%
import hibplib as hb

hb.calc_radrefs_2d_separatrix(traj_list_passed, geomT15)

for tr in traj_list_passed:
    plt.figure(111)
    c = cmap(tr.rho * 0.999)
    gf.plot_point(tr.RV_sec[0, 0:2], color=c)
    plt.figure(222)
    c = cmap(tr.theta * 0.5)
    gf.plot_point(tr.RV_sec[0, 0:2], color=c)

plt.figure(111)
geomT15.plot(plt.gca())
plt.figure(222)
geomT15.plot(plt.gca())

# %%
import freegs
