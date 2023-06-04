# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:15:57 2023

@author: reonid
"""

# import sys
# hibplib_path = 'D:\\reonid\\myPy\\reonid-packages'
# if hibplib_path not in sys.path: sys.path.append(hibplib_path)

import numpy as np
import scipy.special as sp

import matplotlib.pyplot as plt
import math as math
# import types as types

# import geomfunc as gf
from .geomfunc import invMx, stdRotateMx, vNorm, normalized_vector, pt3D, vec3D, line_array

SI_mu0 = 4 * np.pi * 1e-7

sin = np.sin
cos = np.cos  # ??? math
atan2 = math.atan2


def _magneticFdOfRing(a, J, r, z):  # ??? B*1e6 ???
    '''
    Magnetic field of infinitely thin ring
    a - radius [m]
    (r, phi, z) - cillindrical coordinates
    J current [A]
    Landau, Lifschitz VIII, chapter IV (p.164, edition 1982)
    '''
    aa = a * a  # a**2
    zz = z * z  # z**2
    if r == 0.0:  # ???  if np.isclose(r, 0.0):
        return [0.0, 0.0, J * np.pi / 5 * aa * (aa + zz) ** -1.5]

    rr = r * r  # r**2
    aplus = (a + r) ** 2 + zz
    aminus = (a - r) ** 2 + zz
    rrzz = rr + zz
    sqrtaplus = aplus ** 0.5
    arg = 4.0 * a * r / aplus
    K = sp.ellipk(arg)  # ellipticK(arg)
    E = sp.ellipe(arg)  # ellipticE(arg)
    E_aminus = E / aminus
    J1 = J * 0.2
    Hr = J1 * z * ((aa + rrzz) * E_aminus - K) / (r * sqrtaplus)
    Hz = J1 * (K + (aa - rrzz) * E_aminus) / sqrtaplus
    return [Hr, 0.0, Hz]


def _magneticFdOfThickRing(a, J, r, z, wire_radius):  # ??? B*1e6 ???
    '''
    Magnetic field of thick ring 
    a - radius [m]
    (r, phi, z) - cillindrical coordinates
    J current [A]
    wire_radius - thickness [m]
    Modified version of the [ Landau, Lifschitz VIII, chapter IV ]
    '''
    aa = a * a  # a**2
    zz = z * z  # z**2
    if r == 0.0:  # ???  if np.isclose(r, 0.0):
        return [0.0, 0.0, J * np.pi / 5 * aa * (aa + zz) ** -1.5]

    rr = r * r  # r**2
    aplus = (a + r) ** 2 + zz
    aminus = (a - r) ** 2 + zz
    rrzz = rr + zz
    sqrtaplus = aplus ** 0.5
    # sqrtaminus = aminus**0.5  # !!! distance from wire
    arg = 4.0 * a * r / aplus
    K = sp.ellipk(arg)  # ellipticK(arg)
    E = sp.ellipe(arg)  # ellipticE(arg)
    E_aminus = E / aminus
    J1 = J * 0.2
    Hr = J1 * z * ((aa + rrzz) * E_aminus - K) / (r * sqrtaplus)
    Hz = J1 * ((aa - rrzz) * E_aminus + K) / sqrtaplus

    #    dist = aminus**0.5
    #    if dist < wire_radius*0.001:
    #        return [0.0, 0.0, 0.0]
    #    elif dist <= wire_radius:
    #        Hr *= (dist/wire_radius)**2
    #        Hz *= (dist/wire_radius)**2

    wr2 = wire_radius * wire_radius
    if aminus < wr2 * 0.0001:  # !!!
        return [0.0, 0.0, 0.0]
    elif aminus < wr2:
        k = aminus / wr2
        Hr *= k
        Hz *= k

    return [Hr, 0.0, Hz]


def magneticFdOfRing(pt, J, center, radius, normal):
    mx = stdRotateMx(normal)
    i_mx = invMx(mx)
    pt0 = pt - center
    pt0 = mx.dot(pt0)

    r = vNorm(pt0[0:2])  # r = (pt0[0]**2 + pt0[1]**2)**0.5  # math.hypot(X, Y)
    z = pt0[2]

    phi = atan2(pt0[1], pt0[0])  # phi = math.atan2(Y, X)
    Hr, _, Hz = _magneticFdOfRing(radius, J, r, z)

    result = vec3D(Hr * np.cos(phi), Hr * np.sin(phi), Hz)
    result = i_mx.dot(result)  # g3.transformPt(result, mx.I)
    result += center  # g3.translatePt(result, center)

    return result


def magneticFdOfThickRing(pt, J, center, radius, mx, mx_inv, wire_radius):  # ??? B*1e6 ???
    # mx = g3.stdRotateMx(normal)
    pt0 = pt - center  # pt0 = g3.translatePt(pt, -center)
    pt0 = mx.dot(pt0)  # pt0 = g3.transformPt(pt0, mx)

    r = vNorm(pt0[0:2])  # r = (pt0[0]**2 + pt0[1]**2)**0.5  # math.hypot(X, Y)
    z = pt0[2]

    phi = atan2(pt0[1], pt0[0])  # phi = math.atan2(Y, X)
    Hr, _, Hz = _magneticFdOfThickRing(radius, J, r, z, wire_radius)

    result = np.array([Hr * cos(phi), Hr * sin(phi), Hz])
    result = mx_inv.dot(result)  # result = g3.transformPt(result, mx.I)
    result += center  # result = g3.translatePt(result, center)

    return result


def magneticFdOnRingCenter(J, r):  # B
    return J / 2.0 / r * SI_mu0


def _regularPolygon3D(nPoints, center, radius, normal):
    i_std_mx = invMx(stdRotateMx(normal))

    pts = []
    for i in range(nPoints):
        ang = 2.0 * np.pi * i / nPoints
        pt = pt3D(radius * sin(ang), radius * cos(ang), 0.0)
        pt = i_std_mx.dot(pt) + center
        pts.append(pt)

    return pts


class Ring:
    def __init__(self, center, radius, normal, I, wire_radius):
        self.I = I
        self.center = center
        self.radius = radius
        self.normal = normal
        self.wire_radius = wire_radius
        self.std_mx = stdRotateMx(self.normal)
        self.std_mx_inv = invMx(self.std_mx)

        self.std_mx = np.array(self.std_mx)
        self.std_mx_inv = np.array(self.std_mx_inv)

    def calcB(self, r):
        # return -1e-6*magneticFdOfRing(r, self.I, self.center, self.radius, self.normal)  # ???
        return -1e-6 * magneticFdOfThickRing(r, self.I, self.center, self.radius,
                                             self.std_mx, self.std_mx_inv, self.wire_radius)  # ???

    def distance(self, r):
        if all(np.isclose(self.center, r, 1e-6)):
            return self.radius

        norm = normalized_vector(self.normal)
        r_c = r - self.center
        auxilliary_plane_n = normalized_vector(np.cross(r_c, norm))
        dir_on_r_on_ring_plane = np.cross(norm, auxilliary_plane_n)
        h = r_c.dot(norm)
        y = r_c.dot(dir_on_r_on_ring_plane) - self.radius
        d = (h * h + y * y) ** 0.5
        return d

    def plot(self, *args, **kwargs):
        pts = _regularPolygon3D(100, self.center, self.radius, self.normal)
        xx = [p[0] for p in pts]
        yy = [p[1] for p in pts]
        xx.append(xx[0])
        yy.append(yy[0])
        plt.plot(xx, yy, *args, **kwargs)


class FilamentGroup:
    def __init__(self):
        self.elems = []

    def add(self, elem):
        self.elems.append(elem)

    def calcB(self, r):
        b = np.zeros(3)
        for e in self.elems:
            b += e.calcB(r)
        return b

    def plot(self, *args, **kwargs):
        for e in self.elems:
            e.plot(*args, **kwargs)


if __name__ == "__main__":
    probing_line = line_array(pt3D(-2.0, 0.0, 0.0), pt3D(2.0, 0.0, 0.0), 1000)
    ring = Ring(center=pt3D(0, 0, 0), radius=1.0, normal=vec3D(0, 0, 1), I=1.0, wire_radius=0.05)
    bb = np.array([ring.calcB(r) for r in probing_line])
    plt.plot(probing_line[:, 0], bb[:, 2])
