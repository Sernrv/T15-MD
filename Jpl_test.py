# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:40:15 2023

@author: reonid
"""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import hibpcalc.geomfunc as gf

import matplotlib as mpl

# cmap = mpl.colormaps['jet']
cmap = mpl.cm.jet

from T15_magfieldcalc import import_Jplasm
from hibpcalc.geomfunc import vec3D, pt3D
from hibpcalc.ring import Ring, FilamentGroup


class TokameqFile:
    def __init__(self, Jnominal, filename):
        self.Jnominal = Jnominal
        self.filename = filename

        self.J_vals, self.x_vals, self.y_vals = import_Jplasm(filename)
        self.J_max = np.max(self.J_vals)
        self.Jtot = np.sum(self.J_vals)

    def _resample(self, n):
        step_x, step_y = n, n
        self.J_vals = self.J_vals[step_x:self.J_vals.shape[0]:step_x, step_y:self.J_vals.shape[1]:step_y]
        self.x_vals = self.x_vals[step_x:self.x_vals.shape[0]:step_x]
        self.y_vals = self.y_vals[step_y:self.y_vals.shape[0]:step_y]

        self.J_max = np.max(self.J_vals)
        self.Jtot = np.sum(self.J_vals)

    def resample(self, n):
        result = deepcopy(self)
        result._resample(n)
        return result

    def plot(self):
        for i, x in enumerate(self.x_vals):
            for j, y in enumerate(self.y_vals):
                if self.J_vals[j, i] > 0.00001:
                    ms = int(self.J_vals[j, i] / self.J_max * 10)
                    c = cmap(self.J_vals[j, i] / self.J_max * 0.999)
                    gf.plot_point([x, y], ms=ms, color=c)

        # x0, y0 = self.center()

    def center(self):
        x0, y0 = 0.0, 0.0
        for i, x in enumerate(self.x_vals):
            for j, y in enumerate(self.y_vals):
                if self.J_vals[j, i] > 0.00001:
                    x0 += x * self.J_vals[j, i]
                    y0 += y * self.J_vals[j, i]
        x0 = x0 / self.Jtot
        y0 = y0 / self.Jtot
        return x0, y0


tok = TokameqFile(1.0e6, r'1MA_sn.txt')
print(tok.center())


# tok.plot()
# tok = tok.resample(3)
# tok.plot()

def load_fil_group(tok, wire_radius=0.05):
    filaments = FilamentGroup()

    for i, x in enumerate(tok.x_vals):
        for j, y in enumerate(tok.y_vals):
            if tok.J_vals[j, i] > 0.00001:
                ring = Ring(center=pt3D(0, y, 0), radius=x, normal=vec3D(0, -1, 0),
                            I=tok.J_vals[j, i] / tok.Jtot * tok.Jnominal,
                            wire_radius=wire_radius)  # 0.05
                filaments.add(ring)

    return filaments


filaments = load_fil_group(tok)
# single_filament = Ring(center=pt3D(0,  -0.1, 0), radius=1.62, normal=vec3D(0, 1, 0), I=tok.Jnominal, wire_radius=0.65)
single_filament = Ring(center=pt3D(0, -0.0987776, 0), radius=1.46535, normal=vec3D(0, 1, 0), I=tok.Jnominal,
                       wire_radius=0.6)

# filaments.plot()

probing_line = gf.line_array(pt3D(0.0, 0.0, 0.0), pt3D(5.0, 0.0, 0.0), 1000)


def plotB(wr):
    for r in filaments.elems:
        r.wire_radius = wr  # 0.01

    bb = np.array([filaments.calcB(r) for r in probing_line])
    plt.plot(probing_line[:, 0], bb[:, 1])


# plt.figure(24)
# plotB(0.05)

bb = np.array([single_filament.calcB(r) for r in probing_line])
# plt.plot( probing_line[:, 0], bb[:, 1] )

# plt.plot( [0.0, 5.0], [0.0, 0.0] )


plt.figure(17)
# plotB(0.60)
# plotB(0.30)
# plotB(0.15)
# plotB(0.05)
# plotB(0.01)


# geomT15.plot(plt.gca())

# %%

# sum([e.I for e in filaments.elems])

probing_line = gf.line_array(gf.pt3D(0.0, 0.050, 0.0), gf.pt3D(5.0, 0.050, 0.0), 1000)
plt.figure(243)
plotB(0.05)
