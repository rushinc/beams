## Imports
import numpy as np
import beams as bm
import matplotlib.pyplot as plt

## Inputs
y = 1.55
kx = 0
ky = 0
freq = 1 / y
k = bm.Vector2d(kx, ky)
N_modes = bm.Vector2d(9, 9)

## Materials
air = bm.Material()
SiO2 = bm.Material(epsilon=1.5 ** 2)
Si = bm.Material(epsilon=3.4 ** 2)

## Geometry
# Unit cell
px = .810                       # Period in 'x'
py = px                       # Period in 'y'
p = bm.Vector2d(px, py)
h_z = .320                       # Thickness of layers
L = .600 
w_1 = .230                       # Width of Si bar 1
w_2 = .190                       # Width of Si bar 2
gap = .125
C_1 = bm.Vector2d(y=-(w_1 + gap)/2)       # Center of Si bar 1
C_2 = bm.Vector2d(y=(w_2 + gap)/2)        # Center of Si bar 1
res = 1000

bar1 = bm.Rectangle(size=bm.Vector2d(L, w_1), center=C_1, material=Si)
bar2 = bm.Rectangle(size=bm.Vector2d(L, w_2), center=C_2, material=Si)
shapes = [bar1, bar2]
bars = bm.Layer(h_z, res, shapes, air)
inc = bm.Layer(0, res)
trn = bm.Layer(0, res, material=SiO2)
layers = [inc, bars, trn]

## Cell
cell = bm.Cell(p, N_modes, layers)
(R, T) = cell.R_T(freq, bm.Vector3d())
# (E, H) = cell.fields(freq, bm.Vector3d())

