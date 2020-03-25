## Imports
import numpy as np
import beams as bm
import matplotlib.pyplot as plt

## Inputs
y = np.arange(1.500, 1.700, .001)    # Range of wavelength in nm
kx = 0
ky = 0
freqs = 1 / y
k = bm.vec2(kx, ky)
N_modes = bm.vec2(9, 9)

## Materials
air = bm.Material()
SiO2 = bm.Material(index=1.5)
Si = bm.Material(index=3.4)

## Geometry
# Unit cell
px = .810                       # Period in 'x'
py = px                       # Period in 'y'
h_z = .320                       # Thickness of layers
L = .600 
w_1 = .230                       # Width of Si bar 1
w_2 = .190                       # Width of Si bar 2
gap = .125
C_1 = bm.vec2(y=-(w_1 + gap)/2)       # Center of Si bar 1
C_2 = bm.vec2(y=(w_2 + gap)/2)        # Center of Si bar 1
res = 4096

bar1 = bm.Rectangle(size=bm.vec2(L, w_1), center=C_1, material=Si)
bar2 = bm.Rectangle(size=bm.vec2(L, w_2), center=C_2, material=Si)
shapes = [bar1, bar2]
bars = bm.Layer(0, res, shapes, air)
layers = [bars]

## Solver
R = []; T = []
for freq in freqs:
    sol = bm.Solver(freq, k, N_modes, layers)
    sol.compute_all()
    R.append(sol.reflectance())
    T.append(sol.transmittance())

## Plot
plt.plot(freqs, R)
