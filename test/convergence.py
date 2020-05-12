import sys
sys.path.append('/home/rushin/Documents/Python/')

import beams as bm
import numpy as np

p = bm.Vector2d(1, 1)

air = bm.Material()
si = bm.Material(index=3.4)
sio2 = bm.Material(epsilon=1.5 ** 2)
hole = bm.Ellipse(r=.3, material=air)

inc = bm.Layer()
sub = bm.Layer(material=sio2)
phc = bm.Layer(h=.3, material=si, shapes=[hole], resolution=500)

cell = bm.Cell(period=p, N=29, layers=[inc, phc, sub])

f = 0.75
a = bm.Vector3d(np.pi / 3, np.pi / 6, np.pi / 2)

(FD, ft) = phc.fft_convergence(1e4, 25, bm.Vector2d(xy=9), p)
# (RT, tt) = cell.convergence(29, freq=f, angles=a)

