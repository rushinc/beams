import beams as bm
import numpy as np

p = bm.Vector2d(1, 1)

air = bm.Material()
si = bm.Material(index=3.4)
sio2 = bm.Material()
hole = bm.Ellipse(r=.3, material=air)

inc = bm.Layer()
sub = bm.Layer(material=sio2)
phc = bm.Layer(h=.3, material=si, shapes=[hole], resolution=500)

cell = bm.Cell(period=p, N=9, layers=[inc, phc, sub])

freqs = np.linspace(0.6, 0.9, 100)
angles = bm.Vector3d()

(R, T) = cell.spectrum(freqs, angles)

