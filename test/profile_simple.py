import sys
sys.path.append('/home/rushin/Documents/Python/')

import beams as bm
import numpy as np
import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()

p = bm.Vector2d(1, 1)

air = bm.Material()
si = bm.Material(index=3.4)
sio2 = bm.Material(epsilon=1.5 ** 2)
hole = bm.Ellipse(r=.3, material=air)

inc = bm.Layer()
sub = bm.Layer(material=sio2)

freqs = np.linspace(0.6, 0.75, 1)
angles = bm.Vector3d()

#NN = np.arange(1, 21, 2)
#times = np.empty((3, len(NN)))
ress = np.arange(100, 5100, 700)
times = np.empty((3, len(ress)))

#for (nn, N) in enumerate(NN):
#    print('N = ' + str(N))
for (nn, res) in enumerate(ress):
    print('res = ' + str(res))
    phc = bm.Layer(h=.3, material=si, shapes=[hole], resolution=res)
    cell = bm.Cell(period=p, N=11, layers=[inc, phc, sub])

    pr.enable()
    (R, T) = cell.spectrum(freqs, angles)
    pr.disable()

    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats()
    
    for l in s.getvalue().split('\n'):
        if 'ncalls' in l or '__eigs' in l or '__fft' in l or 'linsolve' in l:
            print(l)
            if 'linsolve' in l:
                times[0, nn] = float(l.split()[3])
            if '__eigs' in l:                     
                times[1, nn] = float(l.split()[3])
            if '__fft' in l:                      
                times[2, nn] = float(l.split()[3])

