# beams
The Berkeley ElectromAgnetic Modal Solver or beams solves Maxwell's equations using the Fourier Modal Method (FMM). Currently in the very early stages of development this Python module is riddled with bugs, lacks many basic features, and a comprehensive documentation.

## Installation
To install `beams` clone the repository and install using `python setup.py install`.

## Usage
The best way to familiarize with beams is to experiment with the notebooks in the `examples/` and `tests/` directories. These will be under regular update until a comprehensive documentation is made available. In the meantime, a brief sample is provided here.

Simulating the reflectance and transmission spectra of a square lattice photonic crystal with cylindrical air holes in silicon on a glass substrate with beams looks like this:
```
import beams as bm
import numpy as np

p = bm.Vector2d(1, 1)

air = bm.Material()
si = bm.Material(index=3.4)
sio2 = bm.Material(epsilon=1.5 ** 2)

hole = bm.Ellipse(r=.3, material=air)
phc = bm.Layer(h=.3, material=si,
        shapes=[hole], resolution=100)

inc = bm.Layer()
sub = bm.Layer(material=sio2)
cell = bm.Cell(period=p, N=13, layers=[inc, phc, sub])

freqs = np.linspace(0.6, 0.7, 100)
angles = bm.Vector3d(np.linspace(0, np.pi/3, 10))
(R, T) = cell.spectrum(freqs, angles)
```

## Background
Modal methods compute the functional form of the solution instead of the numeric values on a discrete grid. Since any function can be expressed as an infinite sum over a complete bases set, the coefficients for a truncated subset can be computationally solved for. The use of Fourier functions is a natural choice for the study of waves, especially in periodic environments.

This fundamental difference in approach from finite-difference or element methods provides its own set of advantages as well as limitations. For certain geometries and use cases the algorithm is inherently extremely fast and efficient. In fact, a three dimensional problem requires only a two dimensional numerical formulation. The solution along the remaining dimension is purely analytical and comprises only of boundary conditions at interfaces. However, a naive FMM can demonstrate poor convergence around sharp interfaces, especially those of metals. Similar problems exist across most approaches to solve Maxwell's equations and improving performance for complicated geometries often requires a rework of the mathematical formulation as well as the compulational implementation.

The primary purpose of beams is to provide a clean and intuitive scripting interface to enable quick analysis of optical elements like diffraction gratings, photonic crystals, and metallic nanoparticle arrays. Once a robust backbone is eshtablished devlopment will move towards extending the functionality and improving the accuracy of the solutions by integrating the latest advances in electromagnetic modal methods. However, a major focus will always be to ensure that the final code involved in desiging a structure remains easy to read, understand, and debug.

## To-do
- ~~Add some parallelization and measure the speed-up.~~
- Documentation (proposal, reports, readthedocs, etc.)
- Better build and packaging.
- Implement an inbuilt eigenmode solver.
- Allow PML boundaries for finite, non-periodic structures.
- Add support for magnetic and anisotropic materials.
- Smart slicing of 3D shapes.
- Include Legendre and Chebyshev polynomial bases.
- Integrate with an optimization toolbox for inverse-design.
- Add a non-linear solver.

## Disclaimer
*At present this is an exercise in parallelizing FMM for UC Berkeley's CS267. Nothing here will work as expected.*
