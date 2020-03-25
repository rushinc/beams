# beams
The Berkeley ElectromAgnetic Modal Solver or beams solves the Maxwell's equations using the Fourier Modal Method (FMM). Currently in the very early stages of development this Python module lacks many basic features, a documentation, and an easy to use interface. However there are plans on implementing the latest advances in electromagnetic modal methods as the software matures.

## Installation
To install `beams` clone the repository to a location in your `PYTHONPATH` or add the parent directory to `sys.path`. A mature build process is yet to be implemented.
1. Clone the repository: 
```
git clone https://github.com/rushinc/beams
```
2. In your python script or consol add: 
```
import sys
sys.path.append('path/to/beams')
```
3. Import as any other module
```
import beams as bm
```

## To-do:
- Make the serial implementation work correctly for the simple cases.
- Documentation (proposal, reports, readthedocs, etc.)
- Add some parallelization and measure the speed-up.
- Better build and packaging.
- Add support for dispersive materials.
- Implement an inbuilt eigenmode solver.
- Add support for magnetic materials.
- Add support for anisotropic materials.
- Include Legendre and Chebyshev polynomial bases.
- Integrate with an optimization toolbox for inverse-design.
- Add a non-linear solver for.

## Disclaimer:
*At present this is a playground for parallelizing FMM for UC Berkeley's CS267. Nothing here will work as expected.*
