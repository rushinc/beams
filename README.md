# beams
The Berkeley ElectromAgnetic Modal Solver or beams solves the Maxwell's equations using the Fourier Modal Method (FMM). Currently in the very early stages of development this Python module lacks many basic features, a documentation, and an easy to use interface. However there are plans on implementing the latest advances in electromagnetic modal methods as the software matures.

## Installation
To install `beams` clone the repository to a location in your `PYTHONPATH` or add the parent directory to `sys.path`. An automated build process is yet to be implemented.
1. Clone the repository: 
```
git clone https://github.com/rushinc/beams
```
2. In your python script or console add: 
```
import sys
sys.path.append('path/to/beams')
```
3. Import like you would any other package
```
import beams as bm
```
## Development
The `test/` directory contains the Matlab version of the code `si_sensor.m`. This is just a long unorganized script which is to be converted in a better organized Python package. The structure this script will run is a periodic arrangment of two assymmetric Silicon bars on a glass substrate.

The `test/si_sensor.py` should in the end produce results similar to the Matlab version.

### To-do:
- Make the serial implementation work correctly for Si sensor.
- Documentation (proposal, reports, readthedocs, etc.)
- Add some parallelization and measure the speed-up.
- Better build and packaging.
- More shapes.
- Add support for dispersive materials.
- Implement an inbuilt eigenmode solver.
- Allow PML boundaries for finite, non-periodic structures.
- Add support for magnetic materials.
- Add support for anisotropic materials.
- Include Legendre and Chebyshev polynomial bases.
- Integrate with an optimization toolbox for inverse-design.
- Add a non-linear solver.

## Disclaimer:
*At present this is an exercise in parallelizing FMM for UC Berkeley's CS267. Nothing here will work as expected.*
