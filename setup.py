from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "fftc",
        ["beams/fftc.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='beams',
    version='0.0',
    packages=['beams',],
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}),
    zip_safe=False,
    license='GPL v3',
    long_description=open('README.md').read(),
)
