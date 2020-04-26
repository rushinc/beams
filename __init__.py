import numpy as np
from scipy import linalg as la

try:
    from mpi4py import MPI
    from mpi4py_fft import PFFT, DistArray, newDistArray
    comm = MPI.COMM_WORLD
    procs = comm.Get_size()
    rank = comm.Get_rank()
    if not rank:
        M, m = MPI.Get_version()
        print(f'Initialized MPI version {M}.{m}')
        print(f'{procs} processes')
    with_mpi = True
except ImportError as e:
    print('Failed to load MPI')
    with_mpi = False
    pass

def to_vec(elements, **kwargs):
    if elements is None:
        return None
    if isinstance(elements, Vector):
        return elements
    else:
        try:
            self.elements = Vector(*elements)
        except TypeError:
            self.elements = Vector(elements)

def to_vec2(elements, **kwargs):
    if elements is None:
        return None
    if isinstance(elements, Vector2d):
        return elements
    else:
        try:
            if len(elements) >= 2:
                return Vector2d(*elements[:2], **kwargs)
            else:
                return Vector2d(xy=elements[0], **kwargs)
        except TypeError:
            return Vector2d(xy=elements, **kwargs)

def to_vec3(elements, **kwargs):
    if elements is None:
        return None
    if isinstance(elements, Vector3d):
        return elements
    else:
        try:
            if len(elements) >= 3:
                return Vector3d(*elements[:3], **kwargs)
            else:
                return Vector3d(xyz=elements[0], **kwargs)
        except TypeError:
            return Vector3d(xyz=elements, **kwargs)

def star(S1, S2, dtype=complex):
    (M, N) = S1.shape
    if M != N:
        raise TypeError('Only square matrices are supported for the star product')
    if M % 2:
        raise TypeError('The size of matrices must be divisible by 2')
    if S2.shape != (M, N):
        raise TypeError('The shapes of the input matrices must be the same')

    n = M // 2
    I = np.eye(n, dtype=dtype)
    a11 = S1[:n, :n]
    a12 = S1[:n, n:]
    a21 = S1[n:, :n]
    a22 = S1[n:, n:]
    b11 = S2[:n, :n]
    b12 = S2[:n, n:]
    b21 = S2[n:, :n]
    b22 = S2[n:, n:]

    s11 = b11 @ la.solve(I - a12 @ b21, a11)
    s12 = b12 + (b11 @ a12 @ la.solve(I - b21 @ a12, b22))
    s21 = a21 + (b22 @ b21 @ la.solve(I - a12 @ b21, a11))
    s22 = a22 @ la.solve(I - b21 @ a12, b22)

    return np.block([[s11, s12], [s21, s22]])

from .materials import *
from .vectors import *
from .shapes import *
from .layer import *
from .solver import *
