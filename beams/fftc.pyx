cimport cython
import numpy as np
cimport numpy as np
from scipy.linalg.lapack import cgesv
from libc.stdlib cimport malloc, free
from numpy import linalg as la
from numpy import fft
from cython.parallel import prange
cimport scipy.linalg.cython_lapack as lapack_functions

DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t

cpdef void toeplitz_c(DTYPE_t [:] fft_arr, DTYPE_t [:, :] toep) nogil:
    cdef:
        Py_ssize_t i, j
        Py_ssize_t N = toep.shape[0]

    for i in range(N):
        for j in range(N):
            toep[i, j] = fft_arr[i - j]

cpdef void matx_inv_c(DTYPE_t [:, :] matrix) nogil:
    b = np.eye(matrix.shape[0], dtype=matrix.dtype)
    _, _, inverse, _ = lapack_functions.cgbsv(matrix, b)
    matrix = inverse

cpdef DTYPE_t [:, :] fftc(grid, N, inv=None):
    (G_x, G_y) = grid.shape
    cdef:
        Py_ssize_t G_x_c = G_x
        Py_ssize_t G_y_c = G_y
        Py_ssize_t Nx = N.x
        Py_ssize_t Ny = N.y
        Py_ssize_t pp, qq, rr, ss

    ieps_fft = np.empty(grid.shape, dtype=DTYPE)
    E4 = np.empty((N.x, N.y, N.x, N.y), dtype=DTYPE);
    if inv=='x':
        ieps_t = np.empty((N.x, N.x), dtype=DTYPE)
        i_ieps = np.empty((N.x, G_y, N.x), dtype=DTYPE)
        epsxy_fft = np.empty((N.x, G_y, N.x), dtype=DTYPE)
    elif inv=='y':
        ieps_t = np.empty((N.y, N.y), dtype=DTYPE)
        i_ieps = np.empty((G_x, N.y, N.y), dtype=DTYPE)
        epsxy_fft = np.empty((G_x, N.y, N.y), dtype=DTYPE)
    cdef:
        DTYPE_t [:, :] ieps_fft_c = ieps_fft
        DTYPE_t [:, :, :] i_ieps_c = i_ieps
        DTYPE_t [:, :, :] epsxy_fft_c = epsxy_fft
        DTYPE_t [:, :, :, :] E4_c = E4

    if inv=='x':
        ieps_fft = fft.fft(1 / grid, axis=0) / (G_x)

        for qq in prange(G_y_c, nogil=True):
            toeplitz_c(ieps_fft_c[:, qq], i_ieps_c[:, qq, :])
#           matx_inv_c(i_ieps_c[:, qq, :])

        epsxy_fft = fft.fft(i_ieps, axis=1) / (G_y)

        for pp in prange(Nx, nogil=True):
            for qq in range(Nx):
                toeplitz_c(i_ieps_c[pp, :, qq], E4_c[pp, :, qq, :])
        return np.reshape(E4, (N.t, N.t), order='F')

    if inv=='y':
        iepsy_fft = fft.fft(1 / grid, axis=1) / (G_y)

        for pp in prange(G_x_c, nogil=True):
            toeplitz_c(ieps_fft_c[pp, :], i_ieps_c[pp, :, :])
#           matx_inv_c(i_ieps_c[pp, :, :])

        epsyx_fft = fft.fft(i_ieps, axis=0) / (G_x)

        for rr in prange(Ny, nogil=True):
            for ss in range(Ny):
                toeplitz_c(i_ieps_c[:, rr, ss], E4_c[:, rr, :, ss])
        return np.reshape(E4, [N.t, N.t], order='F')

    EPS = np.zeros((N.t, N.t), dtype=DTYPE)
    eps_fft = fft.fftshift(fft.fft2(grid)) / (G_x * G_y)
    eps_mn = eps_fft[G_x // 2 - N.x + 1:G_x // 2 + N.x,
            G_y // 2 - N.y + 1:G_y // 2 + N.y]

    for pp in range(Nx):
        for qq in range(Ny):
            EPS[pp + N.x * qq, ::-1] = np.reshape(eps_mn[pp:pp + N.x,
                qq:qq + N.y], (1, -1), order='F')
    return EPS

