cimport cython
cimport openmp
cimport numpy as np
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from scipy.linalg.cython_lapack cimport zgetrf, zgetri
import cython
import numpy as np
from cython.parallel import prange
from numpy import fft

DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t
openmp.omp_set_num_threads(1)

@cython.boundscheck(False)
cpdef void toeplitz_c(DTYPE_t [:] fft_arr, DTYPE_t [:, :] toep) nogil:
    cdef:
        Py_ssize_t i, j
        Py_ssize_t N = toep.shape[0]

    for i in range(N):
        for j in range(N):
            toep[i, j] = fft_arr[i - j]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void matx_inv_c(double complex [:, :] matrix) nogil:
    cdef:
        int N = matrix.shape[0]
        int info, lwork = N
        int *ipiv = <int *> malloc(N * sizeof(int))
        double complex *work = <double complex *> malloc(N * N *\
                sizeof(double complex))

    zgetrf(&N, &N, &matrix[0, 0], &N, ipiv, &info)
    zgetri(&N, &matrix[0,0], &N, ipiv, work, &lwork, &info)

    free(ipiv)
    free(work)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] fftc_ix(grid, N):
    G = grid.shape
    N_t = N.x * N.y

    ieps_fft = fft.fft(1 / grid, axis=0) / G[0]
    i_ieps = np.empty((G[1], N.x, N.x), dtype=DTYPE)
    E4 = np.empty((N.x, N.y, N.x, N.y), dtype=DTYPE)

    cdef:
        (int, int) G_c = G
        Py_ssize_t Nx = N.x
        Py_ssize_t Ny = N.y
        Py_ssize_t pp, qq, rr, ss
        DTYPE_t [:, :] ieps_fft_c = ieps_fft
        DTYPE_t [:, :, :] i_ieps_c = i_ieps
        DTYPE_t [:, :, :, :] E4_c = E4

    for qq in prange(G_c[1], nogil=True, schedule='guided'):
        toeplitz_c(ieps_fft_c[:, qq], i_ieps_c[qq, :, :])
        matx_inv_c(i_ieps_c[qq, :, :])

    epsxy_fft = fft.fft(i_ieps, axis=0) / G[1]
    cdef DTYPE_t [:, :, :] epsxy_fft_c = epsxy_fft

    for pp in prange(Nx, nogil=True, schedule='guided'):
        for qq in range(Nx):
            toeplitz_c(epsxy_fft_c[:, pp, qq], E4_c[pp, :, qq, :])

    return np.reshape(E4, (N_t, N_t), order='F')

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] fftc_iy(grid, N):
    G = grid.shape
    N_t = N.x * N.y

    ieps_fft = fft.fft(1 / grid, axis=1) / G[1]
    i_ieps = np.empty((G[0], N.y, N.y), dtype=DTYPE)
    E4 = np.empty((N.x, N.y, N.x, N.y), dtype=DTYPE)

    cdef:
        (int, int) G_c = G
        Py_ssize_t Nx = N.x
        Py_ssize_t Ny = N.y
        Py_ssize_t pp, qq, rr, ss
        DTYPE_t [:, :] ieps_fft_c = ieps_fft
        DTYPE_t [:, :, :] i_ieps_c = i_ieps
        DTYPE_t [:, :, :, :] E4_c = E4

    for pp in prange(G_c[0], nogil=True, schedule='guided'):
        toeplitz_c(ieps_fft_c[pp, :], i_ieps_c[pp, :, :])
        matx_inv_c(i_ieps_c[pp, :, :])

    epsxy_fft = fft.fft(i_ieps, axis=0) / G[0]
    cdef DTYPE_t [:, :, :] epsxy_fft_c = epsxy_fft

    for rr in prange(Ny, nogil=True, schedule='guided'):
        for ss in range(Ny):
            toeplitz_c(epsxy_fft_c[:, rr, ss], E4_c[:, rr, :, ss])

    return np.reshape(E4, [N_t, N_t], order='F')

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] fftc_in(grid, N):
    G = grid.shape
    N_t = N.x * N.y

    eps_fft = fft.fft(grid, axis=0) / G[0]
    eps_mn = np.empty((G[1], N.x, N.x), dtype=DTYPE)
    E4 = np.empty((N.x, N.y, N.x, N.y), dtype=DTYPE)

    cdef:
        (int, int) G_c = G
        Py_ssize_t Nx = N.x
        Py_ssize_t Ny = N.y
        Py_ssize_t pp, qq, rr, ss
        DTYPE_t [:, :] eps_fft_c = eps_fft
        DTYPE_t [:, :, :] eps_mn_c = eps_mn
        DTYPE_t [:, :, :, :] E4_c = E4

    for qq in prange(G_c[1], nogil=True, schedule='guided'):
        toeplitz_c(eps_fft_c[:, qq], eps_mn_c[qq, :, :])

    epsxy_fft = fft.fft(eps_mn, axis=0) / G[1]
    cdef DTYPE_t [:, :, :] epsxy_fft_c = epsxy_fft

    for pp in prange(Nx, nogil=True, schedule='guided'):
        for qq in range(Nx):
            toeplitz_c(epsxy_fft_c[:, pp, qq], E4_c[pp, :, qq, :])

    return np.reshape(E4, (N_t, N_t), order='F')
