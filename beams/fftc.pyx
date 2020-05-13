cimport cython
cimport openmp
cimport numpy as np
from libc.stdlib cimport malloc, free
from scipy.linalg.cython_lapack cimport zgetrf, zgetri
import cython
import pyfftw
import numpy as np
from cython.parallel import prange
from beams import NUM_THREADS

pyfftw.config.NUM_THREADS = NUM_THREADS
openmp.omp_set_num_threads(NUM_THREADS)
DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t

@cython.boundscheck(False)
cpdef void toeplitz_c(DTYPE_t [:] fft_arr, DTYPE_t [:, :] toep, int no_neg=0) nogil:
    cdef:
        Py_ssize_t i, j
        Py_ssize_t N = toep.shape[0]

    for i in range(N):
        for j in range(N):
            if no_neg and i - j < 0:
                toep[i, j] = fft_arr[j - i].conjugate()
            else:
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

    gi = pyfftw.empty_aligned(grid.shape, dtype=grid.dtype)
    if 'complex' not in str(gi.dtype):
        f_shape = (grid.shape[0] // 2 + 1, grid.shape[1])
    else:
        f_shape = grid.shape
    gf = pyfftw.empty_aligned(f_shape, dtype=DTYPE)
    fft_1 = pyfftw.FFTW(gi, gf, axes=(0, ), threads=NUM_THREADS)
    gi[:] = 1 / grid
    ieps_fft = fft_1()
    ieps_fft /= G[0]

#   ieps_fft = fft.fft(1 / grid, axis=0) / G[0]
    i_ieps = pyfftw.empty_aligned((G[1], N.x, N.x), dtype=DTYPE)
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
        toeplitz_c(ieps_fft_c[:, qq], i_ieps_c[qq, :, :], 1)
        matx_inv_c(i_ieps_c[qq, :, :])

    gi_2 = pyfftw.empty_aligned(i_ieps.shape, dtype=DTYPE)
    gf_2 = pyfftw.empty_aligned(i_ieps.shape, dtype=DTYPE)
    fft_2 = pyfftw.FFTW(gi_2, gf_2, axes=(0, ), threads=NUM_THREADS)
    gi_2[:] = i_ieps
    epsxy_fft = fft_2()
    epsxy_fft /= G[1]

#   epsxy_fft = fft.fft(i_ieps, axis=0) / G[1]
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

    gi = pyfftw.empty_aligned(grid.shape, dtype=grid.dtype)
    if 'complex' not in str(gi.dtype):
        f_shape = (grid.shape[0], grid.shape[1] // 2 + 1)
    else:
        f_shape = grid.shape
    gf = pyfftw.empty_aligned(f_shape, dtype=DTYPE)
    fft_1 = pyfftw.FFTW(gi, gf, axes=(1, ), threads=NUM_THREADS)
    gi[:] = 1 / grid
    ieps_fft = fft_1()
    ieps_fft /= G[1]

#   ieps_fft = fft.fft(1 / grid, axis=1) / G[1]
    i_ieps = pyfftw.empty_aligned((G[0], N.y, N.y), dtype=DTYPE)
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
        toeplitz_c(ieps_fft_c[pp, :], i_ieps_c[pp, :, :], 1)
        matx_inv_c(i_ieps_c[pp, :, :])

    gi_2 = pyfftw.empty_aligned(i_ieps.shape, dtype=DTYPE)
    gf_2 = pyfftw.empty_aligned(i_ieps.shape, dtype=DTYPE)
    fft_2 = pyfftw.FFTW(i_ieps, gf_2, axes=(0, ), threads=NUM_THREADS)
    gi_2[:] = i_ieps
    epsxy_fft = fft_2()
    epsxy_fft /= G[0]

#   epsxy_fft = fft.fft(i_ieps, axis=0) / G[0]
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

    gi = pyfftw.empty_aligned(grid.shape, dtype=grid.dtype)
    if 'complex' not in str(gi.dtype):
        f_shape = (grid.shape[0] // 2 + 1, grid.shape[1])
    else:
        f_shape = grid.shape
    gf = pyfftw.empty_aligned(f_shape, dtype=DTYPE)
    fft_1 = pyfftw.FFTW(gi, gf, axes=(0, ), threads=NUM_THREADS)
    gi[:] = grid
    eps_fft = fft_1()
    eps_fft /= G[0]

#   eps_fft = fft.fft(grid, axis=0) / G[0]
    eps_mn = pyfftw.empty_aligned((G[1], N.x, N.x), dtype=DTYPE)
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
        toeplitz_c(eps_fft_c[:, qq], eps_mn_c[qq, :, :], 1)

    gi_2 = pyfftw.empty_aligned(eps_mn.shape, dtype=DTYPE)
    gf_2 = pyfftw.empty_aligned(eps_mn.shape, dtype=DTYPE)
    fft_2 = pyfftw.FFTW(gi_2, gf_2, axes=(0, ), threads=NUM_THREADS)
    gi_2[:] = eps_mn
    epsxy_fft = fft_2()
    epsxy_fft /= G[1]

#   epsxy_fft = fft.fft(eps_mn, axis=0) / G[1]
    cdef DTYPE_t [:, :, :] epsxy_fft_c = epsxy_fft

    for pp in prange(Nx, nogil=True, schedule='guided'):
        for qq in range(Nx):
            toeplitz_c(epsxy_fft_c[:, pp, qq], E4_c[pp, :, qq, :])

    return np.reshape(E4, (N_t, N_t), order='F')
