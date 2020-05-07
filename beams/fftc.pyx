cimport cython
import numpy as np
cimport numpy as np
from scipy.linalg.lapack import cgesv
from libc.stdlib cimport malloc, free
from numpy import linalg as la
from numpy import fft
from cython.parallel import prange

DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t

cpdef void inv_toeplitz(DTYPE_t [:] fft_arr, DTYPE_t [:, :] toep) nogil:
    cdef:
        Py_ssize_t i, j
        Py_ssize_t N = toep.shape[0]

    for i in range(N):
        for j in range(N):
            toep[i, j] = fft_arr[i - j]

'''
cpdef void inv_toeplitz(DTYPE_t [:, :] raw_fft, DTYPE_t [:, :, :] toep_fft,
        int axis):
    cdef:
        Py_ssize_t G = raw_fft.shape[1 - axis]
        Py_ssize_t N = toep_fft.shape[axis]
        Py_ssize_t i, j, p
        DTYPE_t [:, :] toep = np.empty((N, N), dtype=DTYPE)
        DTYPE_t [:, :] I = np.eye(N, dtype=DTYPE)
        DTYPE_t [:] fft_n = np.empty(N - 1, dtype=DTYPE)
        DTYPE_t [:] fft_p = np.empty(N, dtype=DTYPE)

    for p in range(G):
        if axis==1:
            fft_p[:] = raw_fft[p, :N]
            fft_n[:] = raw_fft[p, 1 - N:]
        else:
            fft_p[:] = raw_fft[:N, p]
            fft_n[:] = raw_fft[1 - N:, p]
        for i in range(N):
            for j in range(i + 1):
                toep[i, j] = fft_p[i - j]
            for j in range(i + 1, N):
                toep[i, j] = fft_n[j - i - 1]
        if axis==0:
            toep_fft[:, p, :] = toep
        else:
            toep_fft[p, :, :] = toep
'''

cpdef fftc(grid, N, inv=None):
    (G_y, G_x) = grid.shape
    cdef Py_ssize_t G_y_c = G_y
    cdef Py_ssize_t G_x_c = G_x
    cdef Py_ssize_t Nx = N.x
    cdef Py_ssize_t Ny = N.y
    cdef Py_ssize_t pp, qq

    ieps_fft = np.empty(grid.shape, dtype=np.complex)
    if inv=='x':
        ieps_t = np.empty((N.x, N.x), dtype=np.complex)
        i_ieps = np.empty((N.x, G_y, N.x), dtype=np.complex)
    elif inv=='y':
        ieps_t = np.empty((N.y, N.y), dtype=np.complex)
        i_ieps = np.empty((G_x, N.y, N.y), dtype=np.complex)
    cdef np.complex128_t [:, :] ieps_fft_c = ieps_fft
    cdef np.complex128_t [:, :] ieps_t_c = ieps_t
    cdef np.complex128_t [:, :, :] i_ieps_c = i_ieps

    if inv=='x':
        ieps_fft = fft.fft(1 / grid, axis=0) / (G_x)

        for qq in prange(G_y_c, nogil=True):
            i_ieps_c[:, qq, :] = inv_toeplitz(ieps[:N.x, qq])

        epsxy_fft = fft.fftshift(fft.fft(i_ieps, axis=1),
                axes=1) / (G_y)
        epsxy_mnj = epsxy_fft[:, G_y // 2 + 1 - N.y:G_y // 2 + N.y, :]

        E4 = np.zeros((N.x, N.y, N.x, N.y), dtype=complex);
        for pp in prange(Nx, nogil=True):
            for qq in range(Nx):
                E4[pp, :, qq, :] = la.toeplitz(epsxy_mnj[pp,
                    N.y - 1:2 * N.y, qq], np.flip(epsxy_mnj[pp, :N.y, qq]))
        return np.reshape(E4, (N.t, N.t), order='F')

    if inv=='y':
        iepsy_fft = fft.fft(1 / grid, axis=1) / (G_y)

        for pp in prange(G_x_c, nogil=True):
            ieps_t_c = la.toeplitz(ieps[N.y - 1:2 * N.y],
                    np.flip(ieps[:N.y]))
            i_ieps_c[pp, :, :] = la.inv(ieps_t)

        epsyx_fft = fft.fftshift(fft.fft(i_ieps, axis=0),
                axes=0) / (G_x)
        epsyx_mnl = epsyx_fft[G_x // 2 - N.x + 1:G_x // 2 + N.x, :, :]

        E4 = np.zeros((N.x,N.y,N.x,N.y), dtype=complex)
        for rr in prange(Ny):
            for ss in range(Ny):
                E4[:, rr, :, ss] = la.toeplitz(epsyx_mnl[N.x - 1:2 * N.x - 1,
                    rr, ss], np.flip(epsyx_mnl[:N.x, rr, ss]))
        return np.reshape(E4, [N.t, N.t], order='F')

    EPS = np.zeros((N.t, N.t), dtype=complex)
    eps_fft = fft.fftshift(fft.fft2(grid)) / (G_x * G_y)
    eps_mn = eps_fft[G_x // 2 - N.x + 1:G_x // 2 + N.x,
            G_y // 2 - N.y + 1:G_y // 2 + N.y]

    for pp in prange(Nx):
        for qq in range(Ny):
            EPS[pp + N.x * qq, ::-1] = np.reshape(eps_mn[pp:pp + N.x,
                qq:qq + N.y], (1, -1), order='F')
    return EPS
