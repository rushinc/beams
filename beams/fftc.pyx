import numpy as np
cimport numpy as np
from numpy import linalg as la
from numpy import fft
from cython.parallel import prange

cpdef fftc(grid, N, inv=None):
    (G_y, G_x) = grid.shape
    cdef int G_y_c = G_y

    if inv=='x':
        cdef np.ndarray[np.complex128, ndim=3] i_iepsx_mj = np.zeros((N.x, G_y, N.x), dtype=complex)
        iepsx_fft = fft.fftshift(fft.fft(1 / grid, axis=0),
                axes=0) / (G_x)

        for qq in prange(G_y_c, nogil=True):
            iepsx_m = iepsx_fft[G_x // 2 - N.x + 1:G_x // 2 + N.x, qq]
            iepsx_mj = la.toeplitz(iepsx_m[N.x - 1:2 * N.x],
                    np.flip(iepsx_m[:N.x]))
            i_iepsx_mj[:, qq, :] = la.inv(iepsx_mj)

        epsxy_fft = fft.fftshift(fft.fft(i_iepsx_mj, axis=1),
                axes=1) / (G_y)
        epsxy_mnj = epsxy_fft[:, G_y // 2 + 1 - N.y:G_y // 2 + N.y, :]

        E4 = np.zeros((N.x, N.y, N.x, N.y), dtype=complex);
        for pp in prange(N.x, nogil=True):
            for qq in range(N.x):
                E4[pp, :, qq, :] = la.toeplitz(epsxy_mnj[pp,
                    N.y - 1:2 * N.y, qq], np.flip(epsxy_mnj[pp, :N.y, qq]))
        return np.reshape(E4, (N.t, N.t), order='F')

    if inv=='y':
        i_iepsy_nl = np.zeros((G_x, N.y, N.y), dtype=complex)
        iepsy_fft = fft.fftshift(fft.fft(1 / grid, axis=1),
                axes=1) / (G_y)

        for pp in range(G_x):
            iepsy_n = iepsy_fft[pp, G_y // 2 - N.y + 1:G_y // 2 + N.y]
            iepsy_nl = la.toeplitz(iepsy_n[N.y - 1:2 * N.y],
                    np.flip(iepsy_n[:N.y]))
            i_iepsy_nl[pp, :, :] = la.inv(iepsy_nl)

        epsyx_fft = fft.fftshift(fft.fft(i_iepsy_nl, axis=0),
                axes=0) / (G_x)
        epsyx_mnl = epsyx_fft[G_x // 2 - N.x + 1:G_x // 2 + N.x, :, :]

        E4 = np.zeros((N.x,N.y,N.x,N.y), dtype=complex)
        for rr in range(N.y):
            for ss in range(N.y):
                E4[:, rr, :, ss] = la.toeplitz(epsyx_mnl[N.x - 1:2 * N.x - 1,
                    rr, ss], np.flip(epsyx_mnl[:N.x, rr, ss]))
        return np.reshape(E4, [N.t, N.t], order='F')

    EPS = np.zeros((N.t, N.t), dtype=complex)
    eps_fft = fft.fftshift(fft.fft2(grid)) / (G_x * G_y)
    eps_mn = eps_fft[G_x // 2 - N.x + 1:G_x // 2 + N.x,
            G_y // 2 - N.y + 1:G_y // 2 + N.y]

    for pp in range(N.x):
        for qq in range(N.y):
            EPS[pp + N.x * qq, ::-1] = np.reshape(eps_mn[pp:pp + N.x,
                qq:qq + N.y], (1, -1), order='F')
    return EPS


