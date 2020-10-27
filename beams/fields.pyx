cimport cython
cimport openmp
cimport numpy as np
import cython
import numpy as np
import beams as bm
from cython.parallel import prange
from beams import *

openmp.omp_set_num_threads(NUM_THREADS)

def fields_in_plane(l, pts, amplitudes, components=''):
    N_t = l.N.x * l.N.y
    k0 = 2 * np.pi * l.freq
    U_xy = l.U
    V_xy = l.V
    U_z = la.solve(l._fft_eps,
            l.K.rotate(np.pi / 2).hstack()) @ l.V
    V_z = l.K.rotate(np.pi / 2).hstack() @ l.U

    if amplitudes.x.shape[1] == 2:
        U_xy = np.hstack((U_xy, U_xy))
        V_xy = np.hstack((V_xy, -V_xy))
        U_z = np.hstack((U_z, -U_z))
        V_z = np.hstack((V_z, V_z))

    u_xy = U_xy @ amplitudes.vstack().reshape((amplitudes.x.shape[1]\
            * 2 * N_t, 1), order='F')
    v_xy = V_xy @ amplitudes.vstack().reshape((amplitudes.x.shape[1]\
            * 2 * N_t, 1), order='F')
    u_z = U_z @ amplitudes.vstack().reshape((amplitudes.x.shape[1]\
            * 2 * N_t, 1), order='F')
    v_z = V_z @ amplitudes.vstack().reshape((amplitudes.x.shape[1]\
            * 2 * N_t, 1), order='F')

    e_x = np.zeros((pts.x.size, pts.y.size), dtype=complex)
    e_y = np.zeros((pts.x.size, pts.y.size), dtype=complex)
    e_z = np.zeros((pts.x.size, pts.y.size), dtype=complex)
    h_x = np.zeros((pts.x.size, pts.y.size), dtype=complex)
    h_y = np.zeros((pts.x.size, pts.y.size), dtype=complex)
    h_z = np.zeros((pts.x.size, pts.y.size), dtype=complex)
    E = Vector3d(e_x, e_y, e_z)
    H = Vector3d(h_x, h_y, h_z)

    for (j, yy) in enumerate(pts.y.flatten()):
        r = Vector2d(pts.x, yy)
        k_phase = np.exp(-k0 * l.K.diag().dot(r))
        E.x[:, [j]] = k_phase @ u_xy[:N_t]
        E.y[:, [j]] = k_phase @ u_xy[N_t:]
        H.x[:, [j]] = k_phase @ v_xy[:N_t]
        H.y[:, [j]] = k_phase @ v_xy[N_t:]
        E.z[:, [j]] = k_phase @ u_z
        H.z[:, [j]] = k_phase @ v_z

    return (E, H)

