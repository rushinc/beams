import numpy as np
import scipy as sp
from scipy import linalg as la
from bisect import bisect
import beams as bm
from beams import *

class Cell:
    def __init__(self, period, N_modes, layers):
        self.p = period
        self.N = N_modes
        self.layers = list(layers)
        self.freq = None
        self.k = None
        self.u = None

    def build(self, freq, k):
        L = len(self.layers)
        N_t = self.N.x * self.N.y
        B = np.zeros((4 * N_t * (L + 1), 4 * N_t * (L + 1)), dtype=complex)
        Z = np.zeros((2 * N_t, 2 * N_t), dtype=complex)
        for i, l in enumerate(self.layers):
            l.compute_eigs(freq, k, self.p, self.N)
            b_l = np.block([[l.U, l.U @ l.X], [l.V, -l.V @ l.X],
                [-l.U @ l.X, -l.U], [-l.V @ l.X, l.V]])
            B[4 * i * N_t:4 * (i + 2) * N_t,
                    4 * i * N_t:4 * (i + 1) * N_t] = b_l
            if i == 0:
                B[:4 * N_t, 4 * N_t * L:] = np.block([[-l.U, Z], [l.V, Z]])
            if i == L - 1:
                B[-4 * N_t:, 4 * N_t * L:] = np.block([[Z, l.U], [Z, l.V]])
        self.B = B
        self.freq = freq
        self.k = k

    def linsolve(self, freq, k, b):
        L = len(self.layers)
        N_t = self.N.x * self.N.y

        if self.freq != freq or self.k != k:
            self.build(freq, k)

        C = la.solve(self.B, b)
        self.C = np.reshape(C, (4 * N_t, L + 1), order='F')
        self.C[2 * N_t:4 * N_t, L - 1] = np.zeros(2 * N_t, dtype=complex)

    def angles_to_k(self, freq, angles):
        theta = angles.x
        phi = angles.y
        psi = angles.z
        kx = self.layers[0].material.get('n', freq) * np.outer(np.sin(theta),
                np.cos(phi))
        ky = self.layers[0].material.get('n', freq) * np.outer(np.sin(theta),
                np.sin(phi))
        ks = Vector2d(kx, ky).flatten()
        return ks

    def angles_to_u(self, angles):
        theta = angles.x
        phi = angles.y
        psi = angles.z
        ux = np.zeros((theta.size * phi.size, psi.size))
        uy = np.zeros((theta.size * phi.size, psi.size))
        for (i, p) in enumerate(psi.flat):
            px = np.cos(p) * np.outer(np.cos(theta),
                    np.cos(phi)) - np.sin(p) * np.sin(phi)
            py = np.cos(p) * np.outer(np.cos(theta),
                    np.sin(phi)) - np.sin(p) * np.cos(phi)
            ux[:, i] = px.flatten()
            uy[:, i] = py.flatten()
        return Vector2d(ux, uy)

    def __excitation(self, u, layer=0):
        L = len(self.layers)
        N_t = self.N.x * self.N.y
        l = layer
        exc = np.zeros((2 * N_t, 1))
        exc[N_t // 2] = u.x
        exc[N_t + N_t // 2] = u.y
        ui = np.vstack((exc, self.layers[l].V @ exc))
        self.u = u
        return np.vstack((np.zeros((4 * N_t * l, 1)), ui,
            np.zeros((4 * N_t * (L - l), 1))))

    def diffraction_orders(self, freq=None, angles=None):
        L = len(self.layers)
        N_t = self.N.x * self.N.y
        if angles is not None:
            u = self.angles_to_u(angles)
            k = self.angles_to_k(freq, angles)
        else:
            if self.u is None:
                raise AttributeError('Cell does not contain any solutions. ' +\
                        'R() missing 2 required positional arguments: ' +\
                        '\'freq\', and \'angles\'')
                return
            u = self.u
            k = self.k

        if (freq is None or k is None) and self.C is None:
            raise AttributeError('Cell does not contain any solutions. ' +\
                    'R() missing 2 required positional arguments: ' +\
                    '\'freq\', and \'angles\'')
            return
        elif (freq is not None and freq != self.freq) or\
                (k is not None and k != self.k):
            self.build(freq, k)
            b = self.__excitation(u)
            self.linsolve(freq, k, b)

        g_r = self.layers[0].gamma[:N_t]
        g_t = self.layers[-1].gamma[:N_t]
        rx = self.C[:N_t, L]
        ry = self.C[N_t:2 * N_t, L]
        rz = self.layers[0].K.diag().dot(Vector2d(rx,
            ry)) / g_r
        tx = self.C[2 * N_t:3 * N_t, L]
        ty = self.C[3 * N_t:, L]
        tz = -self.layers[-1].K.diag().dot(Vector2d(tx,
            ty)) / g_t
        return (Vector3d(rx, ry, rz), Vector3d(tx, ty, tz))

    def R_T(self, freq=None, angles=None):
        N_t = self.N.x * self.N.y
        (r, t) = self.diffraction_orders(freq, angles)
        g_r = self.layers[0].gamma[:N_t]
        g_t = self.layers[-1].gamma[:N_t]
        ri = np.real(g_r / g_r[N_t // 2]) @ r.intensity()
        ti = np.real(g_t / g_r[N_t // 2]) @ t.intensity()
        return (ri, ti)

    def fields(self, freq, angles, pts):
        L = len(self.layers)
        N_t = self.N.x * self.N.y
        theta = angles.x
        phi = angles.y
        psi = angles.z

        u = self.angles_to_u(angles)
        k = self.angles_to_k(freq, angles)

        self.build(freq, k)
        b = self.__excitation(u)
        self.linsolve(freq, k, b)

        e_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        e_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        e_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        h_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        h_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        h_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        E = Vector3d(e_x, e_y, e_z)
        H = Vector3d(h_x, h_y, h_z)

        zi = np.cumsum([l.h for l in self.layers])
        for (k, z) in enumerate(pts.z.flatten()):
            li = bisect(zi[:-1], z)
            l = self.layers[li]
            if li > 0: zz = z - zi[li - 1]
            else: zz = z
            c = self.C[:, [li]]
            amp = Vector2d(np.hstack((c[:N_t], c[2 * N_t:3 * N_t])),
                    np.hstack((c[N_t:2 * N_t], c[3 * N_t:])))
            pts_n = Vector3d(pts.x, pts.y, zz)
            (E[:, :, [k]], H[:, :, [k]]) = l.get_fields(pts_n, amp)

        return (E, H)

    def spectrum(self, freqs, angles):
        L = len(self.layers)
        N_t = self.N.x * self.N.y
        theta = angles.x
        phi = angles.y
        psi = angles.z

        R = np.empty((len(freqs), theta.size, phi.size, psi.size))
        T = np.empty((len(freqs), theta.size, phi.size, psi.size))

        u = self.angles_to_u(angles)

        for (i, f) in enumerate(freqs):
            ks = self.angles_to_k(f, angles)
            R_k = np.zeros((theta.size * phi.size, psi.size))
            T_k = np.zeros((theta.size * phi.size, psi.size))
            for (j, k) in enumerate(ks):
                self.build(f, k)
                for l in range(psi.size):
                    b = self.__excitation(u[j, l])

                    self.linsolve(f, k, b)
                    (r, t) = self.diffraction_orders()
                    (ri, ti) = self.R_T()
                    prog = 100 * (l + 1 + j * psi.size +\
                            i * theta.size * phi.size * psi.size) / R.size
                    print('Progress: ' + str(round(prog, 2)) + '%: ' +\
                            'R = ' + str(ri) + ', T = ' + str(ti), end='\r')

                    R_k[j, l] = ri
                    T_k[j, l] = ti
            R[i, :, :, :] = np.reshape(R_k, (theta.size, phi.size, psi.size))
            T[i, :, :, :] = np.reshape(T_k, (theta.size, phi.size, psi.size))

        return (np.squeeze(R), np.squeeze(T))
