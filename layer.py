import numpy as np
import scipy as sp
import beams as bm
from numpy import fft
from scipy import linalg
from beams import *

class Layer:
    def __init__(self, h, resolution, shapes=None, material=Material()):
        self.h = float(h)
        self.resolution = resolution
        if not shapes:
            self.shapes = []
        else:
            self.shapes = shapes
        self.material = material

    def grid(self, period, freq=None, value='id'):
        px = period.x
        py = period.y
        res = self.resolution
        grid = np.mgrid[-px / 2 : px / 2 : 1 / res,
                -py / 2 : py / 2 : 1 / res]
        vgrid = bm.Vector2d(grid[0], grid[1])
        layout = np.zeros(grid[0].shape)
        for id, shape in list(enumerate(self.shapes))[::-1]:
            interior_indices = shape.interior(vgrid)
            layout[np.logical_and(layout == 0, interior_indices)]\
                    = id + 1
        if 'eps' in value: 
            eps_grid = np.full(layout.shape, self.material.get_eps(freq))
            for id, shape in enumerate(self.shapes):
                eps_grid[layout == id + 1] = shape.material.get_eps(freq)
            return eps_grid
        return layout

    def ffts(self, grid, N_modes):
        N_x = int(N_modes.x)
        N_y = int(N_modes.y)
        N_t = N_x * N_y

        (G_x, G_y) = grid.shape

        EPS = np.zeros((N_t, N_t), dtype=complex)
        eps_fft = fft.fftshift(fft.fft2(grid)) / (G_x * G_y)
        eps_mn = eps_fft[G_x // 2 - N_x + 1:G_x // 2 + N_x,
                G_y // 2 - N_y + 1:G_y // 2 + N_y]
    
        for pp in range(N_x):
            for qq in range(N_y):
                EPS[pp + N_x * qq, ::-1] = np.reshape(eps_mn[pp:pp + N_x, qq:qq + N_y], (1, -1), order='F')
        self.fft_eps = EPS

        i_iepsx_mj = np.zeros((N_x, G_y, N_x), dtype=complex)
        iepsx_fft = fft.fftshift(fft.fft(1 / grid, axis=0),
                axes=0) / (G_x)

        for qq in range(G_y):
            iepsx_m = iepsx_fft[G_x // 2 - N_x + 1:G_x // 2 + N_x, qq]
            iepsx_mj = linalg.toeplitz(iepsx_m[N_x - 1:2 * N_x],
                    np.flip(iepsx_m[:N_x]))
            i_iepsx_mj[:, qq, :] = linalg.inv(iepsx_mj)

        epsxy_fft = fft.fftshift(fft.fft(i_iepsx_mj, axis=1),
                axes=1) / (G_y)
        epsxy_mnj = epsxy_fft[:, G_y // 2 + 1 - N_y:G_y // 2 + N_y, :]

        E4 = np.zeros((N_x, N_y, N_x, N_y), dtype=complex);
        for pp in range(N_x):
            for qq in range(N_x):
                E4[pp, :, qq, :] = linalg.toeplitz(epsxy_mnj[pp,
                    N_y - 1:2 * N_y, qq], np.flip(epsxy_mnj[pp, :N_y, qq]))
        self.fft_eps_ix = np.reshape(E4, (N_t, N_t), order='F')

        i_iepsy_nl = np.zeros((G_x, N_y, N_y), dtype=complex)
        iepsy_fft = fft.fftshift(fft.fft(1 / grid, axis=1),
                axes=1) / (G_y)

        for pp in range(G_x):
            iepsy_n = iepsy_fft[pp, G_y // 2 - N_y + 1:G_y // 2 + N_y]
            iepsy_nl = linalg.toeplitz(iepsy_n[N_y - 1:2 * N_y],
                    np.flip(iepsy_n[:N_y]))
            i_iepsy_nl[pp, :, :] = linalg.inv(iepsy_nl)

        epsyx_fft = fft.fftshift(fft.fft(i_iepsy_nl, axis=0),
                axes=0) / (G_x)
        epsyx_mnl = epsyx_fft[G_x // 2 - N_x + 1:G_x // 2 + N_x, :, :]

        E4 = np.zeros((N_x,N_y,N_x,N_y), dtype=complex)
        for rr in range(N_y):
            for ss in range(N_y):
                E4[:, rr, :, ss] = linalg.toeplitz(epsyx_mnl[N_x - 1:2 * N_x - 1,
                    rr, ss], np.flip(epsyx_mnl[:N_x, rr, ss]))
        self.fft_eps_iy = np.reshape(E4, [N_t, N_t], order='F')

    def eig_solve(self, freq, K):
        N_t = K.x.shape[0]
        I = np.eye(N_t)
        EPS = self.fft_eps
        EPSxy = self.fft_eps_ix
        EPSyx = self.fft_eps_iy

        F11 = -K.x @ linalg.solve(EPS, K.y)
        F12 = I + K.x @ linalg.solve(EPS, K.x)
        F21 = -I - K.y @ linalg.solve(EPS, K.y)
        F22 = K.y @ linalg.solve(EPS, K.x)
        self.F = np.block([[F11, F12], [F21, F22]])

        G11 = -K.x @ K.y
        G12 = EPSyx + K.x ** 2
        G21 = -EPSxy - K.y ** 2
        G22 = K.x @ K.y
        self.G = np.block([[G11, G12], [G21, G22]])

        if not np.count_nonzero(EPS - np.diag(np.diag(EPS))):
            self.U = np.eye(2 * N_t)
            k_z = np.lib.scimath.sqrt(np.diag(EPS) - np.diag(K.intensity()))
            self.gamma = 1j * np.hstack((k_z, k_z))
        else:
            (Q, self.U) = linalg.eig(self.F @ self.G)
            self.gamma = sp.sqrt(Q)

        self.V = -self.G @ self.U / self.gamma

    def compute_eigs(self, freq, k, period, N):
        k0 = 2 * np.pi * freq
        m = bm.Vector2d(np.arange(-(N.x // 2), N.x // 2 + 1,
            dtype=int), np.arange(-(N.y // 2), N.y // 2 + 1,
                dtype=int))
        ki = k0 * k + 2 * np.pi * m / period
        K = 1j * ki.grid().flatten().diag() / k0
        self.ffts(self.grid(period, value='eps'), N)
        self.eig_solve(freq, K)
        self.X = np.diag(np.exp(-k0 * self.gamma * self.h))
        self.K = K
        self.freq = freq
        self.period = period
        self.N = N

    def get_fields(self, pts, amplitudes, components=''):
        if not self.K or not self.period or not self.N:
            raise AttributeError('Solve the eigenmodes first by calling Layer.eigs')
            return

        N_t = self.N.x * self.N.y
        k0 = 2 * np.pi * self.freq

        if isinstance(pts, Vector3d):
            e_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            e_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            e_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            h_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            h_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            h_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            E = Vector3d(e_x, e_y, e_z)
            H = Vector3d(h_x, h_y, h_z)
            for (k, z) in enumerate(pts.z.flatten()):
                p_z = np.exp(-k0 * self.gamma * z)
                a_xy = np.zeros(amplitudes.vstack().shape, dtype=complex)
                a_xy[:, 0] = p_z * amplitudes.vstack()[:, 0]
                if amplitudes.x.shape[1] == 2:
                    a_xy[:, 1] *= self.X @ (1 / p_z) *\
                            (amplitudes.vstack()[:, 1])
                pts_xy = Vector2d(*pts.data[:2])
                (E[:, :, k], H[:, :, k]) = self.get_fields(pts_xy,
                        Vector2d(a_xy[:N_t], a_xy[N_t:]), components)
        else:
            U_xy = self.U
            V_xy = self.V
            U_z = linalg.solve(self.fft_eps, self.K.rotate(np.pi / 2).hstack())
            V_z = self.K.rotate(np.pi / 2).hstack()

            if amplitudes.x.shape[1] == 2:
                U_xy = np.hstack((U_xy, U_xy))
                V_xy = np.hstack((V_xy, -V_xy))
                U_z = np.hstack((U_z, -U_z))
                V_z = np.hstack((V_z, V_z))

            u_xy = U_xy @ amplitudes.vstack()
            v_xy = V_xy @ amplitudes.vstack()
            u_z = U_z @ v_xy
            v_z = V_z @ u_xy

            e_x = np.zeros((pts.x.size, pts.y.size), dtype=complex)
            e_y = np.zeros((pts.x.size, pts.y.size), dtype=complex)
            e_z = np.zeros((pts.x.size, pts.y.size), dtype=complex)
            h_x = np.zeros((pts.x.size, pts.y.size), dtype=complex)
            h_y = np.zeros((pts.x.size, pts.y.size), dtype=complex)
            h_z = np.zeros((pts.x.size, pts.y.size), dtype=complex)
            E = Vector3d(e_x, e_y, e_z)
            H = Vector3d(h_x, h_y, h_z)

            for (j, yy) in enumerate(pts.y):
                r = Vector2d(pts.x, yy)
                k_phase = np.exp(-k0 * self.K.diag().dot(r))
                E.x[:, [j]] = k_phase @ u_xy[:N_t]
                E.y[:, [j]] = k_phase @ u_xy[N_t:]
                H.x[:, [j]] = k_phase @ v_xy[:N_t]
                H.y[:, [j]] = k_phase @ v_xy[N_t:]
                E.z[:, [j]] = k_phase @ u_z
                H.z[:, [j]] = k_phase @ v_z

        return (E, H)

