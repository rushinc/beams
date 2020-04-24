import numpy as np
import beams as bm
from numpy import fft
from numpy.lib.scimath import sqrt
from numpy import linalg as la
import time
from beams import *

class Layer:
    def __init__(self, h=0, shapes=None, material=Material(), resolution=None):
        self._h = float(h)
        self._res = to_vec2(resolution)
        self._dispersive = False

        if not shapes:
            self._shapes = []
        else:
            self._shapes = shapes
            for s in shapes:
                if s.material.dispersive:
                    self._dispersive = True

        self._material = material
        if material.dispersive:
            self._dispersive = True
        self.reset()

    def reset(self):
        self._period = None
        self._N = None
        self._K = None
        self._k = None
        self._freq = None
        self._fft_eps = None
        self._fft_eps_ix = None
        self._fft_eps_iy = None
        self.U = None
        self.V = None
        self.gamma = None
        self._reset = True

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, new_m):
        self._material = new_m
        if new_m.dispersive: self._dispersive = True
        self.reset()

    @property
    def shapes(self):
        return self._shapes

    @shapes.setter
    def shapes(self, new_shapes):
        self._shapes = shapes
        if not self.material.dispersive: self._dispersive = False
        for s in new_shapes:
            if s.material.dispersive:
                self._dispersive = True
        if not self._reset: self.reset()

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, new_h):
        self._h = float(h)

    @property
    def resolution(self):
        return self._res

    @resolution.setter
    def resolution(self, new_res):
        self._res = new_res
        if not self._reset: self.reset()

    @property
    def dispersive(self):
        return self._dispersive

    @property
    def period(self):
        return self._period

    @property
    def freq(self):
        return self._freq

    @property
    def k(self):
        return self._k

    @property
    def K(self):
        return self._K

    @property
    def N(self):
        return self._N

    @property
    def X(self):
        N_t = self.N.x * self.N.y
        if self._h == 0: return np.eye(2 * N_t)
        k0 = 2 * np.pi * self.freq
        return np.diag(np.exp(-k0 * self.gamma * self.h))

    def __repr__(self):
        s = self.__str__()
        s0 = "Layer with "
        if "\np" in s:
            s.replace("\np", "\nFFT computed for p")
        else:
            s += "\nNo FFT matrices stored"
        if "\nf" in s:
            s.replace("\nf", "\nEigenvalues for f")
        else:
            s += "\nNo eigenvalue solutions stored"
        return s0 + s

    def __str__(self):
        s = "h = " + str(self.h) + ",\nshapes = " + str(self.shapes)
        s += ",\nbackground = " + str(self.material)
        s += ",\nresolution = " + str(self.resolution)
        if self.period and self.N:
            s += "\np = " + str(self.period) + ", N = " + str(self.N)
        if self.freq and self.k:
            s += "\nf = " + str(round(self.freq, 3)) + ", k = " + str(self.k)
        return s

    def grid(self, resolution=None, period=None, freq=None, feature='shape'):
        if not period:
            if not self.period:
                raise AttributeError('period of Layer object has not been initialized yet')
            else:
                p = self.period
        else:
            p = to_vec2(period)

        if resolution:
            res = to_vec2(resolution)
        elif self.resolution:
            res = self.resolution
        else:
            raise AttributeError('resolution of Layer object has not been initialized')

        if not freq: freq = self.freq

        grid = np.ogrid[-p.x / 2 : p.x / 2 : 1 / res.x,
                -p.y / 2 : p.y / 2 : 1 / res.y]
        vgrid = to_vec2(grid)
        layout = np.zeros((grid[0].size, grid[1].size), dtype=int)
        for i, s in list(enumerate(self.shapes))[::-1]:
            interior_indices = s.interior(vgrid)
            layout[np.logical_and(layout == 0, interior_indices)]\
                    = i + 1

        if feature != 'shape': 
            feat_grid = np.full(layout.shape, self.material.get(feature, freq))
            for i, s in enumerate(self.shapes):
                feat_grid[layout == i + 1] = s.material.get(feature, freq)
            return (vgrid.flatten(), feat_grid)

        return (vgrid.flatten(), layout)

    def __ffts(self, period, N, freq=None):
        N_x = int(N.x)
        N_y = int(N.y)
        N_t = N_x * N_y

        if not freq: freq = self.freq

        if self.resolution:
            res = self.resolution
        else:
            res = (2 * N - 1) / period

        if not self.shapes:
            self._fft_eps = self.material.get('eps', freq) * np.eye(N_t)
            self._fft_eps_ix = self.material.get('eps', freq) * np.eye(N_t)
            self._fft_eps_iy = self.material.get('eps', freq) * np.eye(N_t)

            self._res = res
            self._period = period
            self._N = N
            return

        _, grid = self.grid(res, period, freq, 'eps')
        (G_y, G_x) = grid.shape

        EPS = np.zeros((N_t, N_t), dtype=complex)
        eps_fft = fft.fftshift(fft.fft2(grid)) / (G_x * G_y)
        eps_mn = eps_fft[G_x // 2 - N_x + 1:G_x // 2 + N_x,
                G_y // 2 - N_y + 1:G_y // 2 + N_y]

        for pp in range(N_x):
            for qq in range(N_y):
                EPS[pp + N_x * qq, ::-1] = np.reshape(eps_mn[pp:pp + N_x,
                    qq:qq + N_y], (1, -1), order='F')
        self._fft_eps = EPS

        i_iepsx_mj = np.zeros((N_x, G_y, N_x), dtype=complex)
        iepsx_fft = fft.fftshift(fft.fft(1 / grid, axis=0),
                axes=0) / (G_x)

        for qq in range(G_y):
            iepsx_m = iepsx_fft[G_x // 2 - N_x + 1:G_x // 2 + N_x, qq]
            iepsx_mj = la.toeplitz(iepsx_m[N_x - 1:2 * N_x],
                    np.flip(iepsx_m[:N_x]))
            i_iepsx_mj[:, qq, :] = la.inv(iepsx_mj)

        epsxy_fft = fft.fftshift(fft.fft(i_iepsx_mj, axis=1),
                axes=1) / (G_y)
        epsxy_mnj = epsxy_fft[:, G_y // 2 + 1 - N_y:G_y // 2 + N_y, :]

        E4 = np.zeros((N_x, N_y, N_x, N_y), dtype=complex);
        for pp in range(N_x):
            for qq in range(N_x):
                E4[pp, :, qq, :] = la.toeplitz(epsxy_mnj[pp,
                    N_y - 1:2 * N_y, qq], np.flip(epsxy_mnj[pp, :N_y, qq]))
        self._fft_eps_ix = np.reshape(E4, (N_t, N_t), order='F')

        i_iepsy_nl = np.zeros((G_x, N_y, N_y), dtype=complex)
        iepsy_fft = fft.fftshift(fft.fft(1 / grid, axis=1),
                axes=1) / (G_y)

        for pp in range(G_x):
            iepsy_n = iepsy_fft[pp, G_y // 2 - N_y + 1:G_y // 2 + N_y]
            iepsy_nl = la.toeplitz(iepsy_n[N_y - 1:2 * N_y],
                    np.flip(iepsy_n[:N_y]))
            i_iepsy_nl[pp, :, :] = la.inv(iepsy_nl)

        epsyx_fft = fft.fftshift(fft.fft(i_iepsy_nl, axis=0),
                axes=0) / (G_x)
        epsyx_mnl = epsyx_fft[G_x // 2 - N_x + 1:G_x // 2 + N_x, :, :]

        E4 = np.zeros((N_x,N_y,N_x,N_y), dtype=complex)
        for rr in range(N_y):
            for ss in range(N_y):
                E4[:, rr, :, ss] = la.toeplitz(epsyx_mnl[N_x - 1:2 * N_x - 1,
                    rr, ss], np.flip(epsyx_mnl[:N_x, rr, ss]))
        self._fft_eps_iy = np.reshape(E4, [N_t, N_t], order='F')

        self._res = res
        self._period = period
        self._N = N

    def __eigs(self, freq, K):
        N_t = K.x.shape[0]
        I = np.eye(N_t)
        EPS = self._fft_eps
        EPSxy = self._fft_eps_ix
        EPSyx = self._fft_eps_iy

        F11 = -K.x @ la.solve(EPS, K.y)
        F12 = I + K.x @ la.solve(EPS, K.x)
        F21 = -I - K.y @ la.solve(EPS, K.y)
        F22 = K.y @ la.solve(EPS, K.x)
        self.L_eh = np.block([[F11, F12], [F21, F22]])

        G11 = -K.x @ K.y
        G12 = EPSyx + K.x ** 2
        G21 = -EPSxy - K.y ** 2
        G22 = K.x @ K.y
        self.L_he = np.block([[G11, G12], [G21, G22]])

        if EPS.size > 1 and not np.count_nonzero(EPS - np.diag(np.diag(EPS))):
            self.U = np.eye(2 * N_t)
            k_z = sqrt(np.diag(EPS) - np.diag(K.intensity()))
            k_z = np.conj(k_z)
            self.gamma = 1j * np.hstack((k_z, k_z))
        else:
            (Q, self.U) = la.eig(self.L_eh @ self.L_he)
            self.gamma = sqrt(Q)

        self.V = -self.L_he @ self.U / self.gamma
        self._K = K
        self._freq = freq

    def compute_eigs(self, freq, k, period, N):
        if (self.period != period or self.N != N
                or self.freq != freq or self.k != k):
            if self.period != period or self.N != N or self.dispersive:
                self.__ffts(period, N, freq)
            k0 = 2 * np.pi * freq
            m = bm.Vector2d(np.arange(-(N.x // 2), N.x // 2 + 1,
                dtype=int), np.arange(-(N.y // 2), N.y // 2 + 1,
                    dtype=int))
            ki = k0 * k + 2 * np.pi * m / period
            K = 1j * ki.grid().flatten().diag() / k0
            self.__eigs(freq, K)
            self._k = k

    def get_fields(self, pts, amplitudes, components=''):
        if not self.K or not self.period or not self.N:
            raise AttributeError('Solve the eigenmodes first ' +\
                    'by calling Layer.eigs')
            return

        N_t = self.N.x * self.N.y
        k0 = 2 * np.pi * self.freq

        if isinstance(pts, Vector3d):
            pts_xy = Vector2d(*pts.data[:2])
            e_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            e_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            e_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            h_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            h_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            h_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
            E = Vector3d(e_x, e_y, e_z)
            H = Vector3d(h_x, h_y, h_z)
            for (k, z) in enumerate(pts.z.flatten()):
                if self.h != 0 and (z > self.h or z < 0):
                    raise RuntimeWarning('Points outside layer. ' +\
                            'Solutions may be divergent.')
                p_z = np.exp(-k0 * self.gamma * z)
                c = np.array(amplitudes.vstack(), dtype=complex)
                c[:, 0] *= p_z
                if amplitudes.x.shape[1] == 2:
                    c[:, 1] *= self.X @ (1 / p_z)
                    (E[:, :, k], H[:, :, k]) = self.get_fields(pts_xy,
                            Vector2d(c[:N_t, :], c[N_t:, :]), components)
                else:
                    (E[:, :, k], H[:, :, k]) = self.get_fields(pts_xy,
                            Vector2d(c[:N_t], c[N_t:]), components)
        else:
            U_xy = self.U
            V_xy = self.V
            U_z = la.solve(self._fft_eps,
                    self.K.rotate(np.pi / 2).hstack()) @ self.V
            V_z = self.K.rotate(np.pi / 2).hstack() @ self.U

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

    def fft_convergence(self, max_res, n_res, N, period, n_iter=3):
        N = to_vec2(N)
        max_res = to_vec2(max_res)
        res = bm.Vector2d(np.linspace(2 * N.x - 1, max_res.x, n_res + 1),
                np.linspace(2 * N.y - 1, max_res.y, n_res + 1))
        DT = np.zeros(n_res)
        D = np.zeros(n_res)
        self.resolution = res[0]
        self.__ffts(period, N)
        for i, r in enumerate(res[1:]):
            EPS = self._fft_eps.copy()
            EPSxy = self._fft_eps_ix.copy()
            EPSyx = self._fft_eps_iy.copy()
            self.resolution = r
            t0 = time.clock_gettime_ns(0)
            for _ in range(n_iter):
                self.__ffts(period, N)
            t1 = time.clock_gettime_ns(0)
            DT[i] = (t1 - t0) * 1e-9 / n_iter
            d_eps = la.norm(self._fft_eps - EPS)
            d_eps_x = la.norm(self._fft_eps_ix - EPSxy)
            d_eps_y = la.norm(self._fft_eps_iy - EPSyx)
            D[i] = max(d_eps, d_eps_x, d_eps_y)
            print("Sim " + str(i + 1) + ": res = " + str(r)
                    + "\nTime = " + str(round(DT[i], 3))
                    + ", diff = " + str(round(D[i], 5)))
        return (D, DT)

