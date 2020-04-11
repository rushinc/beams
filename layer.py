import numpy as np
import scipy as sp
import beams as bm
from numpy import fft
from scipy import linalg
from beams import *

class Layer:
    def __init__(self, z=0., resolution=None,
            shapes=[], material=Material()):
        self.z = float(z) if type(z) is int else z
        self.resolution = resolution
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

    def fft(self, grid, N_modes):
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

    def mode_solve(self, K):
        I = np.eye(K.x.shape[0])
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

        (Q, self.W) = linalg.eig(self.F @ self.G)
        self.gamma = sp.sqrt(Q)
        self.V = -self.G @ self.W / self.gamma

    def eigs(self, freq, k, period, N_modes):
        self.freq = freq
        self.period = period
        self.N_modes = N_modes
        k0 = 2 * np.pi * freq
        m = bm.Vector2d(np.arange(-(N_modes.x // 2), N_modes.x // 2 + 1,
            dtype=int), np.arange(-(N_modes.y // 2), N_modes.y // 2 + 1,
                dtype=int))
        ki = k0 * k + 2 * np.pi * m / period
        self.ki = ki
        K = 1j * ki.fgrid().diag() / k0
        self.fft(self.grid(period, value='eps'), N_modes)
        self.mode_solve(K)

        return (self.gamma, self.W, self.V)

    def mode(self, index, res): 
        try:
            N_t = self.N_modes.x * self.N_modes.y
            gamma = self.gamma[index]
            e_mode = Vector2d(self.W[:N_t, index], self.W[N_t:, index])
            h_mode = Vector2d(self.V[:N_t, index], self.V[N_t:, index])
            px = self.period.x
            py = self.period.y
        except AttributeError as ae:
            print(type(ae))
            print("Please solve the eigenmodes for the layer first.")
        except IndexError as ie:
            print(type(ie))
            print("Mode index " + str(index) +
                    " is greater than the number of modes solved for "
                    + str(self.N_modes))

        grid = np.ogrid[-px / 2 : px / 2 : 1 / res,
                -py / 2 : py / 2 : 1 / res]
        e_x = np.zeros((grid[0].size, grid[1].size), dtype=complex)
        e_y = np.zeros((grid[0].size, grid[1].size), dtype=complex)
        h_x = np.zeros((grid[0].size, grid[1].size), dtype=complex)
        h_y = np.zeros((grid[0].size, grid[1].size), dtype=complex)
        for (i, xx) in enumerate(grid[0]):
            for (j, yy) in enumerate(grid[1][0]):
                r = Vector2d(xx, yy)
                k_phase = Vector2d(np.exp(1j * self.ki.fgrid().x * r.x),
                        np.exp(1j * self.ki.fgrid().y * r.y))
                e_field = k_phase @ e_mode
                h_field = k_phase @ h_mode
                e_x[i, j] = e_field.x
                e_y[i, j] = e_field.y
                h_x[i, j] = h_field.x
                h_y[i, j] = h_field.y

        return (Vector2d(e_x, e_y), Vector2d(h_x, h_y))

