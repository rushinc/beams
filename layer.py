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

    def grid(self, period, resolution=None, freq=None, value='id'):
        px = period.x
        py = period.y
        res = resolution if resolution else self.resolution 
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

    def fft_toeplitz(self, grid, N_modes, inv='', property='eps'):
        N_x = int(N_modes.x)
        N_y = int(N_modes.y)
        N_t = N_x * N_y
        (G_x, G_y) = grid.shape
        EPS = np.zeros((N_t, N_t), dtype=complex)

        if inv == '':
            eps_fft = fft.fftshift(fft.fft2(grid))/(G_x*G_y)
            eps_mn = eps_fft[G_x // 2 - N_x + 1:G_x // 2 + N_x,
                    G_y // 2 - N_y + 1:G_y // 2 + N_y]
    
            for pp in range(N_x):
                for qq in range(N_y):
                    EE = np.rot90(eps_mn[pp:pp + N_x, qq:qq + N_y], 2)
                    EPS[N_y * pp + qq, :] = np.reshape(EE, -1)
            self.fft_eps = EPS

        elif inv == 'x':
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
            EPS = np.reshape(E4, [N_t, N_t])
            self.fft_eps_ix = EPS

        elif inv == 'y':
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
            EPS = np.reshape(E4, [N_t, N_t]);
            self.fft_eps_iy = EPS

        return EPS

    def mode_solve(self, K):
        I = np.eye(K.x.shape)
        EPS = self.fft_eps
        EPSxy = self.fft_eps_ix
        EPSyx = self.fft_eps_iy

        F11 = -K.x * linalg.solve(EPS, K.y)
        F12 = I + K.x * linalg.solve(EPS, K.x)
        F21 = -I - K.y * linalg.solve(EPS, K.y)
        F22 = K.y * linalg.solve(EPS, K.x)
        self.F = np.block([[F11, F12], [F21, F22]])

        G11 = -K.x * K.y
        G12 = EPSyx + K.x ** 2
        G21 = -EPSxy - K.y ** 2
        G22 = K.x * K.y
        self.G = np.block([[G11, G12], [G21, G22]])

        (Q, self.W) = linalg.eig(self.F * self.G)
        self.gamma = np.diag(sp.sqrt(Q))
        self.V = -G * self.W / Q

