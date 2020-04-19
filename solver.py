import numpy as np
import scipy as sp
from scipy import linalg
import beams as bm
from beams import *

class Solver:
    def __init__(self, period, freqs, k_vals, N_modes, layers):
        self.p = period
        self.f_list = list(freqs)
        self.k_list = list(k_vals)
        self.N = N_modes
        self.layers = list(layers)

    def build(self, freq, k):
        L = len(self.layers)
        N_t = self.N.x * self.N.y
        B = np.zeros((4 * N_t * (L + 1), 4 * N_t * (L + 1)), dtype=complex)
        Z = np.zeros((2 * N_t, 2 * N_t), dtype=complex)
        for i, l in enumerate(self.layers):
            l.compute_eigs(freq, k, self.p, self.N)
            b_l = np.block([[l.W, l.W @ l.X], [l.V, -l.V @ l.X],
                [-l.W @ l.X, -l.W], [-l.V @ l.X, l.V]])
            B[4 * i * N_t:4 * (i + 2) * N_t,
                    4 * i * N_t:4 * (i + 1) * N_t] = b_l
            if i == 0:
                B[:4 * N_t, 4 * N_t * L:] = np.block([[-l.W, Z], [l.V, Z]])
            if i == L - 1:
                B[-4 * N_t:, 4 * N_t * L:] = np.block([[Z, l.W], [Z, l.V]])
        self.B = B

