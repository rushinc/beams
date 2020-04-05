import numpy as np
import scipy as sp
from scipy import linalg
import beams as bm
from beams import *

class Solver:
    def __init__(self, period, freqs, k_vals, N_modes, layers):
        self.p = period
        self.f_list = freqs
        self.k_list = k_vals
        self.N = N_modes
        self.layers = layers
        self.m = Vector2d(np.arange(-(N_modes.x / 2), N_modes.x // 2, dtype=int),
                np.arange(-(N_modes.y // 2), N_modes.y // 2, dtype=int))
        self.m0 = Vector2d(N_modes.x // 2, N_modes.y // 2)

    def solve_at(self, freq, k):
        k0 = 2 * pi * freq
        ki = k0 * k + 2 * pi * self.m / self.p

