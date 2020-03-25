import numpy as np
import scipy as sp
from scipy import linalg
import beams as bm
from beams import *

class Solver:
    def __init__(self, freqs, k_vals, N_modes, layers):
        self.fs = freqs
        self.ks = k_vals
        self.N = N_modes
        self.layers = layers
        self.m = vec2(np.arange(-(N_modes.x / 2), N_modes.x // 2, dtype=int),
                np.arange(-(N_modes.y // 2), N_modes.y // 2, dtype=int))
        self.m0 = vec2(N_modes.x // 2, N_modes.y // 2)

    def solve_at(self, freq, k)

