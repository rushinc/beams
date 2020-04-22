from collections import Callable
from numpy.lib.scimath import sqrt

class Material:
    def __init__(self, epsilon=1., mu=1., **kwargs):
        self.eps = epsilon
        self.mu = mu
        for i, v in kwargs.items():
            if i == 'index': self.eps = v ** 2

        if isinstance(self.eps, Callable) or isinstance(self.mu, Callable):
            self.dispersive = True
        else:
            self.dispersive = False

    def get(self, material_property, freq=None):
        mp = material_property.lower()

        if mp in ['n', 'index']:
            if freq is not None and isinstance(self.eps, Callable):
                return sqrt(self.eps(freq))
            else:
                return sqrt(self.eps)

        if mp in ['eps', 'epsilon', 'permittivity']:
            if freq is not None and isinstance(self.eps, Callable):
                return self.eps(freq)
            else:
                return self.eps

        if mp in ['mu', 'permeability']:
            if freq is not None and isinstance(self.eps, Callable):
                return self.mu(freq)
            else:
                return self.mu
