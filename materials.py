from collections import Callable
from numbers import Number
from numpy.lib.scimath import sqrt

class Material:
    def __init__(self, epsilon=1., mu=1., **kwargs):
        self._eps = epsilon
        self._mu = mu
        for i, v in kwargs.items():
            if i == 'index': self.eps = v ** 2

        if isinstance(self.eps, Callable) or isinstance(self.mu, Callable):
            self.dispersive = True
        else:
            self.dispersive = False

    def __repr__(self):
        return "Material: " + self.__str__().replace("\n", "\n          ")

    def __str__(self):
        if isinstance(self.eps, Number):
            s = "eps = " + str(round(self.eps, 3))
        else:
            s = "dispersive eps"
        if isinstance(self.mu, Number):
            s += "\nmu = " + str(round(self.mu, 3))
        else:
            s += "dispersive mu"
        return s

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, new_eps):
        self._eps = new_eps
        if isinstance(new_eps, Callable) or isinstance(self.mu, Callable):
            self.dispersive = True
        else:
            self.dispersive = False

    @property
    def mu(self):
        return self._mu

    @eps.setter
    def mu(self, new_mu):
        self._mu = new_mu
        if isinstance(new_mu, Callable) or isinstance(self.eps, Callable):
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
