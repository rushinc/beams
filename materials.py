from collections import Callable

class Material:
    def __init__(self, epsilon=1., mu=1., index=None):
        self.eps = float(epsilon) if type(epsilon) is int else epsilon
        self.mu = float(mu) if type(mu) is int else mu 
        if index:
            self.eps = float(index) ** 2

    def get_eps(self, freq=None):
        if freq and isinstance(self.eps, Callable):
            return self.eps(freq)
        else:
            return self.eps
