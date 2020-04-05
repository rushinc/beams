import numpy as np

class Vector(object):
    def __init__(self, data):
        try:
            self.data = list(data)
        except TypeError:
            self.data = []
            self.data.append(data)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector([a + b for (a, b) in zip(self.data, other.data)])
        return Vector([a + other for a in self.data])

    def __radd__(self, other):
        return Vector([other + a for a in self.data])

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector([a - b for (a, b) in zip(self.data, other.data)])
        return Vector([a - other for a in self.data])

    def __rsub__(self, other):
        return Vector([other - a for a in self.data])

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector([a * b for (a, b) in zip(self.data, other.data)])
        return Vector([a * other for a in self.data])

    def __rmul__(self, other):
        return Vector([other * a for a in self.data])

    def __truediv__(self, other):
        if isinstance(other, Vector):
            return Vector([a / b for (a, b) in zip(self.data, other.data)])
        return Vector([a / other for a in self.data])

    def __rtruediv__(self, other):
        return Vector([other / a for a in self.data])

    def __floordiv__(self, other):
        if isinstance(other, Vector):
            return Vector([a // b for (a, b) in zip(self.data, other.data)])
        return Vector([a // other for a in self.data])

    def __rfloordiv__(self, other):
        return Vector([other // a for a in self.data])

    def __neg__(self):
        return Vector([-a for a in self.data])

    def __pos__(self):
        return Vector([+a for a in self.data])

    def __eq__(self, other):
        if isinstance(other, Vector):
            return np.allclose(self.data, other.data)
        return np.allclose(self.data, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.length() < other.length()

    def __le__(self, other):
        return self.length() <= other.length()

    def __gt__(self, other):
        return self.length() > other.length()

    def __ge__(self, other):
        return self.length() >= other.length()

    def __repr__(self):
        return "Vector" + self.__str__()

    def __str__(self):
        return self.data.__str__()

    def ceil(self):
        return Vector(np.ceil(self.data))

    def floor(self):
        return Vector(np.floor(self.data))

    def length(self):
        return float(np.linalg.norm(self.data))

    def unit(self):
        length = self.length()
        if length == 0.0:
            return Vector(np.zeros(self.data.shape()))
        return Vector(self.data/length)

class Vector2d(Vector):
    def __init__(self, x=0., y=0.):
        self._x = x
        self._y = y
        super().__init__([x, y])

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x_p):
        self._x = x_p
        self.data[0] = self._x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y_p):
        self._y = y_p
        self.data[1] = self._y
