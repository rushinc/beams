import numpy as np

class Vector(object):
    def __init__(self, *args):
        self.data = []
        for arg in args:
            self.data.append(np.array(arg, dtype=np.float64))

    def __add__(self, other):
        if isinstance(other, Vector):
            return self.__class__(*[a + b for (a, b)
                in zip(self.data, other.data)])
        return self.__class__(*[a + other for a in self.data])

    def __radd__(self, other):
        return self.__class__(*[other + a for a in self.data])

    def __sub__(self, other):
        if isinstance(other, Vector):
            return self.__class__(*[a - b for (a, b) in zip(self.data, other.data)])
        return self.__class__(*[a - other for a in self.data])

    def __rsub__(self, other):
        return self.__class__(*[other - a for a in self.data])

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.__class__(*[a * b for (a, b) in zip(self.data, other.data)])
        return self.__class__(*[a * other for a in self.data])

    def __rmul__(self, other):
        return self.__class__(*[other * a for a in self.data])

    def __truediv__(self, other):
        if isinstance(other, Vector):
            return self.__class__(*[a / b for (a, b) in zip(self.data, other.data)])
        return self.__class__(*[a / other for a in self.data])

    def __rtruediv__(self, other):
        return self.__class__(*[other / a for a in self.data])

    def __floordiv__(self, other):
        if isinstance(other, Vector):
            return self.__class__(*[a // b for (a, b) in zip(self.data, other.data)])
        return self.__class__(*[a // other for a in self.data])

    def __rfloordiv__(self, other):
        return self.__class__(*[other // a for a in self.data])

    def __neg__(self):
        return self.__class__(*[-a for a in self.data])

    def __pos__(self):
        return self.__class__(*[+a for a in self.data])

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
        return "Vector(" + self.__str__() + ")"

    def __str__(self):
        s = ""
        for ele in self.data:
            s += ele.__str__().replace("\n", "      \n") + ",\n"
        return s[:-2]

    def ceil(self):
        return self.__class__(np.ceil(self.data))

    def floor(self):
        return self.__class__(np.floor(self.data))

    def dim(self):
        return len(self.data)

    def length(self):
        return float(np.linalg.norm(self.data))

    def unit(self):
        length = self.length()
        if length == 0.0:
            return Vector(np.zeros(self.data.shape()))
        return Vector(self.data/length)

    def diag(self):
        dvec = []
        for ele in self.data:
            try:
                ediag = np.diag(ele)
            except ValueError:
                ediag = ele
            dvec.append(ediag)
        return self.__class__(*dvec)

    def fgrid(self):
        flatg = []
        mgrid = np.meshgrid(*self.data)
        for g in mgrid:
            flatg.append(g.flatten())
        return self.__class__(*flatg)

    def mgrid(self):
        return self.__class__(*np.meshgrid(*self.data))

class Vector2d(Vector):
    def __init__(self, x=0., y=0.):
        self._x = x
        self._y = y
        super().__init__(x, y)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x_p):
        self._x = x_p
        self.data[0] = np.array(self._x, dtype=np.float64)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y_p):
        self._y = y_p
        self.data[1] = np.array(self._y, dtype=np.float64)

class Vector3d(Vector):
    def __init__(self, x=0., y=0., z=0.):
        self._x = x
        self._y = y
        super().__init__(x, y, z)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x_p):
        self._x = x_p
        self.data[0] = np.array(self._x, dtype=np.float64)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y_p):
        self._y = y_p
        self.data[1] = np.array(self._y, dtype=np.float64)

    @property
    def z(self):
        return self._z

    @y.setter
    def z(self, z_p):
        self._z = z_p
        self.data[2] = np.array(self._z, dtype=np.float64)

