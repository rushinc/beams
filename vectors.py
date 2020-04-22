import numpy as np

class Vector(object):
    def __init__(self, *args, **kwargs):
        self.data = []
        for arg in args:
            self.data.append(np.array(arg, **kwargs))
        self.dim = len(self.data)

    def __getitem__(self, indices):
        return self.__class__(*[a.__getitem__(indices) for a in self.data])

    def __setitem__(self, indices, other):
        return self.__class__(*[a.__setitem__(indices, b) for (a, b) in zip(self.data, other.data)])

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

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self.__class__(*[a @ b for (a, b) in zip(self.data, other.data)])
        return self.__class__(*[a @ other for a in self.data])

    def __rmatmul__(self, other):
        return self.__class__(*[other @ a for a in self.data])

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
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, Vector):
            return self.intensity() < other.intensity()
        return self.length() < other

    def __le__(self, other):
        if isinstance(other, Vector):
            return self.intensity() <= other.intensity()
        return self.intensity() <= other

    def __gt__(self, other):
        if isinstance(other, Vector):
            return self.intensity() > other.intensity()
        return self.intensity() > other

    def __ge__(self, other):
        if isinstance(other, Vector):
            return self.intensity() >= other.intensity()
        return self.intensity() >= other

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

    def dot(self, other):
        return sum([a * b for (a, b) in zip(self.data, other.data)])

    def norm(self):
        return float(np.linalg.norm(self.data))

    def intensity(self):
        return sum([np.abs(a) ** 2 for a in self.data])

    def length(self):
        return np.sqrt(sum([np.abs(a) ** 2 for a in self.data]))

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

    def flatten(self):
        return self.__class__(*[a.flatten() for a in self.data])

    def grid(self):
        return self.__class__(*np.meshgrid(*self.data))

    def hstack(self, other=None):
        if not other:
            return np.hstack(self.data)
        else:
            return self.__class__(*[np.hstack((a, b)) for (a, b)\
                    in zip(self.data, other.data)])

    def vstack(self, other=None):
        if not other:
            return np.vstack(self.data)
        else:
            return self.__class__(*[np.vstack((a, b)) for (a, b)\
                    in zip(self.data, other.data)])

class Vector2d(Vector):
    def __init__(self, x=0., y=0., **kwargs):
        if 'xy' in kwargs.keys():
            v = kwargs.pop('xy')
            x = v; y = v 
        super().__init__(x, y, **kwargs)

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, x_p):
        self.data[0] = np.array(x_)

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, y_p):
        self.data[1] = np.array(y_p)

    def rotate(self, angle):
        new_x = np.cos(angle) * self.x - np.sin(angle) * self.y
        new_y = np.sin(angle) * self.x + np.cos(angle) * self.y
        return Vector2d(new_x, new_y)

class Vector3d(Vector):
    def __init__(self, x=0., y=0., z=0., **kwargs):
        if 'xy' in kwargs.keys():
            v = kwargs.pop('xy')
            x = v; y = v 
        if 'yz' in kwargs.keys():
            v = kwargs.pop('yz')
            z = v; y = v 
        if 'zx' in kwargs.keys():
            v = kwargs.pop('zx')
            x = v; z = v 
        if 'xyz' in kwargs.keys():
            v = kwargs.pop('xyz')
            x = v; y = v; z = v 
        super().__init__(x, y, z, **kwargs)

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, x_p):
        self.data[0] = np.array(x_p)

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, y_p):
        self.data[1] = np.array(y_p)

    @property
    def z(self):
        return self.data[2]

    @z.setter
    def z(self, z_p):
        self.data[2] = np.array(z_p)

