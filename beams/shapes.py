import numpy as np
import beams as bm
from beams import *

class Shape:
    def __init__(self, interior, center=Vector2d(), material=bm.Material()):
        self._material = material
        self._center = to_vec2(center)
        self._interior = interior

    def __contains__(self, point):
        return self.interior(point)

    @property
    def material(self):
        return self._material

    @property
    def center(self):
        return self._center

    @property
    def interior(self):
        return self._interior

    def __repr__(self):
        return "Shape: " + self.__str__().replace("\n", "\n       ")

    def __str__(self):
        s = "center = " + str(self.center)
        s += "\nmaterial = " + str(self.material).replace("\n", ", ")
        return s

    def __add__(self, other):
        other = to_vec2(other)
        new_center = self.center + other
        return self.__class__(self.interior, new_center, self.material)

    def __radd__(self, other):
        other = to_vec2(other)
        new_center = self.center + other
        return self.__class__(self.interior, new_center, self.material)

    def __sub__(self, other):
        other = to_vec2(other)
        new_center = self.center - other
        return self.__class__(self.interior, new_center, self.material)

    def __rsub__(self, other):
        other = to_vec2(other)
        new_center = self.center - other
        return self.__class__(self.interior, new_center, self.material)

class Rectangle(Shape):
    def __init__(self, size, **kwargs):
        self._size = to_vec2(size)
        super().__init__(interior=self.interior, **kwargs)

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return "Rectangle: " + self.__str__().replace("\n", "\n           ")

    def __str__(self):
        s = "center = " + str(self.center) + ", size = " + str(self.size)
        s += "\nmaterial = " + str(self.material).replace("\n", ", ")
        return s

    def interior(self, pt):
        return np.logical_and(abs(pt.x - self.center.x) <= self.size.x / 2,
                abs(pt.y - self.center.y) <= self.size.y / 2)

class Ellipse(Shape):
    def __init__(self, r, **kwargs):
        self._r = to_vec2(r)
        super().__init__(interior=self.interior, **kwargs)

    @property
    def radii(self):
        return self._r

    def __repr__(self):
        return "Ellipse: " + self.__str__().replace("\n", "\n         ")

    def __str__(self):
        s = "center = " + str(self.center) + ", radii = " + str(self._r)
        s += "\nmaterial = " + str(self.material).replace("\n", ", ")
        return s

    def interior(self, pt):
        return (pt - self.center) / self.radii <= 1

class Custom(Shape):
    def __init__(self, array, size, **kwargs):
        self._raw = array
        self._size = size
        super().__init__(interior=self.interior, **kwargs)

    @property
    def array(self):
        return self._raw

    @array.setter
    def array(self, new_array):
        self._raw = new_array

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_size):
        self._size = new_size

    def interior(self, pts):
        r_pos = pts - self.center

