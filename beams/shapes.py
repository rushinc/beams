import numpy as np
import beams as bm
from collections import Callable
from copy import copy
from beams import *

class Shape:
    def __init__(self, interior, center=Vector2d(), material=bm.Material()):
        if type(material) is materials.Material:
            self._material = material
        else:
            raise AttributeError('material must be an object of beams.materials.Material class')
        if isinstance(interior, Callable):
            self._interior = interior
        else:
            raise AttributeError('interior must be a function which takes one beams.vectors.Vector2d input and returns True only if the vector is a part of the shape')
        self._center = to_vec2(center)

    def __contains__(self, point):
        pt = to_vec2(point)
        return self.interior(pt)

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, new_material):
        if type(new_material) is materials.Material:
            self._material = new_material
        else:
            raise AttributeError('new_material must be an object of beams.materials.Material class')

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, new_center):
        self._center = to_vec2(new_center)

    @property
    def interior(self):
        return self._interior

    @interior.setter
    def interior(self, new_interior):
        if isinstance(new_interior, Callable):
            self._interior = new_interior
        else:
            raise AttributeError('new_interior must be a function which takes one beams.vectors.Vector2d input and returns True only if the vector is a part of the shape')

    def __repr__(self):
        return "beams.Shape:\n" + self.__str__().replace("\n", "\n       ")

    def __str__(self):
        s = "center = " + str(self.center)
        s += "\nmaterial = " + str(self.material).replace("\n", ", ")
        return s

    def __add__(self, other):
        other = to_vec2(other)
        new_shape = copy(self)
        new_center = self.center + other
        new_shape.center = new_center
        return new_shape

    def __radd__(self, other):
        other = to_vec2(other)
        new_shape = copy(self)
        new_center = self.center + other
        new_shape.center = new_center
        return new_shape

    def __sub__(self, other):
        other = to_vec2(other)
        new_shape = copy(self)
        new_center = self.center - other
        new_shape.center = new_center
        return new_shape

    def __rsub__(self, other):
        other = to_vec2(other)
        new_shape = copy(self)
        new_center = self.center - other
        new_shape.center = new_center
        return new_shape

class Rectangle(Shape):
    def __init__(self, size, **kwargs):
        self._size = to_vec2(size)
        super().__init__(interior=self.interior, **kwargs)

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return "beams.Rectangle:\n          " + self.__str__().replace("\n", "\n           ")

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

    @radii.setter
    def radii(self, new_r):
        r = to_vec2(new_r)
        self._r = r

    def __repr__(self):
        return "beams.Ellipse:\n         " + self.__str__().replace("\n", "\n         ")

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

