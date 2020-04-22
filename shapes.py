import numpy as np
import beams as bm
from beams import *

class Shape:
    def __init__(self, interior, center=Vector2d(), material=bm.Material()):
        self.material = material
        self.center = center
        self.interior = interior

    def __contains__(self, point):
        return self.interior(point)

class Rectangle(Shape):
    def __init__(self, size, **kwargs):
        self.size = to_vec2(size)
        super().__init__(interior=self.interior, **kwargs)

    def interior(self, pt):
        return np.logical_and(abs(pt.x - self.center.x) <= self.size.x / 2,
                abs(pt.y - self.center.y) <= self.size.y / 2)

class Ellipse(Shape):
    def __init__(self, r, **kwargs):
        self.radii = to_vec2(r)
        super().__init__(interior=self.interior, **kwargs)

    def interior(self, pt):
        return (pt - self.center) / self.radii <= 1

