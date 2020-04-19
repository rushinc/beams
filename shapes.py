import numpy as np
import beams as bm

class Shape:
    def __init__(self, material=bm.Material(),
            center=bm.Vector2d(), interior=None):
        self.material = material
        self.center = center
        self.interior = interior

    def __contains__(self, point):
        return self.interior(point)

class Rectangle(Shape):
    def __init__(self, size=bm.Vector2d(), **kwargs):
        self.size = size
        super().__init__(interior=self.interior, **kwargs)

    def interior(self, pt):
        return np.logical_and(abs(pt.x - self.center.x) <= self.size.x / 2,
                abs(pt.y - self.center.y) <= self.size.y / 2)

class Ellipse(Shape):
    def __init__(self, radii=bm.Vector2d(), **kwargs):
        self.radii = radii
        super().__init__(interior=self.interior, **kwargs)

    def interior(self, pt):
        return (pt - self.center) / self.radii <= 1

