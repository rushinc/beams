def to_vec(elements, **kwargs):
    if elements is None:
        return None
    if isinstance(elements, Vector):
        return elements
    else:
        try:
            self.elements = Vector(*elements)
        except TypeError:
            self.elements = Vector(elements)

def to_vec2(elements, **kwargs):
    if elements is None:
        return None
    if isinstance(elements, Vector2d):
        return elements
    else:
        try:
            if len(elements) >= 2:
                return Vector2d(*elements[:2], **kwargs)
            else:
                return Vector2d(xy=elements[0], **kwargs)
        except TypeError:
            return Vector2d(xy=elements, **kwargs)

def to_vec3(elements, **kwargs):
    if elements is None:
        return None
    if isinstance(elements, Vector3d):
        return elements
    else:
        try:
            if len(elements) >= 3:
                return Vector3d(*elements[:3], **kwargs)
            else:
                return Vector3d(xyz=elements[0], **kwargs)
        except TypeError:
            return Vector3d(xyz=elements, **kwargs)

from .materials import *
from .vectors import *
from .shapes import *
from .layer import *
from .solver import *
