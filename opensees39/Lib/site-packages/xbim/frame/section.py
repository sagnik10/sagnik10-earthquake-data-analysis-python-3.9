from collections.abc import Mapping
from xsection import ElasticConstants, PlasticConstants

class _Shape:
    def exterior(self): ... 
    def interior(self): ...
    def triangles(self): ...


class _Fibers:
    pass


class _Model(_Fibers, _Shape):
    pass


class FrameSection:
    _is_fiber:     bool
    _is_shape:     bool
    _is_model:     bool


    def elastic(self)->"ElasticConstants": ...
    def plastic(self)->"PlasticConstants": ...

    def translate(self, location): ... 
    def rotate(self,    angle: float): ...
    def centroid(self): ...


    def create_fibers(self, mesh_scale=None, **kwds): ...

    @property
    def e(self): pass
    @property
    def p(self): pass
    @property
    def w(self): pass


    @property
    def u(self, args): pass

