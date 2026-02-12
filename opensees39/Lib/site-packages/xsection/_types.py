from typing import Protocol

class _Basic: ...

class _Shape(Protocol):
    def exterior(self): ... 
    def interior(self): ...
    def triangles(self): ...

class _Fiber(_Basic, _Shape):
    pass

class _Model(_Fiber):
    pass

class SectionType(Protocol):
    _composite: bool

    def is_fiber(self): ...
    
    def is_shape(self): ...

    def is_model(self): ...

    def is_composite(self): ...

    @property
    def model(self)->"_Model": ...

    @property
    def elastic(self)->"ElasticConstants": ...
    
    @property
    def plastic(self)->"PlasticConstants": ...

    @property 
    def warping(self)->"WarpingConstants": ...

    @property
    def e(self): pass # e["nn"]
    @property
    def p(self): pass
    @property
    def w(self): pass

    #
    def _create_model(self):
        pass

    def _create_fiber(self):
        pass

    def _create_shape(self):
        pass

    def _create_basic(self):
        pass


class Polygon(SectionType):
    _composite: bool = False
    pass


class Annulus(SectionType):
    _composite: bool = False
    pass

class Circle(SectionType):
    _composite: bool = False
    pass

