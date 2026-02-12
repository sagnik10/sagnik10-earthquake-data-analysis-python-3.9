from ._simple import Circle

class Rebar:
    def __init__(self, system, units):
        self._system = system
        self._units = units


    def __getitem__(self, key):
        diameter = int(key)/8*self._units.inch
        return Circle(radius=diameter/2, mesh_scale=1/3, divisions=12, z=2)
