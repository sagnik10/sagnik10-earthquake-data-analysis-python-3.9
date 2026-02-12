from .. import polygon
from ..polygon import PolygonSection
import numpy as np


class _SimpleShape(PolygonSection):
    _parameters: dict

    def __init__(self,
                 exterior,
                 interior=None,
                 angle=None,
                 mesh_size=None,
                 offset=None,
                 **kwds):
        self._mesh_kwds = {}
        if not hasattr(self, "_reference_angle"):
            self._reference_angle = 0


        if offset is not None:
            exterior = exterior + offset
            if interior is not None:
                interior = [i + offset for i in interior]
            self._offset = offset
        else:
            self._offset = np.zeros(2)

        if angle is not None:
            self._angle = angle
            exterior = polygon.rotate(exterior, angle)
            if interior is not None:
                interior = [polygon.rotate(i, angle) for i in interior]
        else:
            self._angle = 0


        super().__init__(
            exterior=exterior,
            interior=interior,
            mesh_size=mesh_size,
            **kwds)


    def translate(self, offset, **kwd):
        typ = type(self)

        kwds = {k: getattr(self, k) for k in typ._parameters}
        new = typ(**kwds,
                #   centroid=self.centroid+offset,
                  offset=self._offset+offset,
                  angle=self._angle,
                )
        new.material = self.material
        new.z        = self.z
        new.name     = self.name
        return new

    
    def rotate(self, angle:float=None, principal=None):

        if angle is None:
            if principal is None:
                raise ValueError("Must provide angle or principal axes")

            angle = 0 # self._principal_rotation()
            if principal[0] == "z":
                angle -= np.pi/2

        typ = type(self)

        kwds = {k: getattr(self, k) for k in typ._parameters}
        new  = typ(**kwds, 
                #    centroid=self.centroid,
                   offset=self._offset,
                   angle=angle + self._angle
                )
        # new.material = self.material
        new.z        = self.z
        new.name     = self.name
        return new


class HalfFlange(_SimpleShape):
    """A polygon representation of a half wide flange (T) section.
    """
    d: float
    "Depth of the section."
    bf: float
    "Width of the flange."
    tf: float
    "Thickness of the flange."
    tw: float
    "Thickness of the web."
    _parameters = ["d", "bf", "tf", "tw", "k", "mesh_scale"]

    def __init__(self, d, 
                 bf=None, 
                 tw=None, tf=None,
                 t=None, 
                 b=None,
                 k = None,
                 centroid=None,
                 mesh_scale=None,
                 **kwds):

        if centroid is None:
            centroid = np.zeros(2)

        if t is None and tf is None and tw is None:
            raise TypeError("Must provide t, tf, or tw")
        if b is None and bf is None:
            raise TypeError("Must provide b or bf")

        if t is not None and tf is not None and t != tf:
            raise ValueError("Got both t and tf")

        if t is not None and tw is not None and t != tw:
            raise ValueError("Got both t and tw")
        elif tf is None:
            tf = t
        elif t is None:
            t = tf
        
        if b is not None and bf is not None and b != bf:
            raise ValueError("Got both b and bf")
        elif b is None:
            b = bf 
        elif bf is None:
            bf = b

        self.d  = d
        self.bf = bf
        if tf is None:
            tf = t
        self.tf = tf

        if tw is None:
            tw = tf
        self.tw = tw

        self.k = k

        if mesh_scale is None:
            mesh_scale = 1/10
        self.mesh_scale = mesh_scale

        #
        #
        #
        bf = self.bf
        b2 = self.tw
        tf = self.tf
        t2 = self.tf
        tw = self.tw
        d  = self.d

        y_top = tf/2.0
        y_bot = -d - tf/2.0

        exterior = np.array([
            [-bf/2,   y_top],         # 1  top-left flange
            [ bf/2,   y_top],         # 2
            [ bf/2,   y_top - tf],    # 3  step down into web
            [ tw/2,   y_top - tf],    # 4
            [ tw/2,   y_bot + t2],    # 5  down web
            [ b2/2,   y_bot + t2],    # 6  step into bottom flange
            [ b2/2,   y_bot],         # 7
            [-b2/2,   y_bot],         # 8
            [-b2/2,   y_bot + t2],    # 9
            [-tw/2,   y_bot + t2],    # 10 up web
            [-tw/2,   y_top - tf],    # 11
            [-bf/2,   y_top - tf],    # 12
        ], dtype=float)

        super().__init__(mesh_size=min(tf, tw)*mesh_scale,
                         exterior = exterior, #+(centroid - polygon.centroid(exterior)),
                         interior = None,
                         **kwds)



class WideFlange(_SimpleShape):
    """A polygon representation of a wide flange (I) section.
    """
    d: float
    "Depth of the section."
    bf: float
    "Width of the flange."
    tf: float
    "Thickness of the flange."
    tw: float
    "Thickness of the web."
    k: float
    "Fillet vertical offset."

    _parameters = ["d", "bf", "tf", "tw", "k", "k1", "mesh_scale"]

    def __init__(self, d, 
                 bf=None, 
                 tw=None, tf=None,
                 t=None, 
                 b=None,
                 k = None,
                 k1 = None,
                 centroid=None,
                 mesh_scale=None,
                 **kwds):

        if centroid is None:
            centroid = np.zeros(2)

        # Shape parameters
        if t is None and tf is None and tw is None:
            raise TypeError("Must provide t, tf, or tw")
        if b is None and bf is None:
            raise TypeError("Must provide b or bf")

        if t is not None and tf is not None and t != tf:
            raise ValueError("Got both t and tf")

        if t is not None and tw is not None and t != tw:
            raise ValueError("Got both t and tw")
        elif tf is None:
            tf = t
        elif t is None:
            t = tf
        
        if b is not None and bf is not None and b != bf:
            raise ValueError("Got both b and bf")
        elif b is None:
            b = bf 
        elif bf is None:
            bf = b

        self.d  = d
        self.bf = bf
        if tf is None:
            tf = t
        self.tf = tf

        if tw is None:
            tw = tf
        self.tw = tw

        if k1 is not None and k is None:
            k = k1
        self.k = k
        if k is not None and k1 is None:
            k1 = k
        self.k1 = k1
        
        # Mesh generation
        if mesh_scale is None:
            mesh_scale = 1/10
        self.mesh_scale = mesh_scale

        #
        #
        #
        bf = self.bf
        b2 = self.bf
        tf = self.tf
        t2 = self.tf
        tw = self.tw
        d  = self.d

        y_top =  d / 2.0
        y_bot = -d / 2.0

        nk = 16
        if self.k is not None:
            exterior = np.array([
                [-bf/2,   y_top],                 # 1  top-left flange corner
                [ bf/2,   y_top],                 # 2  top-right flange corner
                [ bf/2,   y_top - tf],            # 3  step down under top flange
                # Top-right fillet under flange
                *([
                    [k1, y_top-k] + np.array([(k1-tw/2)*np.cos(angle), (k-tf)*np.sin(angle)])
                    for angle in np.linspace(np.pi/2, np.pi, nk)
                ]),
                # Bottom-right fillet above bottom flange
                *([
                    [k1, y_bot+k] + np.array([(k1-tw/2)*np.cos(angle), (k-tf)*np.sin(angle)])
                    for angle in np.linspace(np.pi, 3*np.pi/2, nk)
                ]),

                [ b2/2,   y_bot + t2],            # 6  into bottom flange
                [ b2/2,   y_bot],                 # 7
                [-b2/2,   y_bot],                 # 8
                [-b2/2,   y_bot + t2],            # 9

                # Bottom-left fillet
                *([
                    [-k1, y_bot+k] +np.array([(k1-tw/2)*np.cos(angle), (k-tf)*np.sin(angle)])
                    for angle in np.linspace(3*np.pi/2, 2*np.pi, nk)
                ]),
                # Top-left fillet
                *([
                    [-k1, y_top-k] + np.array([(k1-tw/2)*np.cos(angle), (k-tf)*np.sin(angle)])
                        for angle in np.linspace(0, np.pi/2, nk)
                ]),
                [-bf/2,   y_top - tf],
            ], dtype=float)
        else:
            exterior = np.array([
                [-bf/2,   y_top],         # 1  top-left flange
                [ bf/2,   y_top],         # 2  top-right flange
                [ bf/2,   y_top - tf],    # 3  step down under flange
                [ tw/2,   y_top - tf],    # 4  top web
                [ tw/2,   y_bot + t2],    # 5  down web
                [ b2/2,   y_bot + t2],    # 6  step into bottom flange
                [ b2/2,   y_bot],         # 7
                [-b2/2,   y_bot],         # 8
                [-b2/2,   y_bot + t2],    # 9
                [-tw/2,   y_bot + t2],    # 10 up web
                [-tw/2,   y_top - tf],    # 11
                [-bf/2,   y_top - tf],    # 12
            ], dtype=float)

        super().__init__(mesh_size=min(tf, tw)*mesh_scale,
                         exterior = exterior+polygon.centroid(exterior)-centroid,
                         interior = None,
                         **kwds)


#         # Area and moment of inertia
#         self.A  = tw*(d - tf - t2) + bf*tf + b2*t2

# #       Iy and Iz are wrong for tf !=  t2
# #       self.Iy = tw*(d - tf - t2)**3/12.0 + bf*tf*(0.5*(d - tf))**2 + b2*t2*(0.5*(d - t2)) ** 2
#         self.Iz = 2*tf*bf**3/12

    def shear_factor(self, nu=0.3):
        b  = self.bf
        tf = self.tf
        tw = self.tw
        d  = self.d

        m = 2*b*tf/(d*tw)
        n = b/d
        return (10*(1+nu)*(1+3*m)**2)/((12+72*m + 150*m**2 + 90*m**3) + nu*(11+66*m + 135*m**2 + 90*m**3) + 30*n**2*(m + m**2) + 5*nu*n**2*(8*m+9*m**2))


    # def _create_model(self, mesh_scale=None):
    #     """
    #     Saritas and Filippou (2009) "Frame Element for Metallic Shear-Yielding Members under Cyclic Loading"
    #     """
    #     b  = self.bf
    #     tf = self.tf
    #     tw = self.tw
    #     d  = self.d

    #     # Shear from Saritas and Filippou (2009)
    #     # Ratio of total flange area to web area
    #     alpha = 2*b*tf/d/(2*tw)
    #     # NOTE: This is 1/beta_S where beta_S is Afsin's beta
    #     beta = (1+3*alpha)*(2/3)/((1+2*alpha)**2-2/3*(1+2*alpha)+1/5)
    #     def psi(y, z):
    #         # webs
    #         if abs(y) < (d/2-tf):
    #             return 0 #beta*((1+2*alpha) - (2*y/d)**2) - 1 #+ 1
    #         # flange
    #         else:
    #             return 0 #beta*(2*alpha)*(z/b) - 1

    #     mesh = self._create_mesh(mesh_scale=mesh_scale)

    #     return TriangleModel.from_meshio(mesh, warp_shear=psi)


class Rectangle(_SimpleShape):
    """A polygon representation of a rectangle section.
    """
    _parameters = ["b", "d", "mesh_scale"]
    def __init__(self, b, d, centroid=None, mesh_scale=None, **kwds):
        self.b = b
        self.d = d
        if centroid is None:
            centroid = np.zeros(2)

        self.mesh_scale = mesh_scale or 1/10

        mesh_size = min(b,d)*self.mesh_scale

        super().__init__(mesh_size=mesh_size,
                         interior=None,
                         exterior =  np.array([
                            [-b / 2,  -d / 2],
                            [ b / 2,  -d / 2],
                            [ b / 2,   d / 2],
                            [-b / 2,   d / 2],
                        ])-centroid,
                        **kwds)


class Equigon(_SimpleShape):
    """A regular (equilateral and equiangular) polygon"""
    radius: float 
    "Radius of the section."
    mesh_scale: float
    "Scale factor for the mesh size."
    divisions: int
    "Number of divisions for the circle."

    _parameters = ["radius", "mesh_scale", "divisions"]
    def __init__(self,
                 radius, 
                 centroid=None,
                 mesh_scale=None,
                 divisions=60,
                 **kwds):

        self.radius = radius
        self.divisions = int(divisions)

        if centroid is None:
            centroid = np.zeros(2)

        self.mesh_scale = mesh_scale or 1/10

        rv = radius/np.cos(360/divisions/2*(np.pi/180))

        mesh_size = rv*self.mesh_scale

        o = 360/divisions/2*np.pi/180

        exterior = np.array([
            [rv*np.cos(theta), rv*np.sin(theta)]
            for theta in np.linspace(o, 2*np.pi+o, int(divisions)+1)
        ], dtype=float)

        super().__init__(mesh_size=mesh_size,
                         interior=None,
                         exterior = exterior - centroid,
                        **kwds)


class Circle(_SimpleShape):
    """A polygon representation of a circle section.
    """
    radius: float 
    "Effective radius of the section."
    mesh_scale: float
    "Scale factor for the mesh size."
    divisions: int
    "Number of divisions for the circle."
    _parameters = ["radius", "mesh_scale", "divisions"]
    def __init__(self,
                 radius,
                 centroid=None,
                 mesh_scale=None,
                 divisions=60,
                 **kwds):

        self.radius = radius
        self.divisions = int(divisions)

        if centroid is None:
            centroid = np.zeros(2)

        self.mesh_scale = mesh_scale or 1/10

        # area = np.pi*radius**2
        # s = np.sqrt(area/divisions/np.tan(np.pi/divisions))
        # re = s/(2*np.sin(np.pi/divisions))

        rv = radius*np.sqrt(np.pi/(divisions*np.sin(np.pi/divisions)*np.cos(np.pi/divisions)))

        self.d = rv*2

        mesh_size = rv*self.mesh_scale

        exterior = np.array([
            [rv*np.cos(theta), rv*np.sin(theta)]
            for theta in np.linspace(0, 2*np.pi, int(divisions)+1)
        ], dtype=float)

        super().__init__(mesh_size=mesh_size,
                         interior=None,
                         exterior = exterior-centroid,
                        **kwds)

class Ellipse(_SimpleShape):
    """A polygon representation of an ellipse section.
    """
    a1: float
    "Semi-major axis of the ellipse."
    a2: float
    "Semi-minor axis of the ellipse."
    mesh_scale: float
    "Scale factor for the mesh size."
    divisions: int
    "Number of divisions for the ellipse."
    _parameters = ["a1", "a2", "mesh_scale", "divisions"]
    def __init__(self,
                 d,
                 b=None,
                 centroid=None,
                 mesh_scale=None,
                 divisions=40,
                 **kwds):

        self.divisions = int(divisions)

        if b is None:
            b = d

        a1 = b / 2
        a2 = d / 2
        self.a1 = a1
        self.a2 = a2
        if centroid is None:
            centroid = np.zeros(2)

        self.mesh_scale = mesh_scale or 1/10

        # Calculate the mesh size based on the semi-major axis
        mesh_size = a1 * self.mesh_scale

        exterior = np.array([
            [a1 * np.cos(theta), a2 * np.sin(theta)]
            for theta in np.linspace(0, 2*np.pi, int(divisions)+1)
        ], dtype=float)

        super().__init__(mesh_size=mesh_size,
                         interior=None,
                         exterior=exterior-centroid,
                        **kwds)

class HollowRectangle(_SimpleShape):
    """A polygon representation of a hollow rectangle section.
    """
    d: float
    "Depth of the section."
    b: float
    "Width of the section."
    tf: float
    "Thickness of the flange."
    tw: float
    "Thickness of the web."

    _parameters = ["b", "d", "tf", "tw", "mesh_scale"]

    def __init__(self, b, d, t=None, centroid=None, mesh_scale=1/10, **kwds):
        self.b = b
        self.d = d
        self.tf = kwds.pop("tf", t)
        self.tw = kwds.pop("tw", t)
        self.mesh_scale = mesh_scale

        if centroid is None:
            centroid = np.zeros(2)

        exterior = np.array([
            [-b / 2, -d / 2],
            [ b / 2, -d / 2],
            [ b / 2,  d / 2],
            [-b / 2,  d / 2],
        ], dtype=float)

        tf, tw = self.tf, self.tw
        interior = [np.array([
                [-(b/2 - tw), -(d/2 - tf)],
                [ (b/2 - tw), -(d/2 - tf)],
                [ (b/2 - tw),  (d/2 - tf)],
                [-(b/2 - tw),  (d/2 - tf)],
            ], dtype=float)]
        super().__init__(mesh_size=min(tf, tw)*mesh_scale,
                         exterior= exterior-centroid,
                         interior=interior, **kwds)

class Pipe(_SimpleShape):
    _parameters = ["radius", "thickness", "centroid", "mesh_scale", "divisions"]
    def __init__(self, radius, thickness, centroid=None, mesh_scale=1/10, divisions=40, **kwds):
        
        self.radius = radius
        self.divisions = int(divisions)

        if centroid is None:
            centroid = np.zeros(2)

        self.mesh_scale = mesh_scale or 1/10

        rv = radius*np.sqrt(np.pi/(divisions*np.sin(np.pi/divisions)*np.cos(np.pi/divisions)))

        self.d = rv*2

        mesh_size = rv*self.mesh_scale

        exterior = np.array([
            [rv*np.cos(theta), rv*np.sin(theta)]
            for theta in np.linspace(0, 2*np.pi, int(divisions)+1)
        ], dtype=float)

        interior = np.array([
            [(rv-thickness)*np.cos(theta), (rv-thickness)*np.sin(theta)]
            for theta in np.linspace(0, 2*np.pi, int(divisions)+1)
        ], dtype=float)

        super().__init__(mesh_size=mesh_size,
                         interior=[interior-centroid],
                         exterior = exterior-centroid,
                        **kwds)


class Channel(_SimpleShape):
    """A polygon representation of a channel (C) section.
    """

    d: float
    "Depth of the section."
    b: float
    "Width of the flange."
    tf: float
    "Thickness of the flange."
    tw: float
    "Thickness of the web."

    _parameters = ["d", "b", "tf", "tw", "sf", "mesh_scale"]
    def __init__(self, d, b, tf, tw=None,
                 sf=0,
                #  centroid=None, 
                 mesh_scale=1/500,
                 **kwds):

        self.tf = tf
        self.tw = tw if tw is not None else tf
        self.d = d
        self.b = b
        self.sf = sf

        mesh_size = min(tf, tw)*mesh_scale
        self.mesh_scale = mesh_scale

        y_top =  d/2
        y_bot = -d/2

        exterior = np.array([
            [   - tw / 2, y_top],         # 1  top-left
            [ b - tw / 2, y_top],         # 2  top-right
            [ b - tw / 2, y_top - tf],    # 3  down into flange
            [     tw / 2, y_top - tf],    # 4  over to web
            [     tw / 2, y_bot + tf],    # 5  down web
            [ b - tw / 2, y_bot + tf],    # 6  out to bottom flange
            [ b - tw / 2, y_bot],         # 7  bottom-right
            [   - tw / 2, y_bot],         # 8  bottom-left
        ], dtype=float)


        # if centroid is not None:
        #     offset = (centroid - polygon.centroid(exterior))
        #     exterior = exterior + offset

        super().__init__(exterior=exterior,
                         interior=None,
                         mesh_size=mesh_size, **kwds)


class Angle(_SimpleShape):
    """A polygon representation of an angle (L) section.
   
    ^
    |_
    | |
    | |
    | |
    | |
    | |______
    |________|__>
    """
    b: float
    d: float
    _parameters = ["b", "d", "t", "k", "mesh_scale"]#, "angle"]
    def __init__(self, b, d, t, k=None,
                 centroid=None,
                #  angle=None,
                 mesh_scale=None,
                 **kwds):
        self.t = t
        self.b = b
        self.d = d
        if k is None:
            k = t 
        self.k = k

        if mesh_scale is None:
            mesh_scale = 1/100

        self.mesh_scale = mesh_scale

        # ____
        # |
        # |
        # 
        # exterior = np.array([
        #     [ b - t / 2,  t / 2],     # 1  outer top-right
        #     [-t / 2,      t / 2],     # 2  outer top-left
        #     [-t / 2, -d + t / 2],     # 3  outer bottom-left
        #     [ t / 2, -d + t / 2],     # 4  inner bottom-left of vertical leg
        #     [ t / 2,     -t / 2],     # 5  inner corner
        #     [ b - t / 2, -t / 2],     # 6  outer bottom-right of horizontal leg
        # ], dtype=float)

        if k-t > 0:
            nk = 8
            kt = k-t
            exterior = np.array([
                [   0,   0],
                [   b,   0],
                [   b,   t],
                # *(
                #     [b-kt, t-kt] + kt*np.array([np.cos(angle), np.sin(angle)])
                #     for angle in np.linspace(0, np.pi/2, nk)
                # ),
                # Inside fillet
                *(
                    [k,k] - (k-t)*np.array([np.cos(theta), np.sin(theta)])
                    for theta in np.linspace(np.pi/2, 0, nk)
                ),
                [   t,  d],
                # *(
                #     [t-kt, d-kt] + kt*np.array([np.cos(angle), np.sin(angle)])
                #     for angle in np.linspace(0, np.pi/2, nk) 
                # ),
                [0, d]
                # [   t,   d],
                # [   0,   d]
            ])
        else:
            exterior = np.array([
                [   0,   0],
                [   b,   0],
                [   b,   t],
                [   t,   t],
                [   t,   d],
                [   0,   d]
            ])

        exterior -= [t/2, t/2]

        super().__init__(mesh_size=min(t, b)*mesh_scale,
                         exterior=exterior, **kwds)
