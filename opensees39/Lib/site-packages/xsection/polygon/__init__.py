import sys
import itertools

import numpy as np 
from ..warping import WarpingSection
from shps.frame.solvers import TriangleModel
from shapely.geometry import Polygon as ShapelyPolygon
from .._interpolate import Q4


class PolygonSection(WarpingSection):
    """A homogeneous polygon section.
    """
    def __init__(self,
                 exterior,
                 interior=None,
                 mesh_size=None,
                 material=None,
                 z = 0,
                 name=None,
                 mesher=None):

        self.vertices   = self._exterior  = _sort_exterior(np.asarray(exterior))
        self._interior  = interior 
        # self._p_elastic = elastic


        self._mesher    = mesher
        self.z          = int(z)

        self._moic = None
        self._moig = None
        self._area = None
        self._p_model = None
        self._w_analysis = None
        self._model_kwds = {}
        self.material = material
        self.name = name
        self._shapely = None

        self.mesh_size = self._mesh_size = mesh_size
        if mesh_size is None and False:
            raise ValueError("mesh_size must be provided")
        
    def to_dict(self):
        """Convert the PolygonSection to a dictionary representation."""
        data = {
            "type": "Polygon",
            "exterior": self._exterior.tolist(),
            "interior": [i.tolist() if hasattr(i, "tolist") else i for i in self.interior()],
            "mesh_size": self._mesh_size,
            "material": self.material,
            "z": self.z,
            "name": self.name,
            "mesher": self._mesher
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Create a PolygonSection from a dictionary representation."""
        exterior = data["exterior"]
        interior = data.get("interior", None)
        mesh_size = data.get("mesh_size", None)
        if mesh_size is None:
            mesh_size = 0.1
        material = data.get("material", None)
        z = data.get("z", 0)
        name = data.get("name", None)
        mesher = data.get("mesher", None)

        return cls(exterior, interior, mesh_size, material, z, name, mesher)

    def save(self, file):
        """Save the PolygonSection to a file in JSON format."""
        import json
        data = self.to_dict()
        file.write(json.dumps(data, indent=4))

    def exterior(self, plane=False):
        return self._exterior

    def interior(self, plane=False):
        if self._interior is None:
            return []
        return self._interior
    
    def surface_point(self):
        if self._shapely is None:
            self._shapely = ShapelyPolygon(self.exterior(), self.interior())
        return self._shapely.representative_point().coords[0][:2]
    
    def rotate(self, angle=None, principal=None):
        if angle is None:
            if principal is None:
                raise ValueError("angle or principal must be provided")
            raise NotImplementedError("Principal axis rotation not implemented")
        return rotate(angle) 

    def translate(self, location):
        new = PolygonSection(
            exterior = self._exterior + location,
            interior = [i + location for i in self.interior()],
            mesh_size = self._mesh_size,
            mesher = self._mesher,
            z = self.z,
            name = self.name,
            material = self.material
        )

        # if self._p_model is not None:
        #     new._p_model = self._p_model.translate(location)

        # new._warp_analysis = None  # reset analysis on translation
        new.material = self.material
        new.z = self.z
        new.name = self.name
        new._mesher = self._mesher
        return new

    # @property
    # def elastic(self):
    #     # TODO
    #     return self.model.

    @property
    def model(self):
        if self._p_model is None:
            self._p_model = self._create_model(**self._model_kwds)

        return self._p_model

    def __repr__(self):
        return f"{self.__class__.__name__}{f'({self.name})' if self.name else ''}"
    # Internal
    def _create_model(self, **kwds):
        # NOTE: WideFlange overloads this to include shear warping;
        # Need to rethink how to do this generally

        mesh = self._create_mesh(**kwds)

        return TriangleModel.from_meshio(mesh)

    def _create_mesh(self, mesh_scale=None, engine=None):

        mesh_size = (mesh_scale or 1) * self._mesh_size


        if engine is None and self._mesher is not None:
            engine = self._mesher
        if engine is None:
            engine = "triangle"

        mesh = None
        if engine == "triangle":
            from ..composite import _mesh_cytri
            return _mesh_cytri([self], [mesh_size])

        patches = ShapelyPolygon(self.exterior(), self.interior())
        if engine == "gmsh":
            mesh = _mesh_gmsh(patches, mesh_size)

        elif engine == "dmsh":
            mesh = _mesh_dmsh(patches, mesh_size)

        elif engine == "meshpy":
            mesh = _mesh_meshpy(patches, mesh_size)

        return mesh


    def __contains__(self, p:tuple) -> bool:
        _inside = lambda p, v: (1 == sum(
                _ray_intersects(p, (v[i-1], v[i])) for i in range(len(v)))%2
        )

        for v in self.interior():
            if _inside(p, v):
                return False

        return _inside(p, self.exterior())


    @property
    def centroid(self):
        return centroid(self.exterior(), self.interior())

    #
    # Extra
    #

    @property
    def area(self):
        if self._area is None:
            self._area  = abs(_polygon_area(self.exterior()))
            self._area -= sum(abs(_polygon_area(hole)) for hole in self.interior())
        return self._area
    
    @property 
    def depth(self):
        y, z = np.asarray(self.vertices).T
        return max(z) - min(z)
    
    @property
    def width(self):
        y, z = np.asarray(self.vertices).T
        return max(y) - min(y)

    def moi(self, reference, area=None):
        x,y = (np.asarray(self.vertices) - np.asarray(reference)).T

        area = area or self.area

        alpha = x * np.roll(y, -1) - np.roll(x, -1) * y

        # planar moment of inertia wrt horizontal axis
        ixx = np.sum((y**2 + y * np.roll(y, -1) +
                      np.roll(y, -1)**2)*alpha)/12.0

        # planar moment of inertia wrt vertical axis
        iyy = np.sum((x**2 + x * np.roll(x, -1) +
                      np.roll(x, -1)**2)*alpha)/12.0

        # product of inertia
        ixy = np.sum((x*np.roll(y, -1)
                      + 2.0*x*y
                      + 2.0*np.roll(x, -1) * np.roll(y, -1)
                      + np.roll(x, -1) * y)*alpha)/24.

        return np.array([[ixx, ixy],
                         [ixy,iyy]])

    # @property
    def moic(self):
        if self._moic is None:
            self._moic = self.moi(self.centroid)
        return self._moic

    # @property
    def ixc(self):
        return self.moic[0,0]

    # @property
    def iyc(self):
        return self.moic[1,1]

    def cnn(self):
        A = self.area
        if self.material is not None:
            GA = self.material["G"]*A
            EA = self.material["E"]*A
        else:
            GA = A
            EA = A
        return np.array([[EA,  0,  0],
                         [ 0, GA,  0],
                         [ 0,  0, GA]])

    def cnm(self):
        if self.material is not None:
            G = self.material["G"]
            E = self.material["E"]
        else:
            G = 1
            E = 1
        Qz, Qy = self.centroid * self.area
        return np.array([[    0, E*Qy, -E*Qz],
                         [-G*Qy,    0,     0],
                         [ G*Qz,    0,     0]])
    


def rotate(ring, angle):
    """
    Rotate a polygon ring by a given angle in radians.
    """
    x,y = np.asarray(ring).T
    c,s = np.cos(angle), np.sin(angle)
    return np.column_stack((x*c - y*s, x*s + y*c))


def _sort_exterior(polygon, warping=None):
    """
    Remove duplicate points and collinear points from a polygon.
    """

    # points = np.unique(polygon, axis=0)
    # # Order points by angle from centroid
    # polygon = points[np.argsort( np.arctan2(points[:,1] - points[:,1].mean(), 
    #                                         points[:,0] - points[:,0].mean()))]

    if _polygon_area(polygon) < 0:
        return polygon[::-1]
    return polygon

def _polygon_area(exterior):
    x,y = np.asarray(exterior).T
    return np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)*0.5


def centroid(exterior, interior=None):
    if interior is None:
        interior = []

    def ring_sums(coords):
        """Return (sum_cross, sum_x, sum_y) for one closed ring."""
        x, y = np.asarray(coords, dtype=float).T
        x_n = np.roll(x, -1)
        y_n = np.roll(y, -1)
        cross = x * y_n - x_n * y
        return cross.sum(), ((x + x_n) * cross).sum(), ((y + y_n) * cross).sum()

    # Outer boundary (any orientation OK)
    S_total, Sx_total, Sy_total = ring_sums(exterior)

    # Holes: ensure negative area contribution
    for hole in interior:
        hole = np.asarray(hole, dtype=float)
        S, Sx, Sy = ring_sums(hole)
        if 0.5 * S > 0:  # hole ring is CCW; flip to make it negative
            S, Sx, Sy = ring_sums(hole[::-1])
        S_total  += S
        Sx_total += Sx
        Sy_total += Sy

    A = 0.5 * S_total
    x_cent = Sx_total / (6.0 * A)
    y_cent = Sy_total / (6.0 * A)
    return np.array([x_cent, y_cent])

    # if interior is None:
    #     interior = []

    # x,y = np.asarray(exterior).T

    # area = _polygon_area(exterior)
    # x_cent = (np.sum((x + np.roll(x, -1)) *
    #                     (x*np.roll(y, -1) -
    #                     np.roll(x, -1)*y)))/(6.0*area)
    # y_cent = (np.sum((y + np.roll(y, -1)) *
    #                     (x*np.roll(y, -1) -
    #                     np.roll(x, -1)*y)))/(6.0*area)

    # c = np.array((x_cent, y_cent))*area

    # area_total = area
    # for hole in interior:
    #     hole_area = -_polygon_area(hole)
    #     c += centroid(hole)*hole_area
    #     area_total += hole_area

    # return c/area_total


def _inertia_tensor(exterior, interior=None):
    if interior is None:
        interior = []

    x,y = np.asarray(exterior).T

    alpha = x * np.roll(y, -1) - np.roll(x, -1) * y

    # planar moment of inertia wrt horizontal axis
    ixx = np.sum((y**2 + y * np.roll(y, -1) +
                    np.roll(y, -1)**2)*alpha)/12.0

    # planar moment of inertia wrt vertical axis
    iyy = np.sum((x**2 + x * np.roll(x, -1) +
                    np.roll(x, -1)**2)*alpha)/12.0

    # product of inertia
    ixy = np.sum((x*np.roll(y, -1)
                    + 2.0*x*y
                    + 2.0*np.roll(x, -1) * np.roll(y, -1)
                    + np.roll(x, -1) * y)*alpha)/24.
    
    I =  np.array([[ixx, ixy],
                   [ixy, iyy]])

    for ring in interior:
        assert False

    return I


def principal_angle(exterior, interior=None)->float:
    from shps.rotor import log
    import numpy as np
    import numpy.linalg as la

    I = _inertia_tensor(exterior, interior)
    vals, vecs = la.eig(I)
    Q = np.eye(3)
    Q[1:,1:] = vecs
    theta = log(Q)
    assert np.isclose(theta[1], 0.0), theta
    assert np.isclose(theta[2], 0.0), theta
    return float(theta[0])

def _ray_intersects(p, edge)->bool:
    """
    Take a point p and an edge of two endpoints a,b of a line segment 
    and return bool

    https://rosettacode.org/wiki/Ray-casting_algorithm#Python
    """
    _eps = 0.00001
    _huge = sys.float_info.max
    _tiny = sys.float_info.min

    a,b = edge
    if a[1] > b[1]:
        a,b = b,a
    if p[1] == a[1] or p[1] == b[1]:
        p = np.array((p[0], p[1] + _eps))

    intersect = False

    if (p[1] > b[1] or p[1] < a[1]) or (p[0] > max(a[0], b[0])):
        return False

    if p[0] < min(a[0], b[0]):
        return True

    else:
        if abs(a[0] - b[0]) > _tiny:
            m_red = (b[1] - a[1]) / float(b[0] - a[0])
        else:
            m_red = _huge
        if abs(a[0] - p[0]) > _tiny:
            m_blue = (p[1] - a[1]) / float(p[0] - a[0])
        else:
            m_blue = _huge
        intersect = m_blue >= m_red
    return intersect



def _poly_facets(coords, offset=0):
    """Return (point_list, facet_index_pairs) for a polygon ring."""
    pts   = [tuple(p[:2]) for p in coords[:-1]]   # drop repeated last pt
    n     = len(pts)
    idx   = list(range(offset, offset + n))
    edges = [(idx[i], idx[(i + 1) % n]) for i in range(n)]
    return pts, edges



def _mesh_pymesh(polygon, size, min_angle=20.0, **kwargs):
    """
    Generate a 2D triangular mesh from a Shapely Polygon (or similar object) using PyMesh.
    
    Parameters:
        polygon   (Polygon or convertible): Input polygon geometry (Shapely Polygon or list of coords).
        size      (float): Characteristic length (approximate target triangle edge length).
        min_angle (float): Minimum angle in degrees for triangle quality (default 20°).
        **kwargs: Additional PyMesh triangle meshing options (e.g., max_area, max_steiner_points).
    
    Returns:
        meshio.Mesh: Mesh object containing triangular mesh (points and cells).
    
    Backend Options Considered:
    ---------------------------
    PyMesh.triangle (Shewchuk) – Uses J.R. Shewchuk's Triangle; robust and high-quality refinement.
    PyMesh (CGAL)             – Uses CGAL's 2D meshing algorithms; reliable but potentially slower.
    External (triangle/PyMesh) – Other libraries (e.g., triangle module) not used here for integration.

    """
    import pymesh
    import meshio
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    # Orient polygon for consistent winding (outer CCW, holes CW for auto hole detection)
    polygon = orient(polygon, sign=1.0)
    
    # Extract outer boundary coordinates and remove duplicate closing point if present
    outer_coords = list(polygon.exterior.coords)
    if len(outer_coords) > 1 and np.allclose(outer_coords[0], outer_coords[-1]):
        outer_coords = outer_coords[:-1]
    
    # Prepare point list and segment list for Planar Straight Line Graph (PSLG)
    points = []
    segments = []
    # Add outer loop points and segments
    start_idx = 0
    points.extend(outer_coords)
    n = len(outer_coords)
    segments.extend([(start_idx + i, start_idx + (i + 1) % n) for i in range(n)])
    
    # Add interior (hole) loops points and segments, track a representative point for each hole
    hole_points = []  # will hold one interior point inside each hole
    for interior in polygon.interiors:
        coords = list(interior.coords)
        if len(coords) == 0:
            continue  # skip empty interior
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        start_idx = len(points)
        points.extend(coords)
        m = len(coords)
        segments.extend([(start_idx + j, start_idx + (j + 1) % m) for j in range(m)])
        # Calculate a point inside the hole (using the centroid or representative point)
        hole_poly = Polygon(coords)
        if hole_poly.area > 0:
            interior_pt = hole_poly.representative_point()
            hole_points.append((interior_pt.x, interior_pt.y))
    
    # Configure PyMesh triangulation
    tri = pymesh.triangle()               # initialize Triangle wrapper
    tri.points = np.array(points, dtype=float)
    tri.segments = np.array(segments, dtype=int)
    tri.min_angle = float(min_angle)      # enforce minimum angle for mesh quality
    if size is not None:
        tri.max_area = 0.5 * (float(size) ** 2)  # set max triangle area based on characteristic size
    
    # Handle holes: use explicit hole points if available, else enable automatic hole detection
    if hole_points:
        tri.holes = np.array(hole_points, dtype=float)
    else:
        tri.auto_hole_detection = True
    
    # Apply any additional meshing parameters passed via kwargs
    for key, val in kwargs.items():
        if hasattr(tri, key):
            setattr(tri, key, val)
    
    tri.verbosity = 0   # quiet output for meshing
    tri.run()           # execute triangulation
    
    # Convert PyMesh output to meshio.Mesh
    mesh = tri.mesh                       # PyMesh Mesh object (with vertices and faces)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    # If vertices are 3D (z=0), reduce to 2D for meshio
    if vertices.shape[1] > 2:
        vertices = vertices[:, :2]
    # Create a meshio Mesh with triangular cells
    meshio_mesh = meshio.Mesh(points=vertices, cells=[("triangle", faces)])
    return meshio_mesh


def _mesh_meshpy(sect, size, min_angle=25.0, **kwds):
    """
    Mesh a 2-D section with MeshPy/Triangle.

    Parameters
    ----------
    sect : Any
        Your section description; must be convertible by `_prep_poly` to a
        Shapely Polygon (possibly with holes).
    size : float | (float, …)
        Characteristic length.  MeshPy uses `max_area`, so we set
        max_area ≈ 0.5 * size^2.
    min_angle : float, optional
        Quality constraint passed to Triangle (default 25°).
    **kwds
        Additional keyword args forwarded to `tri.build()`.

    Returns
    -------
    meshio.Mesh
    """
    from meshpy import triangle as tri
    import meshio


    shape = sect

    #
    points, facets, holes = [], [], []

    # exterior ring
    exterior_pts, exterior_facets = _poly_facets(np.array(shape.exterior.coords))
    points.extend(exterior_pts)
    facets.extend(exterior_facets)

    # interior rings (holes)
    for ring in shape.interiors:
        ring_pts, ring_facets = _poly_facets(np.array(ring.coords), offset=len(points))
        points.extend(ring_pts)
        facets.extend(ring_facets)

        # Triangle needs *one* point inside each hole
        holes.append(ring.representative_point().coords[0][:2])

    # -- Build the mesh ------------------------------------------------------
    mesh_info = tri.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    if holes:
        mesh_info.set_holes(holes)

    # Characteristic length -> target triangle area
    if isinstance(size, (list, tuple, np.ndarray)):
        size = float(size[0])
    max_area = 0.5 * size ** 2

    tri_mesh = tri.build(
        mesh_info,
        max_volume=max_area,
        min_angle=min_angle,
        **kwds,          # e.g. allow_boundary_steiner=False, …
    )

    # -- Convert to meshio
    pts = np.column_stack((np.asarray(tri_mesh.points), np.zeros(len(tri_mesh.points))))
    cells = [("triangle", np.asarray(tri_mesh.elements, dtype=int))]

    return meshio.Mesh(pts, cells)


def _mesh_dmsh(sect, size, **kwds):
    import dmsh 
    if isinstance(size, (list, tuple)):
        size = size[0]

    shape = sect
    
    geo = dmsh.Polygon(shape.exterior.coords[:-1])

    for hole in shape.interiors:
        geo  = geo - dmsh.Polygon(hole.coords[:-1])

    msh = dmsh.generate(geo, size, **kwds)
    import meshio
    return meshio.Mesh(*msh)

def _mesh_gmsh(sect, size, **kwds):
    import pygmsh
    if isinstance(size, (int, float)):
        size = [size]*2

    shape = sect

    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_min = size[0]
        geom.characteristic_length_max = size[1]
        coords = np.array(shape.exterior.coords)
        holes = [
            geom.add_polygon(np.array(h.coords)[:-1], size[0], make_surface=False).curve_loop
            for h in shape.interiors
        ]
        if len(holes) == 0:
            holes = None

        poly = geom.add_polygon(coords[:-1], size[1], holes=holes)
        # geom.set_recombined_surfaces([poly.surface])
        mesh = geom.generate_mesh(**kwds)

    mesh.points = mesh.points[:,:2]
    for blk in mesh.cells:
        blk.data = blk.data.astype(int)

    # for cell in mesh.cells:
    #     cell.data = np.roll(np.flip(cell.data, axis=1),3,1)
    return mesh


class rect(PolygonSection):
    # _args = [
    #    Ref("material", type=Material, field="material",
    #        about="tag of previously defined material (`UniaxialMaterial`"\
    #              "tag for a `FiberSection` or `NDMaterial` tag for use "\
    #              "in an `NDFiberSection`)"),
    #    Grp("divs", reverse=True, args=[
    #        Int("ij", about="number of subdivisions (fibers) in the IJ direction."),
    #        Int("jk", about="number of subdivisions (fibers) in the JK direction."),
    #    ]),
    #    Grp("corners", args=[
    #      Grp(args=[Num("yI"), Num("zI")],  reverse=True, about="$y$ & $z$-coordinates of vertex I (local coordinate system)"),
    #      #Grp(args=[Num("yJ"), Num("zJ")],  reverse=True, about="$y$ & $z$-coordinates of vertex J (local coordinate system)"),
    #      Grp(args=[Num("yK"), Num("zK")],  reverse=True, about="$y$ & $z$-coordinates of vertex K (local coordinate system)"),
    #      #Grp(args=[Num("yL"), Num("zL")],  reverse=True, about="$y$ & $z$-coordinates of vertex L (local coordinate system)"),
    #   ])
    # ]
    def init(self):
        self._moic = None
        self._moig = None
        self._area = None
        self._rule = "mid"
        self._interp = None
        self._fibers = None

        if len(self.corners) == 2:
            ll, ur = self.corners
            self.vertices = [ll, [ur[0], ll[1]], ur, [ll[0], ur[1]]]

    @property
    def fibers(self):
        if self._fibers is None:
            if self.divs is None:
                return []
            from shps.gauss import iquad
            # TODO: Use shps for lq4
            interp = self._interp or Q4
            rule = self._rule

            loc, wght = zip(iquad(rule=rule, n=self.divs[0]), iquad(rule=rule, n=self.divs[1]))

            x,y= zip(*(
                    interp(r,s)@self.vertices
                            for r,s in itertools.product(*loc)
            ))

            da = 0.25*np.fromiter(
                (dx*dy*self.area  for dx,dy in itertools.product(*wght)),
                float, len(x)
            )
            self._fibers = [
                Fiber([xi,yi], dai, self.material) for yi,xi,dai in sorted(zip(y,x,da))
            ]
        return self._fibers


class quad(PolygonSection):
    """A quadrilateral shaped patch.
    The geometry of the patch is defined by four vertices: I J K L.
    The coordinates of each of the four vertices is specified in *counter clockwise* sequence
    """
    # _img  = "quadPatch.svg"
    # _args = [
    #    Ref("material", type=Material, field="material",
    #        about="tag of previously defined material (`UniaxialMaterial` "\
    #              "tag for a `FiberSection` or `NDMaterial` tag for use in an `NDFiberSection`)"),
    #    Grp("divs", reverse=True, args=[
    #      Int("ij", about="number of subdivisions (fibers) in the IJ direction."),
    #      Int("jk", about="number of subdivisions (fibers) in the JK direction."),
    #    ]),
    #    Grp("vertices", args=[
    #      Grp("i", args=[Num("x"), Num("y")],  about="$x$ & $y$-coordinates of vertex I (local coordinate system)", reverse=True),
    #      Grp("j", args=[Num("x"), Num("y")],  about="$x$ & $y$-coordinates of vertex J (local coordinate system)", reverse=True),
    #      Grp("k", args=[Num("x"), Num("y")],  about="$x$ & $y$-coordinates of vertex K (local coordinate system)", reverse=True),
    #      Grp("l", args=[Num("x"), Num("y")],  about="$x$ & $y$-coordinates of vertex L (local coordinate system)", reverse=True),
    #   ])
    # ]

    def init(self):
        self._moic = None
        self._moig = None
        self._area = None
        self._interp = None
        self._rule = "mid"
        self._fibers = None

    @property
    def fibers(self):
        if self._fibers is None:
            from shps.gauss import iquad
            interp = self._interp or Q4
            rule = self._rule

            loc, wght = zip(iquad(rule=rule, n=self.divs[0]),
                            iquad(rule=rule, n=self.divs[1]))

            x,y= zip(*(
                    interp(r,s)@self.vertices
                            for r,s in itertools.product(*loc)
            ))

            da = 0.25*np.fromiter(
                (dx*dy*self.area  for dx,dy in itertools.product(*wght)), 
                float, len(x)
            )
            #y, x, da = map(list, zip(*sorted(zip(y, x, da))))
            self._fibers = [Fiber([xi,yi], dai, self.material) for yi,xi,dai in sorted(zip(y,x,da))]
        return self._fibers

def rhom(center, height, width, slope=None, divs=(0,0)):
    vertices = [
        [center[0] - width/2 + slope*height/2, center[1] + height/2],
        [center[0] - width/2 - slope*height/2, center[1] - height/2],
        [center[0] + width/2 - slope*height/2, center[1] - height/2],
        [center[0] + width/2 + slope*height/2, center[1] + height/2]
    ]
    return quad(vertices=vertices, div=divs)
