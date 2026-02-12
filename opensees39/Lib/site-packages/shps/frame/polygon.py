import sys
import numpy as np 
from shps.frame import WarpingSection
from shapely.geometry import Polygon as ShapelyPolygon

class Polygon:
    def __init__(self,
                 exterior, interior=None,
                 centroid=None,
                 mesh_size=None, elastic=None, mesher=None):
        self._exterior  = np.asarray(exterior)
        self._interior  = interior 
        self._centroid  = centroid
        self._p_elastic = elastic
        self._mesher    = mesher

        self._moic = None
        self._moig = None
        self._area = None
        self._fibers = None
        self._mesh_size = mesh_size
        if mesh_size is None:
            raise ValueError("mesh_size must be provided")

    def exterior(self, plane=False):
        return self._exterior

    def interior(self, plane=False):
        if self._interior is None:
            return []
        return self._interior

    def elastic(self):
        return self._p_elastic

    @property 
    def model(self):
        return self._create_model()._analysis

    def _create_model(self, **kwds):
        # NOTE: WideFlange overloads this to include shear warping;
        # Need to rethink how to do this generally

        mesh = self._create_mesh(**kwds)

        return WarpingSection.from_meshio(mesh, warp_shear=False)


    def _create_mesh(self, mesh_scale=None, engine=None):
        # TODO: Add support for CyTriangle which looks nice.

        mesh_size = (mesh_scale or 1) * self._mesh_size

        
        patches = ShapelyPolygon(self.exterior(), self.interior())
        if engine is None and self._mesher is not None:
            engine = self._mesher
        if engine is None:
            engine = "meshpy"

        mesh = None

        if engine == "gmsh":
            mesh = _mesh_gmsh(patches, mesh_size)

        elif engine == "dmsh":
            mesh = _mesh_dmsh(patches, mesh_size)

        elif engine == "meshpy":
            mesh = _mesh_meshpy(patches, mesh_size)

        return mesh

    def __contains__(self, p:tuple) -> bool:
        v = self.vertices
        inside = (1 == sum(
            _rayIntersectSeg(p, (v[i-1], v[i])) for i in range(len(v)))%2
        )
        return inside

    # @property
    def area(self):
        if self._area is None:
            x,y = np.asarray(self.vertices).T
            self._area = np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)*0.5
        return self._area


    def centroid(self):
        x,y = np.asarray(self.vertices).T
        area = self.area
        x_cent = (np.sum((x + np.roll(x, -1)) *
                         (x*np.roll(y, -1) -
                          np.roll(x, -1)*y)))/(6.0*area)
        y_cent = (np.sum((y + np.roll(y, -1)) *
                         (x*np.roll(y, -1) -
                          np.roll(x, -1)*y)))/(6.0*area)
        return np.array((x_cent, y_cent))

    def moi(self, reference, vertices=None, area=None):
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

        return np.array([[ixx, ixy],[ixy,iyy]])

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
    


def _rayIntersectSeg(p, edge)->bool:
    """
    take a point p and an edge of two endpoints a,b of a line segment 
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
    pts   = [tuple(p[:2]) for p in coords[:-1]]          # drop repeated last pt
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
    
    This function prefers PyMesh's Triangle backend for efficiency, quality, and hole support.
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
