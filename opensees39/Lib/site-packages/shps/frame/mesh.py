import numpy as np


def _poly_facets(coords, offset=0):
    """Return (point_list, facet_index_pairs) for a polygon ring."""
    pts   = [tuple(p[:2]) for p in coords[:-1]]          # drop repeated last pt
    n     = len(pts)
    idx   = list(range(offset, offset + n))
    edges = [(idx[i], idx[(i + 1) % n]) for i in range(n)]
    return pts, edges


def sect2shapely(section):
    """
    Generate `shapely` geometry objects
    from `opensees` patches or a FiberSection.
    """
    import shapely.geometry
    from shapely.ops import unary_union
    shapes = []

    if hasattr(section, "patches"):
        patches = section.patches

    elif isinstance(section, list):
        patches = section

    else:
        patches = [section]


    for patch in patches:
        name = patch.__class__.__name__.lower()

        if name in ["quad", "poly", "rect", "_polygon"]:
            points = np.array(patch.vertices)
            width,_  = points[1] - points[0]
            _,height = points[2] - points[0]
            shapes.append(shapely.geometry.Polygon(points))

        else:
            # Assuming its a circle
            n = 64 # points on circumference
            x_off, y_off = 0.0, 0.0
            external = [[
                0.5 * patch.extRad * np.cos(i*2*np.pi*1./n - np.pi/8) + x_off,
                0.5 * patch.extRad * np.sin(i*2*np.pi*1./n - np.pi/8) + y_off
                ] for i in range(n)
            ]
            if patch.intRad > 0.0:
                internal = [[
                    0.5 * patch.intRad * np.cos(i*2*np.pi*1./n - np.pi/8) + x_off,
                    0.5 * patch.intRad * np.sin(i*2*np.pi*1./n - np.pi/8) + y_off
                    ] for i in range(n)
                ]
                shapes.append(shapely.geometry.Polygon(external, [internal]))

            else:
                shapes.append(shapely.geometry.Polygon(external))

    if len(shapes) > 1:
        return unary_union(shapes)
    else:
        return shapes[0]



def sect2pymesh(polygon, size, min_angle=20.0, **kwargs):
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

    # Ensure we have a Shapely Polygon (convert if necessary)
    if not isinstance(polygon, Polygon):
        polygon = Polygon(polygon)
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


def sect2meshpy(sect, size, min_angle=25.0, **kwds):
    """
    Mesh a 2-D section with MeshPy/Triangle.

    Parameters
    ----------
    sect : Any
        Your section description; must be convertible by `sect2shapely` to a
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
    # -- Convert the section to a Shapely polygon
    from shapely.geometry import Polygon
    from meshpy import triangle as tri
    import meshio

    if isinstance(sect, tuple):
        shape = Polygon(*sect)

    elif not isinstance(sect, Polygon):
        shape: Polygon = sect2shapely(sect)

    else:
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

    # -- Convert to meshio ----------------------------------------------------
    pts = np.column_stack((np.asarray(tri_mesh.points), np.zeros(len(tri_mesh.points))))
    cells = [("triangle", np.asarray(tri_mesh.elements, dtype=int))]

    return meshio.Mesh(pts, cells)


def sect2dmsh(sect, size, **kwds):
    import dmsh 
    if isinstance(size, (list, tuple)):
        size = size[0]

    shape = sect2shapely(sect)
    
    geo = dmsh.Polygon(shape.exterior.coords[:-1])

    for hole in shape.interiors:
        geo  = geo - dmsh.Polygon(hole.coords[:-1])

    msh = dmsh.generate(geo, size, **kwds)
    import meshio
    return meshio.Mesh(*msh)

def sect2gmsh(sect, size, **kwds):
    import pygmsh
    if isinstance(size, (int, float)):
        size = [size]*2

    shape = sect2shapely(sect)

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
