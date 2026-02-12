#
# Claudio Perez
#
from ..warping import WarpingSection
from ..polygon import PolygonSection

class Material: pass
class Backbone: pass

class CompositeSection(WarpingSection):
    """
    A non-homogeneous cross-section composed from multiple sections.

    Parameters
    ----------
    patches: list[Section]
        A list of sections to combine into a composite section.
    material: Material, optional
        A default material to assign to patches without a material.
    """

    def __init__(self, patches, **kwds):
        self._patches = _clip_sections(patches)
        self._fibers = None
        self._mesh_size = [patch.mesh_size for patch in patches]
        self._mesher = kwds.get("mesher", "gmsh")
        self._c_model = None
        self._exterior = None
        self._area   = None
        if "material" in kwds:
            mat = kwds["material"]
            for i in self._patches:
                if i.material is None:
                    i.material = mat

    def add_patch(self, patch):
        self._fibers = None
        self._exterior = None
        self._patches.append(patch)

    def add_patches(self, patch):
        self._fibers = None
        self._exterior = None
        self._patches.extend(patch)

    @property 
    def centroid(self):
        pass

    @property 
    def depth(self):
        if self._patches:
            return max(p.z for p in self._patches) - min(p.z for p in self._patches)
        return 0.0
    @property
    def width(self):
        if self._patches:
            return max(p.x for p in self._patches) - min(p.x for p in self._patches)
        return 0.0

    def exterior(self):

        if self._exterior is not None:
            return self._exterior

        import shapely.geometry
        from shapely.ops import unary_union

        shapes = []

        for patch in self._patches:
            shapes.append(shapely.geometry.Polygon(patch.exterior(),
                                                   patch.interior()))
            
        if len(shapes) > 1:
            self._exterior = unary_union(shapes).exterior.coords
        else:
            self._exterior = shapes[0].exterior.coords
        
        return self._exterior
        
    def interior(self):
        holes = []
        for patch in self._patches:
            if patch.interior() is not None:
                holes.extend(patch.interior())
        return holes


    @property 
    def model(self):
        if self._c_model is None:
            self._c_model = self._create_mesh(self._mesh_size, engine=self._mesher) 
        return self._c_model


    def _create_mesh(self, mesh_size: list=None, engine=None):

        mesh = _mesh_cytri(self._patches, mesh_size=mesh_size, min_angle=25.0)

        from shps.frame.solvers import TriangleModel
        return TriangleModel.from_meshio(mesh)

    @property
    def patches(self):
        return [p for p in self._patches]

    @property
    def area(self):
        if self._area is None:
            self._area = sum(i.area for i in self._patches)
        return self._area


    @property
    def ixc(self):
        # TODO: cache
        yc = self.centroid[1]
        return sum(
            p.ixc + (p.centroid[1] - yc)**2*p.area for p in self._patches
        )

    @property
    def iyc(self):
        # TODO: cache
        xc = self.centroid[0]
        return sum(
            p.iyc + (p.centroid[0]-xc)**2*p.area for p in self._patches
        )

    @property
    def moic(self):
        # TODO: cache
        return [
            [p.moi[i] + p.centroid[i]**2*p.area for i in range(2)] + [p.moi[-1]]
            for p in self._patches
        ]

    def __contains__(self, point):
        return any(point in area for area in self._shapes)


def _clip_sections(sections):
    """
    Return each section clipped by all higher-z sections individually.
    Preserves the original number, order, and attributes.
    """
    from shapely.geometry import Polygon
    import numpy as np

    # Sort with index to restore original order later
    indexed = sorted([(i, getattr(s, "z", 0), s) for i, s in enumerate(sections)],
                     key=lambda t: -t[1])  # High z first

    result = [None for i in range(len(sections))]

    for i, (idx_i, z_i, sec_i) in enumerate(indexed):
        poly_i = Polygon(sec_i.exterior(), sec_i.interior())

        # Subtract each higher-priority section
        for j in range(i):
            _, z_j, sec_j = indexed[j]
            poly_j = Polygon(sec_j.exterior(), sec_j.interior())
            poly_i = poly_i.difference(poly_j)

        # Build section

        ext = np.array(poly_i.exterior.coords)
        ints = [np.array(ring.coords) for ring in poly_i.interiors]

        result[idx_i] = PolygonSection(ext, ints,
                                       mesh_size=sec_i.mesh_size,
                                       material=sec_i.material,
                                       name=sec_i.name,
                                       z=z_i)

    return result



def _mesh_cytri(sections, mesh_size=0.05, min_angle=25.0, coarse=False, quadratic=False):
    """
    Generate a triangular mesh with material interfaces preserved using Shewchuk's Triangle.
    """
    import cytriangle as triangle
    import numpy as np
    import shapely

    vertex_map = {}
    points         = []
    facets         = []
    holes          = []
    control_points = []
    mesh_sizes     = []
    z_indices      = []

    def get_vertex_index(pt):
        key = tuple(np.round(pt, 12))  # prevent numerical duplicates
        if key in vertex_map:
            return vertex_map[key]
        idx = len(points)
        points.append(key)
        vertex_map[key] = idx
        return idx
    
    polygons = [
        shapely.Polygon(section.exterior(), section.interior()) 
        for section in sections
    ]

    for i,(section,polygon) in enumerate(zip(sections, polygons)):
        ext       = section.exterior()
        interiors = section.interior()
        material  = section.material

        z_indices.append(section.z)

        # Exterior
        ext_idx = [get_vertex_index(p) for p in ext]
        facets.extend([(ext_idx[i], ext_idx[(i + 1) % len(ext_idx)]) 
                        for i in range(len(ext_idx))])

        # Holes
        for hole,hole_poly in zip(interiors, polygon.interiors):
            rh = shapely.Polygon(hole_poly).representative_point()

            if any(shapely.contains(p,rh) for p in polygons if p != polygon):
                continue

            hole_idx = [get_vertex_index(p) for p in hole]
            facets.extend([
                (hole_idx[i], hole_idx[(i + 1) % len(hole_idx)]) for i in range(len(hole_idx))
            ])
            holes.append(tuple(rh.coords[0]))


        # Control point for region interior
        control_points.append(polygon.representative_point().coords[0]) #tuple(np.mean(ext, axis=0)))

        # Region mesh size
        if isinstance(mesh_size, dict):
            mesh_sizes.append(mesh_size.get(material, 0.05))

        elif isinstance(mesh_size, list):
            if len(mesh_size) != len(sections):
                raise ValueError("mesh_size list must match number of sections")
            mesh_sizes.append(mesh_size[i])

        else:
            mesh_sizes.append(mesh_size)

    # Prepare Triangle input
    tri = {
        "vertices": points,
        "segments": facets,
        "holes": holes,
        "regions": [
            [cp[0], cp[1], z, mesh_sizes[i]]
            for i, (cp,z) in enumerate(zip(control_points, z_indices))
        ]
    }

    for i, r in enumerate(tri["regions"]):
        if len(r) != 4 or not all(isinstance(v, (float, int)) for v in r):
            raise ValueError(f"Bad region entry at index {i}: {r}")

    opts = "pA" if coarse else f"pq{min_angle:.1f}Aa"

    if quadratic:
        # Quadratic (6-node) triangles
        opts += "o2"

    data = triangle.triangulate(tri, opts)


    points = np.array(data["vertices"],
                      dtype=np.double)

    # Quadratic triangles are 6-node triangles (3 vertices + 3 mid-side nodes)
    # Meshio uses "triangle6" for this
    element = "triangle6" if quadratic else "triangle"
    cells = [(element, np.array(data["triangles"], dtype=np.int32))]

    # Optional cell data (e.g., region markers)
    cell_data = {}
    if "triangle_attributes" in data:
        triangle_attr = np.array(data["triangle_attributes"], dtype=int).flatten()
        cell_data = {"region": [triangle_attr]}

    import meshio
    return meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=cell_data
    )



def _create_mesh_cyt(
    points: list[tuple[float, float]],
    facets: list[tuple[int, int]],
    holes: list[tuple[float, float]],
    control_points: list[tuple[float, float]],
    mesh_sizes: list[float] | float,
    min_angle: float,
    coarse: bool,
) -> dict[str, list[list[float]] | list[list[int]]]:
    """Discretize a 2D region by triangles.
    """
    import cytriangle as triangle
    if not isinstance(mesh_sizes, list):
        mesh_sizes = [mesh_sizes]

    tri = {}                  # create tri dictionary
    tri["vertices"] = points  # set point
    tri["segments"] = facets  # set facets

    if holes:
        tri["holes"] = holes  # set holes

    # prepare regions
    regions = []

    for i, cp in enumerate(control_points):
        rg = [cp[0], cp[1], i, mesh_sizes[i]]
        regions.append(rg)

    tri["regions"] = regions  # set regions

    # generate mesh
    if coarse:
        mesh = triangle.triangulate(tri, "pAo2")
    else:
        mesh = triangle.triangulate(tri, f"pq{min_angle:.1f}Aao2")

    return mesh
