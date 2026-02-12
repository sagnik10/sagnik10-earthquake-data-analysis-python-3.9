"""
Dependencies:
- `shapely` is used for shape-building operations (merging, 
  diffing polygons, add holes, etc.)
- `pygmsh` is used for meshing
"""
from math import pi, sin, cos, sqrt
import numpy as np

from opensees import section, patch, layer
import opensees.render.mpl as render
import elle.units

inch = ksi_psi = fce = ec0 = esp = Ec = Gc = Esh = fu = esh = esu = 1.0

#
# Geometry Building
#
def GirderSection(units="english_engineering"):
    # 1. Create the section object
    #------------------------------------------------------
    #                                ^y
    #    |  4'0"   |       12'0"     |             (sym)         |
    # _  |_______________________________________________________|
    #    |_____  _______________ _________ _______________  _____|
    #          \ \             | |       | |             / /
    #5'6"       \ \            | |   |   | |            / /
    #            \ \___________| |_______| |___________/ /
    # _           \__________________+__________________/  ---> x
    #
    #             |                                     |

    units = elle.units.UnitHandler("english_engineering")
    ft, inch, spacing = units.ft, units.inch, units.spacing

    # 2. Dimensions
    #------------------------------------------------------
    web_slope      = 0.5
    thickness_top  = (7 + 1/2) * inch
    thickness_bot  = (5 + 1/2) * inch
    height         = 5*ft + 8*inch
    width_top      = 2*26 * ft
    width_webs     = ([12]*5) * inch
    web_centers    = 4 @ spacing(7*ft + 9*inch, "centered")

    # center-to-center height
    c2c_height = height - thickness_top/2 - thickness_bot/2
    inside_height = height - thickness_bot - thickness_top

    # width of bottom flange
    width_bot = width_top -  \
            2*(2.5*ft + web_slope*(inside_height + thickness_bot))

    # 3. Build section
    #------------------------------------------------------
    girder_section = section.FiberSection(areas=[
        # add rectangle patch for top flange
        patch.rect(vertices=[
            [-width_top/2, c2c_height - thickness_top/2],
            [+width_top/2, c2c_height + thickness_top/2]]),

        # add rectangle patch for bottom flange
        patch.rect(vertices=[
            [-width_bot/2,  -thickness_bot/2],
            [+width_bot/2,  +thickness_bot/2]]),

        # sloped outer webs
        patch.rhom(
            height = inside_height,
            width  = 1.0*ft,
            slope  = -1/web_slope,
            center = [
                -width_bot/2-inside_height/2*web_slope+0.5*ft,
                 (thickness_bot + inside_height)/2
            ]
        ),
        patch.rhom(
            height = inside_height,
            width  = 1.0*ft,
            slope  = 1/web_slope,
            center = [
                 width_bot/2+inside_height/2*web_slope-0.5*ft,
                 (thickness_bot + inside_height)/2
            ]
        ),
    ] + [
    # vertical inner webs[
        patch.rect(vertices=[
            [loc - width/2,        0.0 + thickness_bot/2],
            [loc + width/2, c2c_height - thickness_top/2]]
        )
        for width, loc in zip(width_webs, web_centers)
    ])

    return girder_section

def RegularPolygon(n, Rcol):
    phi =  2*pi/n
    R = Rcol/cos(phi/2)
    vertices = [
        [R*cos(i*phi-phi/2),  R*sin(i*phi-phi/2)]
        for i in range(n)
    ]
    return patch._Polygon(vertices)

#
# Meshing
#

# PyMesh
# PyGMSH

#
# IO
#
def sees2shapely(section):
    """
    Generate `sectionproperties` geometry objects 
    from `anabel` patches.
    """
    # import sectionproperties.pre.geometry as SP_Sections
    import shapely.geometry
    from shapely.ops import unary_union
    shapes = []
    # meshes = []
    if hasattr(section, "patches"):
        patches = section.patches
    else:
        patches = [section]
    for patch in patches:
        name = patch.__class__.__name__.lower()
        if name in ["quad", "poly", "rect", "_polygon"]:
            points = np.array(patch.vertices)
            width,_ = points[1] - points[0]
            _,height = points[2] - points[0]
            shapes.append(shapely.geometry.Polygon(points))
        else:
            n = 64
            x_off, y_off = 0.0, 0.0
            # calculate location of the point
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

def sect2mesh(section):
    import meshio
    return meshio.Mesh(
        section.mesh_nodes,
        cells={"triangle6": section.mesh_elements}
)

def sees2gmsh(sect, size, **kwds):
    import pygmsh
    if isinstance(size, int): size = [size]*2
    shape = sees2shapely(sect)
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

def sees2sectprop(girder_section):
    import sectionproperties.pre.geometry as SP_Sections
    from sectionproperties.analysis.section import Section as CrossSection
    shapes = sees2shapely(girder_section)
    try:
        geom = SP_Sections.Geometry(shapes)
    except ValueError:
        geom = SP_Sections.CompoundGeometry(shapes)

    # geom.plot_geometry();
    return geom
        
#
# Analysis
#
def SP_Model(sect, mesh):
    from sectionproperties.analysis.section import Section
    geom = sees2sectprop(sect)
    geom.create_mesh(mesh)
    return Section(geom, time_info=False)


# ipyvtklink pyvista 
# pythreejs
# pacman -S xorg-server-xvfb

def plot(mesh, values=None, scale=1.0, show_edges=None, savefig:str=None,**kwds):
    from matplotlib import cm
    import pyvista as pv
    pv.set_jupyter_backend("panel")


    pv.start_xvfb(wait=0.05)
    mesh = pv.utilities.from_meshio(mesh)
    if values is not None:
        point_values = scale*values
        mesh.point_data["u"] = point_values
        mesh = mesh.warp_by_scalar("u", factor=scale)
        mesh.set_active_scalars("u")
    if show_edges is None:
        show_edges = True #if sum(len(c.data) for c in mesh.cells) < 1000 else False
    if not pv.OFF_SCREEN:
        plotter = pv.Plotter(notebook=True)
        plotter.add_mesh(mesh,
           show_edges=show_edges,
           cmap=cm.get_cmap("RdYlBu_r"),
           lighting=False,
           **kwds)
        # if len(values) < 1000:
        #     plotter.add_mesh(
        #        pv.PolyData(mesh.points), color='red',
        #        point_size=5, render_points_as_spheres=True)
        if savefig:
            plotter.show(screenshot=savefig)
        else:
            plotter.show()



#
# Loci
#

def moment(r, E, T, xi, wi, thickness=1):
    import jax
    import jax.numpy as jnp
    det = jnp.linalg.det
    slv = jnp.linalg.solve
    def f(x):
        # dx = x[1] - r
        dx = r - x[1]
        return jnp.array([
            [ 1,    dx],
            [dx, dx**2]
        ])
    dT = jax.jacfwd(T)
    jacn_transf = lambda x,X: dT(x).T@X
    
    def dvol(xi, xyz): 
        # return 0.5*thickness*(abs(det(jacn_transf(xi,xyz))))
        return thickness*(abs(det(jacn_transf(xi,xyz))))
    
    def Q(X):
        return sum(f(T(x)@X)*w*dvol(x,X) for x,w in zip(xi,wi))
    
    return Q

