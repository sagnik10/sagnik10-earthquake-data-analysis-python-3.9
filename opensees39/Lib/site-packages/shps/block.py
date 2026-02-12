from itertools import product, count
import numpy as np
from shps import plane, child

from .types import Shape, Nodes, Tuple, Child, Block
Tag = int

class Cell:
    def nodes(self)->list:
        pass



def block(ne: Tuple[int,int],
          family: Shape,
          nstart: Tag     = 1,
          estart: Tag     = 1,
          points: Nodes   = None,
          parent:  Shape  = None,
          stencil: Child  = None,
          append : bool   = True,
          exclude: set    = None,
          join   : Block  = None,
          radius : float      = 1e-8,
          number  = "feap"
          ): # -> nodes, elems
    """
    Use a regular subdivision of a parent master element to create
    a mesh within its perimeter.
    """

    child_family = family
    nn = family.n, family.n

    exclude = set() if exclude is None else exclude

    #
    # 1. Create edge grid in natural coordinates (-1, 1)
    #
    xl, cells = grid(ne, nn)


    #
    # 2. Create internal nodes for element type
    #
    taggen = filter(lambda i: i > len(xl) or i not in xl, count(1))

    for i in cells:
        elem = cells[i]
        conn = [
            elem[0],
            elem[nn[0]-1],
            elem[nn[0]+nn[1]-2],
            elem[-nn[0]+1]
        ]

        conn += [i for i in elem if i not in conn]

        ref = child.IsoparametricMap(plane.Q4, nodes={
            1: xl[elem[0]],
            2: xl[elem[nn[0]-1]],
            3: xl[elem[nn[0]+nn[1]-2]],
            4: xl[elem[-nn[0]+1]]
        })

        for loc, xn in child_family.inner.items():
            tag = next(taggen)
            xl[tag] = ref.coord(xn)
            conn.insert(loc-1, tag)
        cells[i] = conn

    if points is not None and stencil is None:
        block_family = parent or plane.Q9
        stencil = child.IsoparametricMap(block_family, nodes=points)

    if stencil is None:
        return xl, cells

    #
    # 3. Map grid into problem coordinates and merge
    #

    # (xl, cells), join, nstart, estart, append, stencil, radius, 
    import scipy.spatial

    join_nodes = join["nodes"] if join is not None else {} # points

    if join is not None:
        tree = scipy.spatial.KDTree(np.array([x for x in join_nodes.values()]))

    tags = np.array(list(join_nodes.keys()))

    taggen = filter(lambda i: i not in tags, count(nstart))

    rename = {}
    if append:
        nodes = join_nodes
    else:
        nodes = {}

    for loc_tag, loc_coord in xl.items():
        tag   = None
        coord = stencil.coord(loc_coord)

        if join is not None:
            neighbors = tree.query_ball_point(coord, radius)
            if neighbors:
                tag = tags[neighbors[0]]

        if tag is None:
            tag = next(taggen)
            nodes[tag] = coord

        rename[loc_tag] = tag

    # Rename all references in `cells`
    join_cells = join["cells"] if join is not None and join["cells"] is not None else {}
    if append:
        new_cells = join_cells

    else:
        new_cells = {}

    elemgen = filter(lambda i: i not in join_cells, count(estart))

    for k,conn in cells.items():
        new_cells[next(elemgen)] = tuple(rename[n] for n in conn)

    return nodes, new_cells

create_block = block

def join(first, second, nstart=1, estart=1, radius=1e-8, append: bool = True):
    """
    March 2024
    """
    # (xl, cells), join, nstart, estart, append, stencil, radius, 
    import scipy.spatial

    join_nodes, join_cells = first

    tree = scipy.spatial.KDTree(np.array([x for x in join_nodes.values()]))

    tags = np.array(list(join_nodes.keys()))

    taggen = filter(lambda i: i not in tags, count(nstart))

    rename = {}
    if append:
        nodes = join_nodes
    else:
        nodes = {}

    for loc_tag, loc_coord in second[0].items():
        tag   = None
        coord = loc_coord

        neighbors = tree.query_ball_point(coord, radius)

        if neighbors:
            tag = tags[neighbors[0]]

        if tag is None:
            tag = next(taggen)
            nodes[tag] = coord

        rename[loc_tag] = tag

    # Rename all references in `cells`
#   join_cells = first["cells"]
    if append:
        new_cells = join_cells

    else:
        new_cells = {}

    elemgen = filter(lambda i: i not in join_cells, count(estart))

    for k,conn in second[1].items():
        new_cells[next(elemgen)] = tuple(rename[c] for c in conn)

    return nodes, new_cells


def grid(ne: Tuple[int,int], nn=(2,2), ndm=2, corners=((-1,1),(-1,1))):
    """
               |           ne[0] = 2       |

               |             | nn[0] = 3   |

          ---- +------+------+------+------+ ----
               |             |             |
               |             |             |
               |             |             | nn[1] = 2
               |             |             |
               |             |             |
               +------+------+------+------+ ----
               |             |             |
  ne[1] = 3    |             |             |
               |             |             |---------> s
               |             |             |
               |             |             |
               +------+------+------+------+
               |             |             |
               |             |             |
               |             |             |
               |             |             |
               |             |             |
          ---- +------+------+------+------+

    """
    nnx,nny = nn
    nex,ney = ne

    nx = nnx - 1
    ny = nny - 1

    cells = {
        k: [*(   (1+l) + i*nx + (j*ny+    0)*(nex*nx+1) for l in range(nnx)),
            *(    nnx  + i*nx + (j*ny+    l)*(nex*nx+1) for l in range(1,nny-1)),
            *(   (1+l) + i*nx + (j*ny+nny-1)*(nex*nx+1) for l in reversed(range(nnx))),
            *( -nex*nx + i*nx + (j*ny+nny-l)*(nex*nx+1) for l in range(1,nny-1))]

        for k,(j,i) in enumerate(product(range(ney), range(nex)))
    }

    used = {i for cell in cells.values() for i in cell}

    x, y = map(lambda n: np.linspace(*n[2], n[0]*(n[1]-1)+1), ((nex,nnx,corners[0]),(ney,nny,corners[1])))

    nodes = {
        i+1: (xi, yi) for i,(yi,xi) in enumerate(product(y,x)) if i+1 in used
    }

    return nodes, cells



if __name__ == "__main__":
    def plot(nodes, cells):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for cell in cells.values():
            ax.plot(*zip(*[nodes[i] for i in cell], nodes[cell[0]]))
        return ax

    import sys,pprint


    ne = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) > 3:
        nn = int(sys.argv[3]), int(sys.argv[4])

    else:
        nn = 2,2

    # nodes, cells = grid(ne, nn)


# First block
    element = plane.Lagrange(4)
    points  = {
            1: (0.0, 0.0),
            2: (1.1, 0.0),
            3: (1.0, 1.0),
            4: (0.0, 1.0),
            5: (0.5,-0.1),
            6: (1.1, 0.5)
    }

    nodes, cells = block(ne, element, points=points)

# Second Block
    element = plane.Serendipity(4)

    points  = {
            1: (1.1, 0.0),
            2: (2.0, 0.0),
            3: (2.0, 1.0),
            4: (1.0, 1.0),
            5: (1.5,-0.1),
#           7: (2.1, 0.5),
            8: (1.1, 0.5)
    }

    other = dict(nodes=nodes, cells=cells)
    nodes, cells = block(ne, element, points=points, join=other)


    from .plotting import Rendering
    ax = plot(nodes, cells)
    ax.axis("equal")

    ax = None
    Rendering(ax=ax).draw_nodes(nodes).show()

    first = grid((3,2), (2,3))

    nodes, cells = join(first, grid((3,2), corners=((1, 2),(-1,1))))
    ax = plot(nodes, cells)
    ax.axis("equal")

#   ax = None
    # Plotter(ax=ax).nodes(nodes).show()
