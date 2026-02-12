
import matplotlib.pyplot as plt
import matplotlib.tri as tri



# plots a finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements, ax):
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        ax.fill(x, y, edgecolor='black', ls="-", lw=0.5, fill=False)


def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))

    ax.autoscale()


class Rendering:
    def __init__(self, ax = None):
        import matplotlib.pyplot as plt
        if ax is None:
            _,self.ax = plt.subplots()
        else:
            self.ax = ax

    def draw_nodes(self, nodes):
        self.ax.scatter(*zip(*nodes.values()))
        for k,v in nodes.items():
            self.ax.annotate(k, v)
        return self

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

def scatter(solution, nodes, elems):
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(*zip(*nodes.values()), solution)
    return ax

# converts quad elements into tri elements
def _quads_to_tris(quads):
    tris = [
        [None for j in range(3)] for i in range(2*len(quads))
    ]
    for i in range(len(quads)):
        j = 2*i
        tris[j][0]     = quads[i][0]
        tris[j][1]     = quads[i][1]
        tris[j][2]     = quads[i][2]
        tris[j + 1][0] = quads[i][2]
        tris[j + 1][1] = quads[i][3]
        tris[j + 1][2] = quads[i][0]
    return tris


def plot(solution, nodes, elems, ax=None):
    elements_tris = []
    nodes_x, nodes_y = zip(*nodes.values())
    elements_quads = [tuple(i-1 for i in elem) for elem in elems.values()]

    elements = elements_tris + elements_quads


    # plots a finite element mesh
    def plot_fem_mesh(nodes_x, nodes_y, conn, ax):
        for element in conn:
            x = [nodes_x[element[i]] for i in range(len(element))]
            y = [nodes_y[element[i]] for i in range(len(element))]
            # plot filled polygons.
            ax.fill(x, y, edgecolor='black', ls="-", lw=0.5, fill=False)

    # convert all elements into triangles
    elements_all_tris = elements_tris + _quads_to_tris(elements_quads)

    # create an unstructured triangular grid instance
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)

    if ax is None:
        _, ax = plt.subplots()

    # plot the finite element mesh
    plot_fem_mesh(nodes_x, nodes_y, elements, ax=ax)

    # plot the contours
    contours = \
        ax.tricontourf(triangulation, solution, cmap="twilight", alpha=0.5)

    plt.colorbar(contours, ax=ax)
    ax.axis('equal')
    return ax

