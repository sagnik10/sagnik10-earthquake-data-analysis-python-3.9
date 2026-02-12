#
# Claudio Perez
#
from dataclasses import dataclass
from collections import defaultdict

from .torsion import TorsionAnalysis
import numpy as np
from shps.solve.poisson import _map_triangle

class FlexureAnalysis:
    pass

@dataclass
class _Element:
    nodes: tuple # of int
  # gauss: tuple # of Gauss
    shape: str
    model: dict = None
    group: int = 0


def _yield_const(value):
    while True:
        yield value


class PlaneMesh:
    def __init__(self, nodes, elems, assigns=None):#, offset=None):
        self.nodes = nodes 
        self.elems = elems
        if assigns is None:
            assigns = {i: {"e": 1, "g": 1} for i,elem in enumerate(elems)}
        self._assigns = assigns
        # self.offset = offset

    def update(self, values):
        pass 

    def cells(self):
        return [
            elem.nodes for elem in self.elems
        ]

    def translate(self, offset):
        return type(self)(self.nodes - np.array(offset), self.elems, assigns=self._assigns)#, self.offset)

    def rotate(self, angle):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        return type(self)((R@self.nodes.T).T, self.elems)#, self.offset)


class TriangleModel(PlaneMesh):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

        for i,elem in enumerate(self.elems):
            assert len(elem.nodes) == 3
            if self.cell_area(i) < 0:
                elem.nodes = elem.nodes[::-1]


    def save(self, filename):
        """
        Save the mesh to a file in a format that can be read by meshio.
        """
        import meshio
        cells = [("triangle", np.array([elem.nodes for elem in self.elems]))]
        mesh = meshio.Mesh(points=self.nodes, cells=cells)
        mesh.write(filename)

    @classmethod 
    def from_xara(self, data):
        """
        Create a TriangleModel from a Xara plane finite element model
        """
        data = data["StructuralAnalysisModel"]
        mesh_nodes = {
            int(node["name"]): (i,node["crd"]) for i,node in enumerate(data["geometry"]["nodes"])
        }
        materials = {
            int(mat["name"]): {"e" : mat["E"], "g": 1} for mat in data["properties"]["nDMaterials"]
        }
        used_nodes = set()
        for elem in data["geometry"]["elements"]:
            for node in elem["nodes"]:
                used_nodes.add(int(node))
        nodes = {
            k: (i, mesh_nodes[k][1]) for i, k in enumerate(used_nodes)
        }

        elems = []
        for elem in data["geometry"]["elements"]:
            if elem["type"].lower() == "tri31":
                nodes_indices = [nodes[int(node)][0] for node in elem["nodes"]]
                elems.append(_Element(nodes=np.array(nodes_indices), shape="T3", group=int(elem["material"])))
            elif elem["type"] == "SixNodeTri":
                nodes_indices = [nodes[int(node)][0] for node in elem["nodes"]]
                elems.append(_Element(nodes=(nodes_indices[0], nodes_indices[1], nodes_indices[3]), shape="T3", group=elem["material"]))
                elems.append(_Element(nodes=(nodes_indices[1], nodes_indices[2], nodes_indices[4]), shape="T3", group=elem["material"]))
                elems.append(_Element(nodes=(nodes_indices[2], nodes_indices[0], nodes_indices[5]), shape="T3", group=elem["material"]))
                elems.append(_Element(nodes=(nodes_indices[3], nodes_indices[4], nodes_indices[5]), shape="T3", group=elem["material"]))
            elif "quad" in elem["type"].lower():
                material = int(elem["materials"][0])
                nodes_indices = [nodes[int(node)][0] for node in elem["nodes"]]
                elems.append(_Element(nodes=np.array([nodes_indices[0], nodes_indices[1], nodes_indices[2]]), shape="T3", group=material))
                elems.append(_Element(nodes=np.array([nodes_indices[0], nodes_indices[2], nodes_indices[3]]), shape="T3", group=material))
            else:
                raise ValueError(f"Unsupported element type {elem['type']} in Xara data.")
        nodes = np.array([node[1] for node in nodes.values()])

        return TriangleModel(nodes, elems, assigns=materials)


    @classmethod
    def from_meshio(cls, mesh, **kwds):
        # meshio object, assume all tri3s
        nodes = mesh.points
        cells = None
        if nodes.shape[1] == 3:
            nodes = nodes[:, :2]

        elems = []

        for i,cells in enumerate(mesh.cells):
            if "region" in mesh.cell_data:
                regions = mesh.cell_data["region"][i]
            else:
                regions = _yield_const(None)

            #
            if cells.type == "triangle":

                elems.extend([
                    _Element(nodes=cell, shape="T3", group=group)
                    for cell,group in zip(cells.data, regions)
                ])

            elif cells.type == "quad":

                for cell, group in zip(cells.data, regions):
                    elems.append(_Element(nodes=(cell[0], cell[1], cell[2]), shape="T3", group=group))
                    elems.append(_Element(nodes=(cell[0], cell[2], cell[3]), shape="T3", group=group))
            
            elif cells.type == "tri6":
                elems.extend([
                    _Element(nodes=(cell[0], cell[1], cell[3]), shape="T3", group=group),
                    _Element(nodes=(cell[1], cell[2], cell[4]), shape="T3", group=group),
                    _Element(nodes=(cell[2], cell[0], cell[5]), shape="T3", group=group),
                    _Element(nodes=(cell[3], cell[4], cell[5]), shape="T3", group=group)
                ] for cell, group in zip(cells.data, regions))

        return TriangleModel(nodes, elems)


    def create_handle(self, state):
        """
        Returns a function handle that approximates the solution at any (x,y)
        by performing piecewise linear interpolation over the mesh.

        Assumes that self.state is a numpy array of nodal values.
        """

        def u(point):
            point = np.array(point)
            # Loop over all elements to locate the point.
            for elem in self.elems:
                indices = elem.nodes  # indices into self.nodes and state
                # Get the triangle's vertices
                A, B, C = self.nodes[indices]
                # Compute barycentric coordinates for point P with respect to triangle ABC.
                v0 = B - A
                v1 = C - A
                v2 = point - A

                d00 = np.dot(v0, v0)
                d01 = np.dot(v0, v1)
                d11 = np.dot(v1, v1)
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)

                denom = d00 * d11 - d01 * d01
                if denom == 0:
                    # Degenerate triangle; skip it.
                    continue
                a = (d11 * d20 - d01 * d21) / denom
                b = (d00 * d21 - d01 * d20) / denom
                c = 1 - a - b

                # Check if the point lies inside the triangle (with a tolerance).
                tol = 1e-12
                if a >= -tol and b >= -tol and c >= -tol:
                    # Interpolate the solution using barycentric coordinates.
                    return a * state[indices[0]] + b * state[indices[1]] + c * state[indices[2]]

            # If the point is not found in any triangle, raise an error.
            x, y = point
            raise ValueError(f"Point ({x}, {y}) is not inside any triangle in the mesh.")

        return u

    def exterior(self):
        """
        Returns a list of (x, y) nodes that form the exterior boundary.
        """
        cycles = _boundary_cycles(self.cells())
        # Convert cycles of node indices to cycles of (x, y) coordinates.
        cycles_pts = [[self.nodes[i] for i in cyc] for cyc in cycles if len(cyc) >= 3]
        if not cycles_pts:
            return []

        # The cycle with the largest absolute area is assumed to be the exterior.
        areas = [abs(_signed_area(pts)) for pts in cycles_pts]
        exterior_cycle = cycles_pts[areas.index(max(areas))]
        # # exterior_cycle.append(exterior_cycle[0])
        from veux.utility.alpha_shape import remove_collinear_points
        return remove_collinear_points(np.array(exterior_cycle))
        # return np.array(exterior_cycle) #[:-1,:]

    def interior(self):
        """
        Returns a list of lists, each inner list is a list of (x, y) nodes
        defining a hole in the mesh.
        """
        cycles = _boundary_cycles(self.cells())
        cycles_pts = [[self.nodes[i] for i in cyc] for cyc in cycles if len(cyc) >= 3]
        if not cycles_pts:
            return []

        areas = [abs(_signed_area(pts)) for pts in cycles_pts]
        # Assume the cycle with the maximum area is the exterior.
        ext_index = areas.index(max(areas))
        holes = [pts for i, pts in enumerate(cycles_pts) if i != ext_index]
        return holes


    def cell_area(self, tag=None)->float:
        if tag is None:
            return sum(self.cell_area(i) for i in range(len(self.elems)))

        y, z = self.nodes[self.elems[tag].nodes].T
        z1, z2, z3 = z
        y1, y2, y3 = y
        a = float(0.5 * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)))
        assert a >= 0, f"Negative area {a} for element {tag} with nodes {self.elems[tag].nodes}"
        return a


    def cell_solution(self, tag, state):
        return float(sum(state[self.elems[tag].nodes]))/3


    def cell_gradient(self, tag, field):
        # May also try Cook equation 3.4-6
        u1, u2, u3 = field[self.elems[tag].nodes]
        ((y1, y2, y3), (z1, z2, z3)) = self.nodes[self.elems[tag].nodes].T
        z12 = z1 - z2
        z23 = z2 - z3
        z31 = z3 - z1
        y32 = y3 - y2
        y13 = y1 - y3
        y21 = y2 - y1
        A = self.cell_area(tag)
        return 1/(2*A)*np.array([
            z23*u1 + z31*u2 + z12*u3,
            y32*u1 + y13*u2 + y21*u3
        ])

    def energy(self, u, v, weight=None)->float:
        if weight is None:
            weight = "e"

        q = 0.0
        for elem in self.elems:
            by,bz,area = _map_triangle(self.nodes[elem.nodes].T)
            K_by = np.outer(by, by)
            K_bz = np.outer(bz, bz)
            
            Ke = (1.0 / (4.0 * area)) * (K_by + K_bz)

            w = self._assigns[elem.group][weight]

            ve = v[elem.nodes]
            q += u[elem.nodes].dot(Ke@ve)*w

        return q

    def _assemble_K1_K2(self, v, w, weight="e"):
        """
        u = [v, w] on a T3 mesh.
        Returns:
        K1 = ∫ (e_i ⊗ ∂_i u)(∂_k u ⊗ e_k) dΩ  (2x2 over spatial axes y,z)
        K2 = ∫ (∂_i u ⊗ e_i)(e_k ⊗ ∂_k u) dΩ  (2x2 over components v,w)
        """
        K1 = np.zeros((2, 2), dtype=float)  # rows/cols: (y,z)
        K2 = np.zeros((2, 2), dtype=float)  # rows/cols: (v,w)

        for elem in self.elems:
            by, bz, area = _map_triangle(self.nodes[elem.nodes].T)  # len-3 each
            inv4A = 1.0 / (4.0 * area)

            # Element derivative Gram blocks
            Kyy = inv4A * np.outer(by, by)  # ∫ ∂y(·) ∂y(·)
            Kzz = inv4A * np.outer(bz, bz)  # ∫ ∂z(·) ∂z(·)
            Kyz = inv4A * np.outer(by, bz)  # ∫ ∂y(·) ∂z(·)
            Kzy = inv4A * np.outer(bz, by)  # ∫ ∂z(·) ∂y(·)

            wt = float(self._assigns[elem.group][weight])

            ve = v[elem.nodes].astype(float, copy=False)
            we = w[elem.nodes].astype(float, copy=False)

            # ---- K1 = ∫ (∂_i u · ∂_k u) e_i ⊗ e_k  (spatial 2x2)
            # yy, zz
            K1[0, 0] += (ve @ Kyy @ ve + we @ Kyy @ we) * wt
            K1[1, 1] += (ve @ Kzz @ ve + we @ Kzz @ we) * wt
            # yz, zy
            K1[0, 1] += (ve @ Kyz @ ve + we @ Kyz @ we) * wt
            K1[1, 0] += (ve @ Kzy @ ve + we @ Kzy @ we) * wt

            # ---- K2 = ∫ (∇u)(∇u)^T  (component 2x2)
            KyKz = Kyy + Kzz
            vv = (ve @ KyKz @ ve) * wt    # ∫ |∇v|^2
            ww = (we @ KyKz @ we) * wt    # ∫ |∇w|^2
            vw = (ve @ KyKz @ we) * wt    # ∫ ∇v·∇w

            K2[0, 0] += vv
            K2[1, 1] += ww
            K2[0, 1] += vw
            K2[1, 0] += vw

        # Numerical clean-up: enforce symmetry
        K1 = 0.5 * (K1 + K1.T)
        K2 = 0.5 * (K2 + K2.T)
        return K1, K2

    def integrate_I2(self, U, weight="weight"):
        """
        Compute I2 = ∫ (∇iu o ei)(ei o ∇iu) dΩ for a 2D vector field U (n_nodes x 2).
        Returns a 2x2 matrix:
        [ ∫ ∂y u · ∂y u   ∫ ∂y u · ∂z u
            ∫ ∂z u · ∂y u   ∫ ∂z u · ∂z u ]
        """
        I = np.zeros((2, 2), dtype=float)
        for elem in self.elems:
            by, bz, area = _map_triangle(self.nodes[elem.nodes].T)  # length-3 each
            inv4A = 1.0 / (4.0 * area)

            # Element matrices for the four (j,k) derivative pairs
            Kyy = inv4A * np.outer(by, by)
            Kzz = inv4A * np.outer(bz, bz)
            Kyz = inv4A * np.outer(by, bz)
            Kzy = inv4A * np.outer(bz, by)

            w = self._assigns[elem.group][weight]
            Ue = U[elem.nodes]  # (3 x 2), columns = vector components m

            # Sum over vector components m
            # I[j,k] += sum_m u_m^T K_{jk} u_m
            for m in range(Ue.shape[1]):
                ue = Ue[:, m]
                I[0,0] += (ue @ Kyy @ ue) * w   # (j,k) = (y,y)
                I[1,1] += (ue @ Kzz @ ue) * w   # (z,z)
                I[0,1] += (ue @ Kyz @ ue) * w   # (y,z)
                I[1,0] += (ue @ Kzy @ ue) * w   # (z,y)
        return I

    def inertia(self, va, ua, weight="e")->float:
        r"""
        v.dot( ([\int N.T rho @  N dA] @u)
        """
        q = 0.0
        for i,elem in enumerate(self.elems):
            v1, v2, v3 = va[elem.nodes]
            u1, u2, u3 = ua[elem.nodes]
            w = self.cell_area(i)*self._assigns[elem.group][weight]
            # v[nodes].dot(int(N.T@N)@u[nodes])
            q += w/12.0*(u1*(2*v1 + v2 + v3) + u2*(v1 + 2*v2 + v3) + u3*(v1 + v2 + 2*v3))

        return float(q)

    def poisson(self, u, v):
        """
        Integrate ∫ (dNdy u) + (dNdz v) dΩ over a T3 mesh.
        """
        q = 0
        for elem in self.elems:
            by, bz, area = _map_triangle(self.nodes[elem.nodes].T)
            q += (by@u[elem.nodes] + bz@v[elem.nodes])/2.0 # *area
        return q

    def burgers(self, u,v,w, i):
        """
        Integrate ∫ (N u)(N v)(dNdx[i] w) dΩ over a T3 mesh.

        convection/advection
        """
        q = 0

        M0 = np.array([[2,1,1],
                       [1,2,1],
                       [1,1,2]],dtype=float)/12

        for elem in self.elems:
            by, bz, area = _map_triangle(self.nodes[elem.nodes].T)
            B = [by,bz][i]
            q += (u[elem.nodes] @ M0 @ v[elem.nodes]) * B@w[elem.nodes]/2
        return q

    def cubic(self, u,v,w)->float:
        """
        Integrate ∫ (N u)(N v)(N w) dΩ over a T3 mesh.
        - u, v, w: 1D arrays of nodal values (len = n_nodes)
        - elems: iterable of elements; each item is either a length-3 index array
                or an object with attribute `.nodes` (length-3 indices)
        - nodes: (n_nodes, 2) array with columns [y, z]
        """
        q = 0.0
        for elem in self.elems:
            idx = elem.nodes

            by, bz, A = _map_triangle(self.nodes[elem.nodes].T)

            ue = u[idx].astype(float, copy=False)
            ve = v[idx].astype(float, copy=False)
            we = w[idx].astype(float, copy=False)

            su = ue.sum(); sv = ve.sum(); sw = we.sum()
            uv = ue * ve
            uw = ue * we
            vw = ve * we

            S0 = float(uv @ we)                 # Σ u_i v_i w_i        (i=j=k)
            S1 = float(uv.sum() * sw - S0)      # Σ_{i=j≠k} u_i v_i w_k
            S2 = float(uw.sum() * sv - S0)      # Σ_{i=k≠j} u_i v_j w_i
            S3 = float(vw.sum() * su - S0)      # Σ_{j=k≠i} u_i v_j w_j
            T  = float(su * sv * sw)            # Σ_{i,j,k} u_i v_j w_k

            # ∫ N_i N_j N_k dΩ = A * {1/10 if all same, 1/30 if two same, 1/60 if all distinct}
            q += (A / 60.0) * (5.0 * S0 + S1 + S2 + S3 + T)

        return q

    def quartic(self, u, v, w, x):
        """
        ∫ (N u)(N v)(N w)(N x) dΩ over a T3 mesh.
        - u, v, w, x: 1D arrays of nodal values (len = n_nodes)
        - elems: iterable of 3-node elements (either index triplets or objects with .nodes)
        - nodes: (n_nodes, 2) array with columns [y, z]
        """
        q = 0.0
        for elem in self.elems:
            idx = elem.nodes

            # area
            _, _, A = _map_triangle(self.nodes[idx].T)

            ue = u[idx].astype(float, copy=False)
            ve = v[idx].astype(float, copy=False)
            we = w[idx].astype(float, copy=False)
            xe = x[idx].astype(float, copy=False)

            # singles
            su, sv, sw, sx = ue.sum(), ve.sum(), we.sum(), xe.sum()

            # pairs
            P_uv = float((ue*ve).sum())
            P_uw = float((ue*we).sum())
            P_ux = float((ue*xe).sum())
            P_vw = float((ve*we).sum())
            P_vx = float((ve*xe).sum())
            P_wx = float((we*xe).sum())

            # triples
            T_uvw = float((ue*ve*we).sum())
            T_uvx = float((ue*ve*xe).sum())
            T_uwx = float((ue*we*xe).sum())
            T_vwx = float((ve*we*xe).sum())

            # quadruple (all same node)
            Q = float((ue*ve*we*xe).sum())

            # contributions by multiplicity of node indices in N_i N_j N_k N_l
            C1111 = Q / 15.0  # i=j=k=l

            # 3+1 (three the same, one different) — 4 singleton choices
            C31 = (
                (T_vwx*su - Q) +  # singleton u
                (T_uwx*sv - Q) +  # singleton v
                (T_uvx*sw - Q) +  # singleton w
                (T_uvw*sx - Q)    # singleton x
            ) / 60.0

            # 2+2 (two pairs) — 3 pairings
            C22 = (
                (P_uv*P_wx - Q) +
                (P_uw*P_vx - Q) +
                (P_ux*P_vw - Q)
            ) / 90.0

            # 2+1+1 (one pair, two distinct singles) — 6 pair choices
            C211 = (
                # pair uv, singles w,x
                (P_uv*(sw*sx - P_wx) - (sx*T_uvw + sw*T_uvx) + 2*Q) +
                # pair uw, singles v,x
                (P_uw*(sv*sx - P_vx) - (sx*T_uvw + sv*T_uwx) + 2*Q) +
                # pair ux, singles v,w
                (P_ux*(sv*sw - P_vw) - (sw*T_uvx + sv*T_uwx) + 2*Q) +
                # pair vw, singles u,x
                (P_vw*(su*sx - P_ux) - (sx*T_vwx + su*T_uvw) + 2*Q) +
                # pair vx, singles u,w
                (P_vx*(su*sw - P_uw) - (sw*T_uvx + su*T_vwx) + 2*Q) +
                # pair wx, singles u,v
                (P_wx*(su*sv - P_uv) - (sv*T_uwx + su*T_vwx) + 2*Q)
            ) / 180.0

            q += A * (C1111 + C31 + C22 + C211)

        return q


    def curl(self, u, v):

        q = 0.0
        for elem in self.elems:
            ((y1, y2, y3), (z1, z2, z3)) = self.nodes[elem.nodes].T
            z12 = z1 - z2
            z23 = z2 - z3
            z31 = z3 - z1
            y32 = y3 - y2
            y13 = y1 - y3
            y21 = y2 - y1

            ((y1, y2, y3), (z1, z2, z3)) = u[elem.nodes].T

            f = 1/6.*np.array([
                ((y1*y32 - z1*z23) + (y2*y32 - z2*z23) + (y3*y32 - z3*z23)),
                ((y1*y13 - z1*z31) + (y2*y13 - z2*z31) + (y3*y13 - z3*z31)),
                ((y1*y21 - z1*z12) + (y2*y21 - z2*z12) + (y3*y21 - z3*z12))])

            q += f.dot(v[elem.nodes])

        return q


    def cell_inertia(self, va, ua, tag=None):
        pass

def _boundary_cycles(elems):
    """
    Given a list of triangles (each as a 3-tuple of node indices),
    returns a list of cycles (each cycle is a list of node indices)
    corresponding to the boundary edges.
    """
    # Count how many triangles share each edge.
    edge_count = defaultdict(int)
    for tri in elems:
        for a, b in [(tri[0], tri[1]),
                     (tri[1], tri[2]),
                     (tri[2], tri[0])]:
            key = tuple(sorted((a, b)))
            edge_count[key] += 1

    # Boundary edges are those that appear only once.
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Build an undirected graph from boundary edges.
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Walk the graph to extract closed cycles.
    cycles = []
    visited = set()
    for start in list(adj.keys()):
        if start in visited:
            continue

        cycle = []
        current = start
        prev = None
        while True:
            cycle.append(current)
            visited.add(current)
            neighbors = adj[current]
            if not neighbors:
                break  # In a well-formed mesh, this shouldn't happen.
            # Pick the neighbor that is not the one we came from.
            if prev is None:
                nxt = neighbors[0]
            else:
                nxt = neighbors[0] if neighbors[0] != prev else (neighbors[1] if len(neighbors) > 1 else neighbors[0])
            if nxt == start:
                break
            prev, current = current, nxt
        cycles.append(cycle)
    return cycles

def _signed_area(pts):
    """
    Computes the signed area of a polygon defined by the list of points.
    """
    area = 0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += (x0 * y1 - y0 * x1)
    return area / 2


