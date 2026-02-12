import numpy as np
import warnings

from veux.utility.earcut import earcut
from scipy.spatial.transform import Rotation



class _SolidExtrusion:
    def __init__(self, model, direction=None, assigns=None, start_node=1, start_cell=1):
        if isinstance(model, tuple):
            nodes, cells = model
        else:
            nodes = model.nodes
            cells = model.cells()

        self._start_node = start_node
        self._next_cell  = start_cell
        self._section_nodes = nodes
        self._section_cells = cells
        self._assigns = assigns

        self._location = np.array([0.0, 0.0, 0.0])
        self._orientation = Rotation.identity()

        self._node_start =  [start_node, start_node + len(nodes)]
        self._initial = True

        if direction is not None:
            self._direction = np.array(direction, dtype=float)
            self._update_orientation(direction)
        else :
            self._direction = np.array([0.0, 0.0, 1.0])


    def _update_orientation(self, direction):
        # Convert and normalize the direction vector.
        direction = np.array(direction, dtype=float)
        self._direction = direction
        norm = np.linalg.norm(direction)
        new_dir = direction / norm

        # The reference normal is that of the initial section (z-axis).
        ref = np.array([0.0, 0.0, 1.0])

        # If the new direction is nearly the reference, no rotation is needed.
        if np.allclose(new_dir, ref):
            self._orientation = Rotation.identity()

        # If the new direction is opposite to the reference, a 180° rotation is needed.
        elif np.allclose(new_dir, -ref):
            self._orientation = Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
        else:
            # Compute the rotation axis as the cross product of ref and new_dir.
            v = np.cross(ref, new_dir)
            s = np.linalg.norm(v)
            c = np.dot(ref, new_dir)
            # Rodrigues' formula
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
            self._orientation = Rotation.from_matrix(R_matrix)


    def advance(self, direction=None):

        self._node_start[0] = self._node_start[1]
        self._node_start[1] = self._node_start[1] + len(self._section_nodes)
        if direction is not None:
            self._update_orientation(direction)

        self._location += self._direction

        self._next_cell += len(self._section_cells)*self._n_cell
        self._initial = False


    def nodes(self):

        for i in range(int(not self._initial), self._n_edge):
            for j,node in enumerate(self._section_nodes):
                yield (
                    self._node_start[i] + j, 
                    self._orientation.apply( self._location + i/(self._n_edge-1)*self._direction + [*node,0.0])
                )


class ExtrudeTetrahedron(_SolidExtrusion):
    def __init__(self, *args, **kwds):

        self._n_edge = 2
        self._n_cell = 3

        super().__init__(*args, **kwds)

    def cells(self):
        ncl = self._n_cell
        for i, cell in enumerate(self._section_cells):

            bottom_nodes = [self._node_start[0] + idx for idx in cell]
            top_nodes = [self._node_start[1] + idx for idx in cell]


            assign = (self._assigns[i], ) if self._assigns else ()

            yield (self._next_cell + i*ncl+0,
                   (*bottom_nodes, top_nodes[0]),
                   *assign
            )

            yield (self._next_cell + i*ncl+1,
                   (*bottom_nodes[1:], *top_nodes[:2]),
                   *assign
            )

            yield (self._next_cell + i*ncl+2, 
                   (*bottom_nodes[2:], *top_nodes[:3]),
                   *assign
            )


class ExtrudeHexahedron(_SolidExtrusion):
    def __init__(self, *args, **kwds):

        """
        for node in extr.nodes():
            model.node(...)

        model.advance([])

        for node in model.nodes():
            model.node(...)
        """
        nen = 4
        if nen == 4:
            self._n_edge = 2 # order + 1
        elif nen == 9:
            self._n_edge = 3

        self._n_cell = 1 # 1 solid cell per plane cell per layer

        super().__init__(*args, **kwds)


    def cells(self):
        for i, cell in enumerate(self._section_cells):

            bottom_nodes = [self._node_start[0] + idx for idx in cell]
            top_nodes = [self._node_start[1] + idx for idx in cell]

            conn = bottom_nodes + top_nodes
            assign = (self._assigns[i], ) if self._assigns else ()
            yield (self._next_cell + i, conn,  *assign)


class FrameMesh:
    """
    A utility class that, given a structural model with beam/frame elements,
    performs an extrusion of each element's cross-section rings and end caps.

    It accumulates:
      - `positions` (list of 3D points for each ring vertex)
      - `triangles` (list of triangular faces for side surfaces + end caps)
      - `ring_ranges` so we know which subset of vertices belong to each ring
      - **Optionally** a transform for each ring (element_name, j), if you want it.

    Typical usage:
        e = FrameMesh(model, scale=1.0)
        verts = e.vertices()          # Nx3 array of float
        tris  = e.triangles()         # Mx3 array of integer indices
        rings = e.ring_ranges()       # [((elem, j), start_idx, end_idx), ...]

#
#     x-------o---------o---------o
#   /       /         /
# x--------o<--------o---------o
# |        |       / ^
# |        |     /   |
# |        |   /     |
# |        | /       |
# x--------o-------->o---------o
#

    """
    def __init__(self, nen, sections, scale=1.0, do_end_caps=True):
        """
        :param model:  Dictionary-like structural model with "assembly",
                       "frame_section", "frame_orientation", etc.
        :param scale:  Uniform scale factor for cross-sectional outline
        :param do_end_caps: Whether to triangulate and add end caps
        :param earcut_fn: A function that triangulates a 2D outline (N,2)-> list of (i0,i1,i2)
        """
        self.nen = nen
        self.sections = sections
        self.scale = scale
        self.do_end_caps = do_end_caps

        self._positions = []
        self._triangles = []
        self._ring_ranges = []

        self._build_extrusion()

    def _build_extrusion(self):
        """
        Loops over each element in self.model["assembly"],
        accumulates side faces, end caps, and tracks ring ranges.
        """
        global_vertex_offset = 0

        outline_0 = self.sections[0]

        # Coordinates of the element’s nodes in reference config
        nen = self.nen
        noe = len(outline_0)  # number of points in cross-section outline

        if noe < 2 or nen < 1:
            raise ValueError("Invalid cross-section")


        # Build faces connecting rings
        for j in range(nen):
            ring_start = len(self._positions)

            outline_j = self.sections[j].copy()

            outline_j[:,1:] *= self.scale

            # Accumulate ring vertices
            for k, pt in enumerate(outline_j):
                self._positions.append(pt.astype(float))

                # Build side faces (connect ring j to ring j-1)
                if j > 0 and k < noe - 1:
                    self._triangles.append([
                        global_vertex_offset + noe*j + k,
                        global_vertex_offset + noe*j + (k+1),
                        global_vertex_offset + noe*(j-1) + k
                    ])
                    self._triangles.append([
                        global_vertex_offset + noe*j + (k+1),
                        global_vertex_offset + noe*(j-1) + (k+1),
                        global_vertex_offset + noe*(j-1) + k
                    ])
                elif j > 0 and k == (noe - 1):
                    # wrap-around for side faces
                    self._triangles.append([
                        global_vertex_offset + noe*j + k,
                        global_vertex_offset + noe*j,
                        global_vertex_offset + noe*(j-1) + k
                    ])
                    self._triangles.append([
                        global_vertex_offset + noe*j,
                        global_vertex_offset + noe*(j-1),
                        global_vertex_offset + noe*(j-1) + k
                    ])

            ring_end = len(self._positions)

            # Record ring range
            self._ring_ranges.append((j, ring_start, ring_end))



        if self.do_end_caps:
            try:
                # front cap = ring j=0
                front_outline = self.sections[0][:,1:].copy()
                front_outline *= self.scale
                front_tris = earcut(front_outline)
                j0_offset = global_vertex_offset

                for tri in front_tris:
                    self._triangles.append([
                        j0_offset + tri[0],
                        j0_offset + tri[1],
                        j0_offset + tri[2]
                    ])

                # back cap = ring j=nen-1
                back_outline = self.sections[nen-1][:,1:].copy()
                back_outline *= self.scale
                back_tris = earcut(back_outline)
                jN_offset = global_vertex_offset + noe*(nen-1)

                for tri in back_tris:
                    self._triangles.append([
                        jN_offset + tri[0],
                        jN_offset + tri[1],
                        jN_offset + tri[2]
                    ])
            except Exception as e:
                warnings.warn(f"Earcut failed: {e}")

        global_vertex_offset += nen * noe

    def vertices(self):
        """Return Nx3 float array of all vertex positions."""
        return np.array(self._positions,
                        dtype=float) #

    def triangles(self):
        """Return Mx3 int array of triangle faces."""
        return np.array(self._triangles, dtype=int)

    def ring_ranges(self):
        """
        Return a list of ((element_name, ring_j), start_idx, end_idx).
        The vertex indices in [start_idx:end_idx] belong to that ring.
        """
        return self._ring_ranges


class _FrameMesher:
    """

    """
    def __init__(self, model, scale=1.0, do_end_caps=True):
        """
        :param model:  Dictionary-like structural model with "assembly",
                       "frame_section", "frame_orientation", etc.
        :param scale:  Uniform scale factor for cross-sectional outline
        :param do_end_caps: Whether to triangulate and add end caps
        :param earcut_fn: A function that triangulates a 2D outline (N,2)-> list of (i0,i1,i2)
        """
        self.model = model
        self.scale = scale
        self.do_end_caps = do_end_caps

        self._positions = []
        self._triangles = []
        self._ring_ranges = []

        self._build_extrusion()

    def _build_extrusion(self):
        """
        Loops over each element in self.model["assembly"],
        accumulates side faces, end caps, and tracks ring ranges.
        """
        assembly = self.model["assembly"]
        global_vertex_offset = 0

        for element_name, el in assembly.items():
            outline_0 = self.model.frame_section(element_name, 0)
            if outline_0 is None:
                # No cross-section for this element
                continue

            # Coordinates of the element’s nodes in reference config
            X = np.array(el["crd"])   # shape: (nen, 3)
            nen = len(X)              # number of element nodes
            noe = len(outline_0)      # number of points in cross-section outline

            if noe < 2 or nen < 1:
                continue

            # We scale the "outline" in the local cross-section directions
            # (often that means outline_j[:,1:] *= scale, depending on your orientation convention)
            # But let's do it ring-by-ring in the loop below if needed.
            outline_scale = self.scale

            # Build side faces for each ring j
            for j in range(nen):
                ring_start = len(self._positions)

                outline_j = self.model.frame_section(element_name, j)

                outline_j[:,1:] *= outline_scale

                # Accumulate ring vertices
                for k, pt in enumerate(outline_j):
                    self._positions.append(pt.astype(float))

                    # Build side faces (connect ring j to ring j-1)
                    if j > 0 and k < noe - 1:
                        self._triangles.append([
                            global_vertex_offset + noe*j + k,
                            global_vertex_offset + noe*j + (k+1),
                            global_vertex_offset + noe*(j-1) + k
                        ])
                        self._triangles.append([
                            global_vertex_offset + noe*j + (k+1),
                            global_vertex_offset + noe*(j-1) + (k+1),
                            global_vertex_offset + noe*(j-1) + k
                        ])
                    elif j > 0 and k == (noe - 1):
                        # wrap-around for side faces
                        self._triangles.append([
                            global_vertex_offset + noe*j + k,
                            global_vertex_offset + noe*j,
                            global_vertex_offset + noe*(j-1) + k
                        ])
                        self._triangles.append([
                            global_vertex_offset + noe*j,
                            global_vertex_offset + noe*(j-1),
                            global_vertex_offset + noe*(j-1) + k
                        ])

                ring_end = len(self._positions)

                # Record ring range
                self._ring_ranges.append(((element_name, j), ring_start, ring_end))

            # End caps (optional)
            if self.do_end_caps and self.earcut is not None:
                try:
                    # front cap = ring j=0
                    front_outline = self.model.frame_section(element_name, 0)[:,1:].copy()
                    front_outline *= outline_scale
                    front_tris = self.earcut(front_outline)
                    j0_offset = global_vertex_offset

                    for tri in front_tris:
                        self._triangles.append([
                            j0_offset + tri[0],
                            j0_offset + tri[1],
                            j0_offset + tri[2]
                        ])

                    # back cap = ring j=nen-1
                    back_outline = self.model.frame_section(element_name, nen-1)[:,1:].copy()
                    back_outline *= outline_scale
                    back_tris = self.earcut(back_outline)
                    jN_offset = global_vertex_offset + noe*(nen-1)

                    for tri in back_tris:
                        self._triangles.append([
                            jN_offset + tri[0],
                            jN_offset + tri[1],
                            jN_offset + tri[2]
                        ])
                except Exception as e:
                    warnings.warn(f"Earcut failed on element '{element_name}': {e}")

            # Increase global offset now that we’ve processed all nen rings
            global_vertex_offset += nen * noe

    def vertices(self):
        """Return Nx3 float array of all vertex positions."""
        return np.array(self._positions, dtype=float)

    def triangles(self):
        """Return Mx3 int array of triangle faces."""
        return np.array(self._triangles, dtype=int)

    def ring_ranges(self):
        """
        Return a list of ((element_name, ring_j), start_idx, end_idx).
        The vertex indices in [start_idx:end_idx] belong to that ring.
        """
        return self._ring_ranges
