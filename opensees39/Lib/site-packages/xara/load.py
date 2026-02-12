#===----------------------------------------------------------------------===//
#
#                                   xara
#                              https://xara.so
#
#===----------------------------------------------------------------------===//
#
# Copyright (c) 2025, OpenSees/Xara Developers
# All rights reserved.  No warranty, explicit or implicit, is provided.
#
# This source code is licensed under the BSD 2-Clause License.
# See LICENSE file or https://opensource.org/licenses/BSD-2-Clause
#===----------------------------------------------------------------------===//
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Dict, Iterator, Optional
import math
import numpy as np


def _cell_type(model, eleTag: int) -> str:
    name = model.eleType(eleTag).lower()
    nodes = model.eleNodes(eleTag)
    if "quad" in name and len(nodes) in (4, 8, 9):
        return f"quad{len(nodes)}"

    if "tri" in name and len(nodes) in (3, 6):
        return "tri6" if len(nodes) == 6 else "tri3"


#
# Shape function libraries on a reference edge (ξ ∈ [-1, 1])
#
def N_line2(xi: float) -> np.ndarray:
    # nodes (0, 1) ordered along the edge
    return np.array([(1.0 - xi) * 0.5, (1.0 + xi) * 0.5], dtype=float)


def N_line3(xi: float) -> np.ndarray:
    # nodes (0, 1, 2) ordered along the edge
    # standard quadratic Lagrange on [-1,1]
    return np.array([
        0.5 * xi * (xi - 1.0),   # at -1
        1.0 - xi**2,             # at  0
        0.5 * xi * (xi + 1.0)    # at +1
    ], dtype=float)


SHAPE_FUNCS_FOR_EDGE_NODECOUNT: Dict[int, Callable[[float], np.ndarray]] = {
    2: N_line2,
    3: N_line3,
}


# Local edge definitions per canonical element type
# Indices here are local connectivity indices of the element.
EDGE_MAP: Dict[str, List[Tuple[int, ...]]] = {
    # Quads
    "quad4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "quad8": [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)],
    "quad9": [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)],

    # Tris
    "tri3":  [(0, 1), (1, 2), (2, 0)],
    "tri6":  [(0, 3, 1), (1, 4, 2), (2, 5, 0)],

    # Lines (degenerate surfaces, but supported)
    "line2": [(0, 1)],
    "line3": [(0, 1, 2)],
}

def _edge_length(coords: np.ndarray) -> float:
    # coords: (k, ndm) for edge's k nodes; length based on end nodes (straight chord)
    a, b = coords[0], coords[-1]
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))

@dataclass(frozen=True)
class EdgeRef:
    elem: int
    cell_type: str
    local_nodes: Tuple[int, ...]     # element-local node indices that form this edge
    node_tags: Tuple[int, ...]       # global node tags for the edge nodes
    coords: np.ndarray               # (k, ndm)
    length: float                    # physical chord length (endpoints)
    xi: Tuple[float, ...]            # (-1,1) for 2-node; (-1,0,1) for 3-node

def _edge_pair_lookup(edges: List[EdgeRef]) -> Dict[frozenset, List[Tuple[EdgeRef, float, float]]]:
    """
    Map any unordered node-tag pair on an edge to that edge and the corresponding (xi_a, xi_b).
    For k=2 -> one pair; for k=3 -> 3 pairs (corner↔mid, mid↔corner, corner↔corner).
    """
    lookup: Dict[frozenset, List[Tuple[EdgeRef, float, float]]] = {}
    for e in edges:
        k = len(e.node_tags)
        tags = e.node_tags
        xi   = e.xi

        if k == 2:
            pairs = [(0, 1)]
        elif k == 3:
            pairs = [(0, 1), (1, 2), (0, 2)]
        else:
            continue

        for i, j in pairs:
            key = frozenset((tags[i], tags[j]))
            lookup.setdefault(key, []).append((e, xi[i], xi[j]))

    return lookup

def _build_edge_adjacency(edges: List[EdgeRef]) -> Dict[int, set]:
    adj = defaultdict(set)
    for e in edges:
        tags = list(e.node_tags)
        k = len(tags)
        # immediate neighbors on the edge
        if k == 2:
            pairs = [(0,1)]
        elif k == 3:
            pairs = [(0,1), (1,2)]  # keep (0,2) as a “skip over mid” *extra* option below
        else:
            continue
        for i, j in pairs:
            a, b = tags[i], tags[j]
            adj[a].add(b); adj[b].add(a)
        # # also allow corner↔corner adjacency (Q9/Q8)
        # if k == 3:
        #     a, b = tags[0], tags[2]
        #     adj[a].add(b); adj[b].add(a)
    return adj

def _order_nodes_along_mesh(line_nodes: Sequence[int], adj: Dict[int, set]) -> List[int]:
    S = set(line_nodes)
    # restrict adjacency to the line’s nodes
    r_adj = {n: [m for m in adj.get(n, []) if m in S] for n in S}
    # endpoints (deg 1) preferred; fallback: arbitrary start
    ends = [n for n, ns in r_adj.items() if len(ns) == 1]
    start = ends[0] if ends else (next(iter(S)) if S else None)
    if start is None:
        return list(line_nodes)
    order, visited = [], set()
    cur, prev = start, None
    while cur is not None:
        order.append(cur); visited.add(cur)
        nxts = [m for m in r_adj[cur] if m != prev and m not in visited]
        prev, cur = cur, (nxts[0] if nxts else None)
        if cur is None and len(order) < len(S):
            # disconnected piece: pick another component’s endpoint
            remaining = [n for n in S if n not in visited]
            if remaining:
                cur, prev = remaining[0], None
    # only return if we consumed all nodes; otherwise fallback to original
    return order if len(order) == len(S) else list(line_nodes)



def find_element_edges(model) -> List[EdgeRef]:

    def _edge_xi_for(local_edge: Tuple[int, ...]) -> Tuple[float, ...]:
        k = len(local_edge)
        if k == 2: return (-1.0, +1.0)
        if k == 3: return (-1.0, 0.0, +1.0)
        raise NotImplementedError(f"edge with {k} nodes not supported")

    edges: List[EdgeRef] = []
    for ele in model.getEleTags():
        ctype = _cell_type(model, ele)
        if ctype not in EDGE_MAP:
            continue

        enodes = model.eleNodes(ele)
        for local_edge in EDGE_MAP[ctype]:
            tags = tuple(enodes[i] for i in local_edge)
            coords = np.vstack([
                np.asarray(model.nodeCoord(t), dtype=float) for t in tags
            ])
            L = float(np.linalg.norm(coords[-1] - coords[0]))  # chord
            edges.append(EdgeRef(
                elem=ele, 
                cell_type=ctype,
                local_nodes=tuple(local_edge),
                node_tags=tags, coords=coords, length=L,
                xi=_edge_xi_for(local_edge),
            ))
    return edges



class Line:
    """
    A polyline domain defined by node tags [n1, n2, ..., nk].
    x ∈ [0,1] for the load function is global normalized arclength along this polyline.
    """
    def __init__(self, model, node_tags: Sequence[int]) -> None:
        self.model = model
        self.nodes: List[int] = list(node_tags)
        if len(self.nodes) < 2:
            raise ValueError("Line needs at least two nodes.")


        # Precompute coordinates and cumulative arclength
        self._coords: List[np.ndarray] = [
            np.asarray(self.model.nodeCoord(n), dtype=float)
            for n in self.nodes
        ]
        seg_lengths = [
            np.linalg.norm(self._coords[i+1] - self._coords[i])
            for i in range(len(self.nodes)-1)
        ]

        self._cumlen: List[float] = [0.0]
        total = 0.0
        for L in seg_lengths:
            total += float(L)
            self._cumlen.append(total)
        self._totlen = total
        if self._totlen <= 0.0:
            raise ValueError("Polyline has zero total length.")

    @property
    def measure(self) -> float:
        return self._totlen


    def global_x_from_segment_param(self, seg_index: int, t01: float) -> float:
        """
        Map a local segment parameter t ∈ [0,1] on segment 'seg_index' to global x ∈ [0,1].
        """
        s0 = self._cumlen[seg_index]
        s1 = self._cumlen[seg_index + 1]
        return (s0 + (s1 - s0) * float(t01)) / self._totlen

    def segments(self) -> Iterator[Tuple[int, int, float]]:
        """
        Yield (ni, nj, length) for consecutive pairs.
        """
        for i in range(len(self.nodes)-1):
            ni, nj = self.nodes[i], self.nodes[i+1]
            L = float(np.linalg.norm(self._coords[i+1] - self._coords[i]))
            yield ni, nj, L


    def cell_edges(self) -> List[EdgeRef]:
        """
        Find all element edges that coincide with any polyline segment (by endpoint tags).
        - For higher-order edges (3 nodes), we accept matches where the edge's end nodes
          equal the segment endpoints (mid-node is allowed/ignored for matching).
        """
        def _edge_xi_for(local_edge: Tuple[int, ...]) -> Tuple[float, ...]:
            k = len(local_edge)
            if k == 2: return (-1.0, +1.0)
            if k == 3: return (-1.0, 0.0, +1.0)
            raise NotImplementedError(f"edge with {k} nodes not supported")

        # Build quick lookup of every element's edges
        edges: List[EdgeRef] = []
        for ele in self.model.getEleTags():
            ctype = _cell_type(self.model, ele)
            if ctype not in EDGE_MAP:
                continue

            enodes = self.model.eleNodes(ele)  # element connectivity as global node tags
            for local_edge in EDGE_MAP[ctype]:
                edge_tags = tuple(enodes[i] for i in local_edge)
                # coordinates for the edge nodes (ordered as local_edge)
                coords = np.vstack([np.asarray(self.model.nodeCoord(t), dtype=float)
                                    for t in edge_tags])
                L = _edge_length(coords)
                edges.append(EdgeRef(
                    elem=ele, cell_type=ctype, local_nodes=tuple(local_edge),
                    node_tags=edge_tags, coords=coords, length=L, xi=_edge_xi_for(local_edge)
                ))

        # Keep only edges that match any (nA,nB) (in either direction)
        # We match by endpoints only.
        want_pairs = set()
        for i in range(len(self.nodes)-1):
            nA, nB = self.nodes[i], self.nodes[i+1]
            want_pairs.add((nA, nB))
            want_pairs.add((nB, nA))

        def endpoints(edge: EdgeRef) -> Tuple[int, int]:
            return edge.node_tags[0], edge.node_tags[-1]

        matched = [e for e in edges if endpoints(e) in want_pairs]
        return matched



class SurfaceLoad:
    """
    Consistent nodal load for a scalar function q(x),
    where x ∈ [0,1] is normalized arclength along the Line domain.

    By default applies to a single DOF index 'dof' (0-based) of size model.getNDF().
    """
    def __init__(self,
                 domain: Line,
                 q: Callable[[float], float],
                 dof: int = 1,
                 degree_hint: Optional[int] = None,
                 n_gauss: Optional[int] = None,
                 scale: float = 1.0) -> None:
        
        self._pattern = None
        self._tag = None

        if not isinstance(domain, Line):
            raise TypeError("SurfaceLoad currently supports Line domains.")
        
        self.domain = domain
        self.q = q
        self.dof = int(dof)
        self.scale = float(scale)
        self.degree_hint = int(degree_hint) if degree_hint is not None else None
        self.n_gauss = int(n_gauss) if n_gauss is not None else None

        # ndf = int(self.domain.model.getNDF())
        # if self.dof < 0 or self.dof >= ndf:
        #     raise ValueError(f"dof={self.dof} out of range for model.getNDF()={ndf}")


    def _gauss_rule(self, edge_node_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        If degree_hint wasnt provided and q is polynomial of that degree, choose
        n = ceil((degree_hint + (edge_node_count-1) + 1)/2) as a safe rule of thumb.
        Otherwise, default to 4 or user-provided n_gauss.
        """
        if self.n_gauss is not None:
            n = self.n_gauss

        elif self.degree_hint is not None:
            # integrand degree ~ deg(q) + deg(N). deg(N) = p, where p=edge_node_count-1
            p = edge_node_count - 1
            deg = self.degree_hint + p
            n = max(2, math.ceil((deg + 1) / 2))  # GL(n) integrates deg 2n-1 exactly
        
        else:
            n = 4

        xi, w = np.polynomial.legendre.leggauss(n)
        return xi.astype(float), w.astype(float)



    # def nodal_loads(self) -> Iterable[Tuple[int, np.ndarray]]:
    #     """
    #     Yields (nodeTag, forceVector) with size model.getNDF() (only index 'dof' filled).
    #     Aggregates contributions if a node appears on multiple matched edges.
    #     """
    #     model = self.domain.model
    #     ndf = int(model.getNDF())
    #     accum: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(ndf, dtype=float))

    #     # Build a map from segment endpoints -> (seg_index, Lseg)
    #     seg_info: Dict[Tuple[int, int,], Tuple[int, float, bool]] = {}
    #     for i, (nA, nB, Lseg) in enumerate(self.domain.segments()):
    #         seg_info[(nA, nB)] = (i, Lseg, True)   # forward (nA -> nB)
    #         seg_info[(nB, nA)] = (i, Lseg, False)  # reversed (nB -> nA)

    #     edges = self.domain.find_element_edges()


    #     for edge in edges:
    #         edge_nodes = list(edge.node_tags)
    #         k = len(edge_nodes)
    #         if k not in SHAPE_FUNCS_FOR_EDGE_NODECOUNT:
    #             raise NotImplementedError(f"No edge shape function for k={k} node edge on {edge.cell_type}")
    #         Nfun = SHAPE_FUNCS_FOR_EDGE_NODECOUNT[k]

    #         # Identify which polyline segment this edge corresponds to (by endpoints)
    #         seg_idx, Lseg, forward = seg_info[(edge_nodes[0], edge_nodes[-1])]

    #         # Quadrature rule & Jacobian
    #         xi, w = self._gauss_rule(k)
    #         J = 0.5 * edge.length  # mapping [-1,1] -> physical chord
    #         if J <= 0.0:
    #             continue  # degenerate

    #         #
    #         # Integrate and scatter to edge nodes
    #         #
    #         # Each q evaluation uses global x = normalized arclength along the full Line.
    #         print(k)
    #         for gp, wt in zip(xi, w):
    #             print(" ", gp,wt)
    #             N = Nfun(float(gp))  # length k
    #             # map [-1,1] -> [0,1] on this segment
    #             t_edge = 0.5 * (gp + 1.0)          # [-1,1] -> [0,1] in edge's own direction
    #             t01 = t_edge if forward else (1.0 - t_edge)   # align with Line’s direction
    #             x_global = self.domain.global_x_from_segment_param(seg_idx, t01)
    #             qval = float(self.q(x_global)) * self.scale
    #             factor = qval * wt * J  # contributes to the edge nodal vector

    #             for a, node in enumerate(edge_nodes):
    #                 accum[node][self.dof] += N[a] * factor
    #                 print("   ", node, accum[node][self.dof])

    #     # Yield aggregated nodal loads
    #     for node, f in accum.items():
    #         yield node, f


    # def nodal_loads2(self) -> Iterable[Tuple[int, np.ndarray]]:
    #     model = self.domain.model
    #     ndf = int(model.getNDF())
    #     accum = defaultdict(lambda: np.zeros(ndf, dtype=float))

    #     # all edges once (normalized types)
    #     edges    = find_element_edges(model)
    #     pair_lut = _edge_pair_lookup(edges)
    #     adj      = _build_edge_adjacency(edges)

    #     # re-order the line nodes to follow actual edge-neighbors
    #     line_nodes = _order_nodes_along_mesh(self.domain.nodes, adj)
    #     print("ordered line nodes:", line_nodes)

    #     # build segments from the ordered list
    #     segments = [(line_nodes[i], line_nodes[i+1], i)
    #                 for i in range(len(line_nodes)-1)
    #                 if line_nodes[i] != line_nodes[i+1]]

    #     for (nA, nB, seg_idx) in segments:
    #         key = frozenset((nA, nB))
    #         cand = pair_lut.get(key, [])
    #         if not cand:
    #             # no matching edge for this pair—skip
    #             continue

    #         edge, xi_a, xi_b = cand[0]
    #         k = len(edge.node_tags)
    #         Nfun = SHAPE_FUNCS_FOR_EDGE_NODECOUNT[k]

    #         s_pts, s_wts = self._gauss_rule(k)

    #         A = np.asarray(model.nodeCoord(nA), dtype=float)
    #         B = np.asarray(model.nodeCoord(nB), dtype=float)
    #         Jseg = 0.5 * float(np.linalg.norm(B - A))     # physical dℓ/ds (s∈[-1,1])

    #         print(k)
    #         for s, wt in zip(s_pts, s_wts):
    #             xi = 0.5*(xi_a + xi_b) + 0.5*(xi_b - xi_a)*float(s)
    #             print(" ", xi, wt)
    #             N = Nfun(xi)
    #             t01 = 0.5*(float(s) + 1.0)
    #             x_global = self.domain.global_x_from_segment_param(seg_idx, t01)
    #             qval = float(self.q(x_global)) * self.scale
    #             factor = qval * wt * Jseg
    #             for a, node in enumerate(edge.node_tags):
    #                 accum[node][self.dof] += N[a] * factor
    #                 print("   ", node, accum[node][self.dof])

    #     for node, f in accum.items():
    #         yield node, f

    def nodal_loads(self):
        """
        Matches each consecutive line segment (nA->nB) to the unique edge that contains {nA,nB}
        and integrates the corresponding sub-interval [xi_a, xi_b].
        """
        model = self.domain.model
        ndf = int(model.getNDF())
        accum = defaultdict(lambda: np.zeros(ndf, dtype=float))

        # 1) Build edge inventory once (with xi tags)
        edges = find_element_edges(model)  # your existing helper; uses normalized ctypes + xi
        pair_lut = _edge_pair_lookup(edges)  # unordered {na,nb} -> [(edge, xi_a, xi_b), ...]

        line_nodes = list(self.domain.nodes)


        # 3) Integrate per segment
        for seg_idx in range(len(line_nodes) - 1):
            nA, nB = line_nodes[seg_idx], line_nodes[seg_idx + 1]
            if nA == nB:
                continue

            key = frozenset((nA, nB))
            cands = pair_lut.get(key, [])
            if not cands:
                # No mesh edge contains this pair; skip quietly (or log)
                continue

            # For boundary edges there should be exactly one. If >1 (interior), take first.
            edge, xi_a, xi_b = cands[0]

            k = len(edge.node_tags)
            Nfun = SHAPE_FUNCS_FOR_EDGE_NODECOUNT.get(k)
            if Nfun is None:
                raise NotImplementedError(f"No edge shape function for k={k} on {edge.cell_type}")

            # Gauss on s ∈ [-1,1], mapped to ξ ∈ [xi_a, xi_b]
            s_pts, s_wts = self._gauss_rule(k)

            # Physical sub-segment chord for this line segment
            A = np.asarray(model.nodeCoord(nA), dtype=float)
            B = np.asarray(model.nodeCoord(nB), dtype=float)
            Jseg = 0.5 * float(np.linalg.norm(B - A))   # dℓ/ds (s∈[-1,1])

            for s, wt in zip(s_pts, s_wts):
                xi = 0.5 * (xi_a + xi_b) + 0.5 * (xi_b - xi_a) * float(s)  # sub-edge point
                N  = Nfun(xi)                                              # len=k

                # Global x along the whole line, consistent with user's order
                t01 = 0.5 * (float(s) + 1.0)  # local [0,1] on this segment
                x_eval = self.domain.global_x_from_segment_param(seg_idx, t01)

                qx   = self.q(x_eval)
                factor = wt * Jseg * self.scale

                # Scatter to the k edge nodes in the element's edge order
                for a, node in enumerate(edge.node_tags):
                    for dof, qdof in enumerate(qx):
                        accum[node][dof] += N[a] * factor * qdof

        # 4) Emit
        for n in line_nodes:
            if n in accum:
                yield n, accum[n]

        for n, f in accum.items():
            if n not in set(line_nodes):
                yield n, f


    def nodal_loads(self):
        """
        - For each segment (nA->nB) in the given Line order:
        * find all edges that contain {nA,nB}
        * choose the best candidate (prefer higher k, then prefer non-'line' ctypes)
        * integrate only the sub-interval [xi_a, xi_b] on that edge
        """
        model = self.domain.model
        ndf = int(model.getNDF())
        accum = defaultdict(lambda: np.zeros(ndf, dtype=float))

        # build edge inventory
        edges = find_element_edges(model)  # returns EdgeRef with: node_tags, xi, cell_type, ...
        
        # pair -> list[EdgeRef] (we'll compute xi_a, xi_b per-segment to respect orientation)
        pair_lut = {}
        for e in edges:
            tags = e.node_tags
            k = len(tags)
            if k == 2:
                pairs = [(0, 1)]
            elif k == 3:
                pairs = [(0, 1), (1, 2), (0, 2)]
            else:
                continue
            for i, j in pairs:
                key = frozenset((tags[i], tags[j]))
                pair_lut.setdefault(key, []).append(e)

        # rank candidates: higher k first; then prefer non-line elements
        def _rank(edge):
            k = len(edge.node_tags)
            is_line = str(edge.cell_type).lower().startswith("line")
            return (k, 0 if is_line else 1)  # larger is better

        #
        line_nodes = list(self.domain.nodes)

        # Integrate each segment
        for seg_idx in range(len(line_nodes) - 1):
            nA, nB = line_nodes[seg_idx], line_nodes[seg_idx + 1]
            if nA == nB:
                continue

            cands = pair_lut.get(frozenset((nA, nB)), [])
            if not cands:
                continue  # no matching edge for this pair

            # choose best edge (prefer k=3 over k=2, prefer non-line over line)
            edge = sorted(cands, key=_rank, reverse=True)[0]

            k = len(edge.node_tags)
            Nfun = SHAPE_FUNCS_FOR_EDGE_NODECOUNT.get(k)
            if Nfun is None:
                raise NotImplementedError(f"No edge shape function for k={k} on {edge.cell_type}")

            # map node tag -> xi on this edge; get [xi_a, xi_b] in the segment's direction
            xi_by_tag = {t: edge.xi[i] for i, t in enumerate(edge.node_tags)}
            try:
                xi_a = xi_by_tag[nA]; xi_b = xi_by_tag[nB]
            except KeyError:
                # safety: if edge doesn't actually contain both nodes (shouldn't happen), skip
                continue

            # Gauss on s ∈ [-1,1], map to ξ ∈ [xi_a, xi_b]; physical J from segment chord
            s_pts, s_wts = self._gauss_rule(k)
            A = np.asarray(model.nodeCoord(nA), float)
            B = np.asarray(model.nodeCoord(nB), float)
            Jseg = 0.5 * float(np.linalg.norm(B - A))  # dℓ/ds with s∈[-1,1]

            for s, wt in zip(s_pts, s_wts):
                xi = 0.5 * (xi_a + xi_b) + 0.5 * (xi_b - xi_a) * float(s)
                N  = Nfun(xi)
                t01 = 0.5 * (float(s) + 1.0)  # local [0,1] along nA->nB
                x_eval = self.domain.global_x_from_segment_param(seg_idx, t01)
                qx   = self.q(x_eval)
                factor = self.scale * wt * Jseg
                for a, node in enumerate(edge.node_tags):
                    for dof, qdof in enumerate(qx):
                        accum[node][dof] += N[a] * factor * qdof

        # Emit
        emitted = set()
        for n in line_nodes:
            if n in accum:
                yield n, accum[n]; emitted.add(n)

        for n, f in accum.items():
            if n not in emitted:
                yield n, f


    def _activate(self, model, pattern=None):

        if pattern is None:
            pattern = 1
        model.pattern("Plain", pattern, "Linear")

        for node, force in self.nodal_loads():
            model.load(node, tuple(force.tolist()), pattern=pattern)


class NodalLoads:
    """
    Prescribed nodal loads.
    - loads: Iterable of (nodeTag, forceVector) where forceVector has size model.getNDF().
    - By default applies to a single DOF index 'dof' (0-based) of size model.getNDF().
    """
    def __init__(self,
                 loads: Iterable[Tuple[int, Sequence[float]]],
                 tag: Optional[int] = None,
                 persist: bool = True) -> None:

        if isinstance(loads, dict):
            loads = loads.items()
        self._loads = [(int(n), np.asarray(f, dtype=float)) for n, f in loads]
        self.tag = tag
        self._persistent = bool(persist)


    def _activate(self, model, pattern=None):
        if pattern is None:
            pattern = 1

        model.pattern("Plain", pattern, "Linear")

        for node, force in self._loads:
            model.load(node, tuple(force.tolist()), pattern=pattern)

NodalLoad = NodalLoads  # alias
