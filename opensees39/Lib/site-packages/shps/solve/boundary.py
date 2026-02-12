
#
#
#
import numpy as np
from math import comb

from collections import defaultdict

def find_boundaries(conn):
    """
    Yield ordered boundary loops (outer and holes) as lists of node indices.

    Parameters
    ----------
    nodes : (N,2) array_like
        Nodal coordinates [y,z]. (Not required for detection, but kept for API symmetry.)
    conn : iterable of iterables
        Element connectivity; each entry is a sequence of node indices (e.g. len=3 for T3, 4 for Q4).

    Yields
    ------
    loop : list[int]
        Ordered node indices around one boundary loop (start not repeated at end).
    """
    # 1) Count undirected edge occurrences
    edge_count = {}
    for elem in conn:
        e = list(elem)
        if len(e) < 2: 
            continue
        for a, b in zip(e, e[1:] + e[:1]):
            if a == b: 
                continue
            k = (a, b) if a < b else (b, a)
            edge_count[k] = edge_count.get(k, 0) + 1

    # 2) Build adjacency graph using only edges that occur once (boundary edges)
    adj = defaultdict(list)
    for elem in conn:
        e = list(elem)
        if len(e) < 2:
            continue
        for a, b in zip(e, e[1:] + e[:1]):
            if a == b:
                continue
            k = (a, b) if a < b else (b, a)
            if edge_count.get(k, 0) == 1:
                adj[a].append(b)
                adj[b].append(a)

    # 3) Walk boundary loops; mark undirected edges visited
    visited = set()  # stores undirected edge keys (min, max)

    for u in list(adj.keys()):
        for v in adj[u]:
            k0 = (u, v) if u < v else (v, u)
            if k0 in visited:
                continue

            loop = [u]
            curr, nxt = u, v

            while True:
                visited.add((curr, nxt) if curr < nxt else (nxt, curr))
                loop.append(nxt)

                # choose the next neighbor from 'nxt' that isn't the edge we just came from
                nxt_neighbors = adj[nxt]
                w = None
                for cand in nxt_neighbors:
                    if cand == curr:
                        continue
                    k = (nxt, cand) if nxt < cand else (cand, nxt)
                    if k not in visited:
                        w = cand
                        break

                if w is None:
                    # either we've closed the loop or hit an open chain end
                    if loop[0] == loop[-1]:
                        loop.pop()  # remove duplicated start if present
                    yield loop
                    break

                curr, nxt = nxt, w


# ----- c * y^ey * z^ez -----
class Monomial2D:
    __slots__=("ey","ez","c")
    def __init__(self, ey, ez, c):
        self.ey=int(ey); self.ez=int(ez); self.c=float(c)
    def edge_coeffs(self, yi,zi,yj,zj):
        dy, dz = (yj-yi), (zj-zi)
        deg = self.ey + self.ez
        a = np.zeros(deg+1, float)
        # (yi+dy s)^ey (zi+dz s)^ez = sum_p sum_q binom * yi^(ey-p) dy^p * zi^(ez-q) dz^q * s^(p+q)
        for p in range(self.ey+1):
            by = comb(self.ey,p) * (yi**(self.ey-p)) * (dy**p)
            for q in range(self.ez+1):
                bz = comb(self.ez,q) * (zi**(self.ez-q)) * (dz**q)
                a[p+q] += self.c * by * bz
        return a

class ScalarPoly:
    def __init__(self, terms=None):
        self.terms=[t for t in (terms or []) if t is not None and t.c!=0.0]
    def edge_coeffs(self, yi,zi,yj,zj):
        acc = np.zeros(1, float)
        for t in self.terms:
            a = t.edge_coeffs(yi,zi,yj,zj)
            if a.size>acc.size: acc = np.pad(acc,(0,a.size-acc.size))
            if a.size<acc.size: a = np.pad(a,(0,acc.size-a.size))
            acc += a
        return acc

class VectorPoly:
    def __init__(self, gy_terms=None, gz_terms=None):
        self.qy = ScalarPoly(gy_terms or [])
        self.qz = ScalarPoly(gz_terms or [])
    def edge_projection_coeffs(self, yi,zi,yj,zj, ny,nz):
        ay = self.qy.edge_coeffs(yi,zi,yj,zj)
        az = self.qz.edge_coeffs(yi,zi,yj,zj)
        m = max(len(ay), len(az))
        if len(ay)<m: ay = np.pad(ay,(0,m-len(ay)))
        if len(az)<m: az = np.pad(az,(0,m-len(az)))
        return ny*ay + nz*az


def _loop_orientation_sign(coords, idx):
    # +1 for CCW, -1 for CW, using (y,z) as (x,y)
    ys = coords[idx,0]; zs = coords[idx,1]
    ys = np.r_[ys, ys[0]]; zs = np.r_[zs, zs[0]]
    area2 = np.sum(ys[:-1]*zs[1:] - ys[1:]*zs[:-1])
    return 1.0 if area2>0 else -1.0

def _edge_nodal_from_coeffs(a, L):
    k = np.arange(len(a), dtype=float)
    Fi = L * np.sum(a * (1.0/(k+1.0) - 1.0/(k+2.0)))
    Fj = L * np.sum(a * (1.0/(k+2.0)))
    return Fi, Fj


class BoundaryTerm:
    """Return coefficients a[0..d] for g(s) = sum_k a[k] s^k along an edge."""
    def poly_coeffs_along_edge(self, yi, zi, yj, zj):
        raise NotImplementedError

# General monomial: c * y^ey * z^ez
class MonomialBoundary(BoundaryTerm):
    def __init__(self, ey, ez, c):
        self.ey = int(ey); self.ez = int(ez); self.c = float(c)

    def poly_coeffs_along_edge(self, yi, zi, yj, zj):
        dy, dz = (yj - yi), (zj - zi)
        deg = self.ey + self.ez
        a = np.zeros(deg + 1, dtype=float)
        # (yi + dy s)^ey (zi + dz s)^ez = sum_{p=0..ey} sum_{q=0..ez} binom * ... * s^{p+q}
        for p in range(self.ey + 1):
            by = comb(self.ey, p) * (yi ** (self.ey - p)) * (dy ** p)
            for q in range(self.ez + 1):
                bz = comb(self.ez, q) * (zi ** (self.ez - q)) * (dz ** q)
                a[p + q] += self.c * by * bz
        return a


def _edge_nodal_from_coeffs(a, L):
    """
    Given coefficients a[k] for g(s)=sum a[k] s^k on s in [0,1], return (Fi,Fj)
    with N_i=1-s, N_j=s and ds_edge = L ds.
    """
    k = np.arange(len(a), dtype=float)
    Fi = L * np.sum(a * (1.0/(k+1.0) - 1.0/(k+2.0)))
    Fj = L * np.sum(a * (1.0/(k+2.0)))
    return Fi, Fj

def integrate_boundary0(node_coords, boun_indices, terms, close_loop=True):
    """
    Exact consistent nodal loads for a straight-edge boundary loop of T3/Q4.
    Works for any polynomial degree via analytic formulas.
    """
    coords = np.asarray(node_coords, float)
    n = coords.shape[0]
    F = np.zeros(n, float)

    idx = list(boun_indices)
    if len(idx) < 2:
        return F

    pairs = [(idx[k], idx[k+1]) for k in range(len(idx)-1)]
    if close_loop and idx[0] != idx[-1] and len(idx) >= 3:
        pairs.append((idx[-1], idx[0]))

    for i, j in pairs:
        yi, zi = coords[i]; yj, zj = coords[j]
        L = float(np.hypot(yj - yi, zj - zi))
        if L == 0.0:
            continue

        # accumulate polynomial coefficients (length may vary per term)
        a_sum = np.zeros(1, float)  # start with degree 0
        for t in terms:
            # Backward-compat: accept old terms with .coeffs_along_edge()
            if hasattr(t, "poly_coeffs_along_edge"):
                a = np.asarray(t.poly_coeffs_along_edge(yi, zi, yj, zj), float)
            else:
                a0,a1,*rest = t.coeffs_along_edge(yi, zi, yj, zj)
                a = np.array([a0,a1,*rest], float)
            # pad/extend
            if len(a) > len(a_sum):
                a_sum = np.pad(a_sum, (0, len(a)-len(a_sum)))
            if len(a) < len(a_sum):
                a = np.pad(a, (0, len(a_sum)-len(a)))
            a_sum += a

        Fi, Fj = _edge_nodal_from_coeffs(a_sum, L)
        F[i] += Fi; F[j] += Fj
    return F


def integrate_boundary(node_coords,
                       boun_indices, 
                       gy_terms, 
                       gz_terms,
                       close_loop=True, normal_sign_override=None):
    """
    Exact, consistent Neumann load assembly for one boundary loop (outer or hole).
    node_coords: (N,2) [y,z]
    boun_indices: ordered node indices around the loop
    gy_terms, gz_terms: lists of Monomial2D terms for q=(qy,qz)
    normal_sign_override: None or +/-1 to flip outward if your orientation differs
    """
    coords = np.asarray(node_coords, float)
    F = np.zeros(len(coords), float)
    if len(boun_indices)<2: return F

    orient = _loop_orientation_sign(coords, np.array(boun_indices, int))
    if normal_sign_override is not None:
        orient *= float(normal_sign_override)

    pairs = [(boun_indices[k], boun_indices[k+1]) for k in range(len(boun_indices)-1)]

    if close_loop and boun_indices[0] != boun_indices[-1] and len(boun_indices)>=3:
        pairs.append((boun_indices[-1], boun_indices[0]))

    q = VectorPoly(gy_terms, gz_terms)

    for i,j in pairs:
        yi,zi = coords[i]; yj,zj = coords[j]
        dy, dz = (yj-yi), (zj-zi)
        L = float(np.hypot(dy,dz))
        if L==0.0: continue

        # outward unit normal (right normal for CCW)
        ny, nz = orient*np.array([dz/L, -dy/L])

        # polynomial coefficients for g(s)=q·n
        a = q.edge_projection_coeffs(yi,zi,yj,zj, ny,nz)

        Fi, Fj = _edge_nodal_from_coeffs(a, L)
        F[i] += Fi; F[j] += Fj

    return F