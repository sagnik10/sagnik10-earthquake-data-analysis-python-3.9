"""
solve the problem 

div grad u = f  in Domain
(grad u) n = g  on Boundary

"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import multiprocessing as mp
from functools import partial
from time import perf_counter

try:
    from pypardiso import spsolve as _spsolve
    _PARDISO = True
except ModuleNotFoundError:
    from scipy.sparse.linalg import spsolve as _spsolve
    _PARDISO = False

try:
    from numba import njit as jit 
    from numba import prange
    jit(lambda x: x**2)(1.0)  # test if numba is available

except:
    prange = range
    def jit(*_, **__):
        def _decorator(func):
            return func
        return _decorator


def _assemble_triplet(conn, ke, rows, cols, data):
    """
    Append COO triplets for a 1-dof-per-node element
    """
    for a, i in enumerate(conn):
        for b, j in enumerate(conn):
            rows.append(i)
            cols.append(j)
            data.append(ke[a, b])


@jit(cache=True, fastmath=True, nogil=True)
def _assemble_matrix(Ka, ke, conn, ndf):
    nne = len(conn)
    for a in range(nne):
        ia = conn[a] * ndf
        for b in range(nne):
            ib = conn[b] * ndf
            Ka[ia:ia+ndf, ib:ib+ndf] += ke[a*ndf:(a+1)*ndf,b*ndf:(b+1)*ndf]


@jit(cache=True, fastmath=True, nogil=True)
def _assemble_vector(Fa, fe, conn, ndf):
    nne = len(conn)
    for a in range(nne):
        ia = conn[a] * ndf
        Fa[ia:ia+ndf] += fe[a*ndf:(a+1)*ndf]


def _wrap_elem(nodes,  loads, cell):
    xyz = nodes[cell.nodes].T
    me, ke, by, bz, area = _ElemMap[cell.shape](xyz)

    fe = sum(load(cell, by,bz,area) for load in loads)
    return cell, (me, ke, fe)


M0 = np.array([[2,1,1],
               [1,2,1],
               [1,1,2]],dtype=float)/12

@jit(cache=True, fastmath=True)
def _map_triangle(xyz):

    ((y1, y2, y3), (z1, z2, z3)) = xyz

    z12, z23, z31 = z1-z2, z2-z3, z3-z1
    y32, y13, y21 = y3-y2, y1-y3, y2-y1

    area = abs(0.5 * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)))


    # ∇N
    by = np.array([ z23,  z31,  z12])
    bz = np.array([ y32,  y13,  y21])

    return by, bz, area 

def _hilbert_kernel(cell,by,bz,area, f):
    return area*M0@f[cell.nodes]
    # conn = elem.nodes
    # xyz = nodes[conn].T
    # area  = 0.5 * abs(np.linalg.det(np.vstack((xyz, np.ones(xyz.shape[1])))))
    # f_i, f_j, f_k = f[conn]
    # return area / 12.0 * np.array([2*f_i + f_j + f_k,
    #                                f_i + 2*f_j + f_k,
    #                                f_i + f_j + 2*f_k])

def HilbertLoad(f_nodal):
    return partial(_hilbert_kernel,
                   f=np.asarray(f_nodal))

def _burgers_source(cell,by,bz,area, v, w,i,c):
    S = float(v[cell.nodes] @ M0 @ w[cell.nodes])
    w = 1/2
    return c*sum(S*w * (by,bz)[j] for j in i)

def _poisson_source(cell,by,bz,area, fy, fz):

    #  ∫ (∂N/∂y)^T N dΩ
    Py = (1.0/6.0) * np.outer( by, [1.0, 1.0, 1.0])
    #  ∫ (∂N/∂z)^T N dΩ
    Pz = (1.0/6.0) * np.outer( bz, [1.0, 1.0, 1.0])
    return Py@fy[cell.nodes] + Pz@fz[cell.nodes]

def PoissonLoad(fy, fz):
    return partial(_poisson_source,
                   fy=np.asarray(fy),
                   fz=np.asarray(fz))

def BurgersSource(v, w, i=(0,1), c=1.0):
    return partial(_burgers_source,
                   v=np.asarray(v),
                   w=np.asarray(w),
                   i=i,
                   c=c)

class T3:
    """Linear (3-node) triangle for Poisson/Laplace."""

    @staticmethod
    @jit(cache=True, fastmath=True)
    def elem(xyz):
        ((y1, y2, y3), (z1, z2, z3)) = xyz

        z12, z23, z31 = z1-z2, z2-z3, z3-z1
        y32, y13, y21 = y3-y2, y1-y3, y2-y1

        by, bz, area = _map_triangle(xyz)

        #  ∫ N^T N dΩ
        me = area*M0

        # ∫ (∇N)^T (∇N) dΩ

        k11 =  y32**2  + z23**2
        k12 =  y13*y32 + z23*z31
        k13 =  y21*y32 + z12*z23
        k22 =  y13**2  + z31**2
        k23 =  y13*y21 + z12*z31
        k33 =  y21**2  + z12**2
        ke  = 1/(4.0*area) * np.array([[k11, k12, k13],
                                       [k12, k22, k23],
                                       [k13, k23, k33]])


        # fe = gz @ np.array([y1, y2, y3]) + gy @ np.array([z1, z2, z3])
        # fe =  1/6. * np.array([
        #       (y1*y32 - z1*z23) + (y2*y32 - z2*z23) + (y3*y32 - z3*z23),
        #       (y1*y13 - z1*z31) + (y2*y13 - z2*z31) + (y3*y13 - z3*z31),
        #       (y1*y21 - z1*z12) + (y2*y21 - z2*z12) + (y3*y21 - z3*z12)])
        return me, ke, by, bz, area


class T6:
    """Quadratic (6-node) triangle for Poisson/Laplace."""

    @staticmethod
    @jit(cache=True, fastmath=True)
    def elem(xyz): # (2,6) array: rows=(y,z)
        # quick geometry (use only corner nodes)
        y, z = xyz[:, :3]
        area = 0.5 * ((y[1]-y[0])*(z[2]-z[0]) - (y[2]-y[0])*(z[1]-z[0]))
        if area == 0.0:
            raise ValueError("Degenerate triangle")

        beta  = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        gamma = np.array([z[2]-z[1], z[0]-z[2], z[1]-z[0]])
        gradL = np.vstack((beta, gamma)) / (2.0*area)     # (2,3)

        # 3-point Gauss rule
        gp = np.array([[1/6, 1/6, 2/3],
                       [1/6, 2/3, 1/6],
                       [2/3, 1/6, 1/6]])
        w  = np.full(3, 1/3)

        ke = np.zeros((6, 6))
        for (L1, L2, L3), wt in zip(gp, w):
            gradN = np.zeros((6, 2))

            # vertex nodes
            gradN[0] = (4*L1 - 1)*gradL[:, 0]
            gradN[1] = (4*L2 - 1)*gradL[:, 1]
            gradN[2] = (4*L3 - 1)*gradL[:, 2]
            # midside nodes
            gradN[3] = 4*(L2*gradL[:, 0] + L1*gradL[:, 1])
            gradN[4] = 4*(L3*gradL[:, 1] + L2*gradL[:, 2])
            gradN[5] = 4*(L3*gradL[:, 0] + L1*gradL[:, 2])

            ke += wt*area * (gradN @ gradN.T)

        me = (area / 180.0) * np.array(
            [[ 6, -1, -1,  0,  0,  0],
             [-1,  6, -1,  0,  0,  0],
             [-1, -1,  6,  0,  0,  0],
             [ 0,  0,  0, 32, 16, 16],
             [ 0,  0,  0, 16, 32, 16],
             [ 0,  0,  0, 16, 16, 32]])
        fe = np.zeros(6)
        return me, ke, fe



_GaussQ4 = [
#   ((0.0, 0.0), 1.0)
    ((-1/np.sqrt(3), -1/np.sqrt(3)), 1.0),
    (( 1/np.sqrt(3), -1/np.sqrt(3)), 1.0),
    (( 1/np.sqrt(3),  1/np.sqrt(3)), 1.0),
    ((-1/np.sqrt(3),  1/np.sqrt(3)), 1.0),
]

class Q4:

    @staticmethod
    @jit(cache=True, fastmath=True)
    def elem(xyz):
        pass
        


_ElemMap = {
        'T3': T3.elem,
        'Q4': Q4.elem,
}

def poisson_neumann(
        nodes,
        elements,
        materials=None,
        force=None,
        loads=None,
        threads=6,
        chunk=200,
        fix_node=None,
        fix_value=1.0,
        verbose=False
):

    if loads is None:
        loads = []

    ndf          = 1
    ndof_total   = ndf * len(nodes)
    rows, cols, data = [], [], []
    Fa           = np.zeros(ndof_total)
    if force is not None:
        Fa += force

    #
    tic = perf_counter()
    with mp.Pool(threads) as pool:
        for elem, (_, ke, fe) in pool.imap_unordered(
                partial(_wrap_elem, nodes, loads),
                elements,
                chunk):

            if elem.group is not None and materials is not None:
                assert materials[elem.group] != 0
                ke *= materials[elem.group]

            _assemble_triplet(elem.nodes, ke, rows, cols, data)
            _assemble_vector (Fa, fe, elem.nodes, ndf)

    if verbose:
        print(f"Assembly   : {perf_counter() - tic:6.3f} s")

    # build sparse CSR
    K = sp.coo_matrix(
            (data, (rows, cols)), shape=(ndof_total, ndof_total)
        ).tocsr()

    #
    # Dirichlet fix
    #
    if fix_node is None:
        fix_node = 0
    tic = perf_counter()
    fixed  = np.array([fix_node * ndf])
    free   = np.setdiff1d(np.arange(ndof_total), fixed)

    Pf = Fa[free] - K[free][:, fixed].toarray().ravel() * fix_value
    if verbose:
        print(f"Compatibility: {abs(Fa.sum())/(100*Fa.max()):.6e} (should be 0.0)")

    Kf = K[free][:, free]

    #
    # Solve
    #
    Uf = _spsolve(Kf, Pf)

    if verbose:
        print(f"Solve      : {perf_counter() - tic:6.3f} s "
            f"({ 'PARDISO' if _PARDISO else 'SuperLU' })")

    #
    u = np.empty(ndof_total)
    u[free]  = Uf
    u[fixed] = fix_value
    return u


def pick_elem(xyz):
    return T3.elem(xyz) if xyz.shape[1] == 3 else T6.elem(xyz)


