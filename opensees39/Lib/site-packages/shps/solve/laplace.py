import numpy as np
from functools import partial
import multiprocessing

def _assemble_matrix(Ka, ke, conn, nde, ndf):
    nne = len(conn)
    for j in range(nne):
        for k in range(j + 1):
            for m in range(nde):
                for n in range(nde):
                    Ka[conn[j]*ndf + m, conn[k]*ndf + n] += ke[j*nde + m, k*nde + n]

                    if j != k:
                        Ka[conn[k]*ndf + n, conn[j]*ndf + m] += ke[j*nde + m, k*nde + n]
    return Ka


def _assemble_vector(Fa, fe, nodes, nde, ndf):
    nen = len(nodes)
    for j in range(nen):
        for m in range(nde):
            Fa[nodes[j]* ndf + m] += fe[j * nde + m]
    return Fa


def laplace_neumann(nodes, elements):
    ndf = 1
    nde = 1
    nf = ndf*len(nodes)
    Ka = np.zeros((nf, nf))
    # Ma = np.zeros((nf, nf))
    Fa = np.zeros(nf)

    threads = 6
    chunk = 200
    with multiprocessing.Pool(threads) as pool:
        for conn, (me, ke, fe) in pool.imap_unordered(
                    partial(_wrap_elem02, nodes),
                    elements,
                    chunk):
            Ka  = _assemble_matrix(Ka, ke, conn, nde, ndf)
            # Ma  = _assemble_matrix(Ma, me, conn, nde, ndf)
            Fa  = _assemble_vector(Fa, fe, conn, nde, ndf)

    # Lock the solution at one node and solve for the others
    Pf = Fa[:nf-1]
    for i in range(nf-1):
        Pf[i] -= Ka[i, nf-1]

    Kf = Ka[:nf-1, :nf-1]
    Uf = np.linalg.solve(Kf, Pf)
    ua = np.append(Uf, 1.0)
    return ua

def _wrap_elem02(nodes, elem):
    return elem.nodes, _laplace_triangle(nodes[elem.nodes].T)

def _laplace_triangle(xyz):
    ((y1, y2, y3), (z1, z2, z3)) = xyz

    z12 = z1 - z2
    z23 = z2 - z3
    z31 = z3 - z1
    y32 = y3 - y2
    y13 = y1 - y3
    y21 = y2 - y1

    area = 0.5 * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1))

    me = area/12*(np.eye(3) + np.ones((3,3)))

    k11 = ( y32**2 +  z23**2)
    k12 = (y13*y32 + z23*z31)
    k13 = (y21*y32 + z12*z23)
    k22 = ( y13**2 +  z31**2)
    k23 = (y13*y21 + z12*z31)
    k33 = ( y21**2 +  z12**2)
    ke = 1/(4.0*area)*np.array([[k11, k12, k13],
                                [k12, k22, k23],
                                [k13, k23, k33]])

    fe = -1/6.*np.array([
         ((y1*y32 - z1*z23) + (y2*y32 - z2*z23) + (y3*y32 - z3*z23)),
         ((y1*y13 - z1*z31) + (y2*y13 - z2*z31) + (y3*y13 - z3*z31)),
         ((y1*y21 - z1*z12) + (y2*y21 - z2*z12) + (y3*y21 - z3*z12))])

    return me, ke, fe
