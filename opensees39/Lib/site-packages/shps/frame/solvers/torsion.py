#
# Claudio Perez
#
import numpy as np
from functools import partial
import multiprocessing

class TorsionAnalysis:
    def __init__(self, model):
        self.model = model
        self.nodes = model.nodes
        self.elems = model.elems

        self._solution = None
        self._warping  = None 
        self._centroid = None
        self._shear_center = None

        self._nn = None
        self._mm = None 
        self._ww = None
        self._vv = None
        self._nm = None
        self._mw = None
        self._mv = None
        self._nv = None

    def translate(self, vect):
        return TorsionAnalysis(self.model.translate(vect))

    def section_tensor(self):
        owv = np.zeros((1,1))
        return np.block([[self.cnn()  , self.cnm(),   self.cnw(), self.cnv()],
                         [self.cnm().T, self.cmm(),   self.cmw(), self.cmv()],
                         [self.cnw().T, self.cmw().T, self.cww(),      owv  ],
                         [self.cnv().T, self.cmv().T,      owv.T, self.cvv()]])

    def cnn(self):
        if self._nn is not None:
            return self._nn
        e = np.ones(len(self.model.nodes))
        A = self.model.inertia(e,e)
        self._nn = np.array([[A, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])
        return self._nn

    def cmm(self):
        if self._mm is not None:
            return self._mm 

        y,z = self.model.nodes.T
        izy = self.model.inertia(z,y)
        izz = self.model.inertia(y,y)
        iyy = self.model.inertia(z,z)
        self._mm = np.array([[izz+iyy,   0,    0],
                             [   0   , iyy, -izy],
                             [   0   ,-izy,  izz]])
        return self._mm

    def cww(self):
        """
        \\int \\varphi \\otimes \\varphi
        """
        if self._ww is None:
            w = self.solution()
            Iw = self.model.inertia(w, w)
            self._ww = np.array([[Iw]])
        return self._ww

    def cvv(self):
        if self._vv is None:
            # w = self.warping()
            w = self.solution()
            Iww = self.model.energy(w, w)
            self._vv = np.array([[Iww]])
        return self._vv

    def cnm(self):
        if self._nm is not None:
            return self._nm
        y,z = self.model.nodes.T
        e  = np.ones(len(self.model.nodes))
        Qy = self.model.inertia(e,z)
        Qz = self.model.inertia(e,y)
        self._nm = np.array([[ 0,  Qy, -Qz],
                             [-Qy,  0,   0],
                             [ Qz,  0,   0]])
        return self._nm

    def cmw(self):
        if self._mw is not None:
            return self._mw
        y,z =  self.model.nodes.T
        w = self.solution()
        iwy = self.model.inertia(w,z)
        iwz = self.model.inertia(w,y)
        self._mw = np.array([[  0 ],
                             [ iwy],
                             [-iwz]])
        return self._mw

    def cmv(self):
        if self._mv is not None:
            return self._mv

        w = self.solution()

        yz  = self.model.nodes
        cxx = self.model.curl(yz, w)
        self._mv = np.array([[cxx],
                             [0.0],
                             [0.0]])
        return self._mv

    def cnv(self):
        if self._nv is not None:
            return self._nv
        w = self.solution()

        i = np.zeros_like(self.model.nodes)
        i[:,1] = -1
        cxy = self.model.curl(i, w)
        i[:,0] = 1
        i[:,1] = 0
        cxz = self.model.curl(i, w)
        self._nv = np.array([[0.0],
                             [cxy],
                             [cxz]])
        return self._nv


    def cnw(self, ua=None)->float:
        # Normalizing Constant = -warpIntegral / A
        c = 0.0

        if ua is not None:
            for i,elem in enumerate(self.model.elems):
                area = self.model.cell_area(i)
                c += sum(ua[elem.nodes])/3.0 * area

        return np.array([[ c ], 
                         [0.0],
                         [0.0]])


    def solution(self):
        """
        # We should have 
        #   self.model.inertia(np.ones(nf), warp) ~ 0.0
        """
        if self._solution is None:
            self._solution = laplace_neumann(self.model.nodes, self.model.elems)
            cnw = self.cnw(self._solution)[0,0]
            cnn = self.cnn()[0,0]
            self._solution -= cnw/cnn
        return self._solution
    

    def centroid(self):
        if self._centroid is not None:
            return self._centroid
        A = self.cnn()[0,0]
        cnm = self.cnm()
        Qy = cnm[0,1] # int z
        Qz = cnm[2,0] # int y
        self._centroid = float(Qz/A), float(Qy/A)
        return self._centroid

    def shear_center(self):
        if self._shear_center is not None:
            return self._shear_center

        cmm = self.translate(-self.centroid()).cmm()
        # cmm = self.cmm()

        I = np.array([[ cmm[1,1],  cmm[1,2]],
                      [ cmm[2,1],  cmm[2,2]]])

        _, iwy, iwz = self.cmw()[:,0]
        # _, iwz, iwy = -cen.cmw()
        ysc, zsc = np.linalg.solve(I, [iwy, iwz])
        self._shear_center = (
            float(ysc), #-c[0,0], 
            float(zsc), #+c[1,0]
        )
        return self._shear_center

    def warping(self):
        if self._warping is not None:
            return self._warping

        w = self.solution() 
        # w = self.translate(self.centroid()).solution()

        y,   z = self.model.nodes.T
        cy, cz = self.centroid()
        yc = y - cy 
        zc = z - cz
        sy, sz = self.shear_center()
        # sy = -sy 
        # sz = -sz
        # w =  w + np.array([ys, -zs])@self.model.nodes.T
        w = w + sy*zc - sz*yc

        self._warping = w

        return self._warping


    def torsion_constant(self):
        """
        Compute St. Venant's constant.
        """
        # J = Io + Irw
        return self.cmm()[0,0] + self.cmv()[0,0]

        nodes = self.model.nodes
        J  = 0
        for i,elem in enumerate(self.model.elems):
            ((y1, y2, y3), (z1, z2, z3)) = nodes[elem.nodes].T

            z23 = z2 - z3
            z31 = z3 - z1
            z12 = z1 - z2
            y32 = y3 - y2
            y13 = y1 - y3
            y21 = y2 - y1

            u1, u2, u3 = warp[elem.nodes]

            # Element area
            area = self.model.cell_area(i)

            # St. Venant constant
            Czeta1  = ( u2*y1 * y13 + u3 *  y1 * y21 + u1 * y1*y32 - u3 * z1 * z12 - u1*z1 * z23 - u2*z1*z31)/(2*area)
            Czeta2  = (u2*y13 *  y2 + u3 *  y2 * y21 + u1 * y2*y32 - u3 * z12 * z2 - u1*z2 * z23 - u2*z2*z31)/(2*area)
            Czeta3  = (u2*y13 *  y3 + u3 * y21 *  y3 + u1 * y3*y32 - u3 * z12 * z3 - u1*z23 * z3 - u2*z3*z31)/(2*area)
            Czeta12 = 2*y1*y2 + 2*z1*z2
            Czeta13 = 2*y1*y3 + 2*z1*z3
            Czeta23 = 2*y2*y3 + 2*z2*z3
            Czeta1s =   y1**2 +   z1**2
            Czeta2s =   y2**2 +   z2**2
            Czeta3s =   y3**2 +   z3**2
            J += ((Czeta1+Czeta2+Czeta3)/3. \
                + (Czeta12+Czeta13+Czeta23)/12. \
                + (Czeta1s+Czeta2s+Czeta3s)/6.)*area

        return float(J)


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

    me = None #area/12*(np.eye(3) + np.ones((3,3)))

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
