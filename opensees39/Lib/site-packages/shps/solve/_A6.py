import numpy as np
import math

try:
    from numba import njit
    njit(lambda x: x**2)(1.0)  # test if numba is available

except:
    def njit(*_, **__):
        def _decorator(func):
            return func
        return _decorator



# T6 shape terms in barycentric form: list of (a,b,c,coef)
Ni_terms = [
    [(2,0,0, 2.0), (1,0,0,-1.0)],        # N1
    [(0,2,0, 2.0), (0,1,0,-1.0)],        # N2
    [(0,0,2, 2.0), (0,0,1,-1.0)],        # N3
    [(1,1,0, 4.0)],                      # N4
    [(0,1,1, 4.0)],                      # N5
    [(1,0,1, 4.0)],                      # N6
]

# dN/dL1, dN/dL2, dN/dL3 for each Ni 
dN_dL = [
    ([(1,0,0, 4.0), (0,0,0,-1.0)], [], []),                       # N1
    ([], [(0,1,0, 4.0), (0,0,0,-1.0)], []),                       # N2
    ([], [], [(0,0,1, 4.0), (0,0,0,-1.0)]),                       # N3
    ([(0,1,0, 4.0)], [(1,0,0, 4.0)], []),                         # N4
    ([], [(0,0,1, 4.0)], [(0,1,0, 4.0)]),                         # N5
    ([(0,0,1, 4.0)], [], [(1,0,0, 4.0)]),                         # N6
]

#
# Exact barycentric monomial integral:
# ∫_Ω L1^a L2^b L3^c dΩ = 2A * (a! b! c!) / (a+b+c+2)!

def _I(A, a, b, c):
    return 2.0*A * math.factorial(a)*math.factorial(b)*math.factorial(c) / math.factorial(a+b+c+2)

# Exact barycentric monomial integral WITHOUT area factor:
# I0(a,b,c) = ∫ L1^a L2^b L3^c dΩ / A 
#           = 2 * (a! b! c!) / (a+b+c+2)!   (since ∫ = A * I0)
def _I0(a, b, c):
    return 2.0 * math.factorial(a)*math.factorial(b)*math.factorial(c) / math.factorial(a+b+c+2)


def _init_hilbert():
    # 
    # M0: 6x6 matrix with ∫ Ni Nj dΩ = A * M0[i,j]
    M0 = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            s = 0.0
            for (a1,b1,c1,ci) in Ni_terms[i]:
                for (a2,b2,c2,cj) in Ni_terms[j]:
                    s += ci*cj * _I0(a1+a2, b1+b2, c1+c2)
            M0[i,j] = s

    return M0


def _init_poisson():
    # Convection terms ,like ∫ N (c . ∇u) dΩ

    # IJ0: 6x6x3 with ∫ (dNi/dLk) Nj dΩ = A * IJ0[i,j,k], k=0..2 for L1,L2,L3
    IJ0 = np.zeros((6,6,3))
    for i in range(6):
        for j in range(6):
            for k in range(3):
                s = 0.0
                for (a1,b1,c1,ci) in dN_dL[i][k]:
                    for (a2,b2,c2,cj) in Ni_terms[j]:
                        s += ci*cj * _I0(a1+a2, b1+b2, c1+c2)
                IJ0[i,j,k] = s

    return IJ0

def _init_sobolev():
    # Stiffness, or diffusion terms

    # IK0: 6x6x3x3 with ∫ (dNi/dLk)(dNj/dLl) dΩ = A * IK0[i,j,k,l]
    IK0 = np.zeros((6,6,3,3))
    for i in range(6):
        for j in range(6):
            for k in range(3):
                for l in range(3):
                    s = 0.0
                    for (a1,b1,c1,ci) in dN_dL[i][k]:
                        for (a2,b2,c2,cj) in dN_dL[j][l]:
                            s += ci*cj * _I0(a1+a2, b1+b2, c1+c2)
                    IK0[i,j,k,l] = s
    return IK0

def _init_burgers():
    # Barycentric moments
    # J0[i,j,k,m] = ∫ (dNi/dLm) Nj Nk dΩ / A   (i,j,k=0..5, m=0..2 for L1..L3)
    J0 = np.zeros((6,6,6,3))
    for i in range(6):
        for j in range(6):
            for k in range(6):

                # multiply dNi/dLm term-by-term with Nj and Nk
                for m in range(3):
                    s = 0.0
                    for (a1,b1,c1,ci) in dN_dL[i][m]:
                        for (a2,b2,c2,cj) in Ni_terms[j]:
                            for (a3,b3,c3,ck) in Ni_terms[k]:
                                s += ci*cj*ck * _I0(a1+a2+a3, b1+b2+b3, c1+c2+c3)
                    J0[i,j,k,m] = s

    return J0


M0  = _init_hilbert()  # 6x6 matrix for Hilbert integral
IJ0 = _init_poisson()
IK0 = _init_sobolev()
IK0_flat = IK0.reshape(9, 36)
J0  = _init_burgers()

class A6:
    """Straight-sided (Affine) T6 (vertices define affine map)."""

    @staticmethod
    @njit(cache=True, fastmath=True)
    def poisson(me, ke, by, c, area, x, y):
        # gy = (1/(2A)) * (b_k * A * IJ0[:,:,k]) = 0.5 * Σ_k b_k IJ0[:,:,k]
        Py = 0.5 * (by[0]*IJ0[:,:,0] + by[1]*IJ0[:,:,1] + by[2]*IJ0[:,:,2])

        # gz = 0.5 * Σ_k c_k IJ0[:,:,k]
        Pz = 0.5 * (c[0]*IJ0[:,:,0] + c[1]*IJ0[:,:,1] + c[2]*IJ0[:,:,2])
        return Py@x + Pz@y
    


    @staticmethod
    def form(xyz):
        ((y1, y2, y3), (z1, z2, z3)) = xyz

        z12, z23, z31 = z1 - z2, z2 - z3, z3 - z1
        y32, y13, y21 = y3 - y2, y1 - y3, y2 - y1

        area = abs(0.5 * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)))

        #  ∫ N^T N dΩ
        me = area*M0

        # ∫ (∇N)^T (∇N) dΩ 
        by = np.array([ z23,  z31,  z12])
        bz = np.array([ y32,  y13,  y21])

        me = area * M0

        # ke = (1/(4A)) * Σ_{k,l} (b_k b_l + c_k c_l) * IK0[:,:,k,l]
        W  = np.outer(bz,bz) + np.outer(by,by)       # 3x3
        # ke = (1.0/(4.0*A)) * np.einsum('kl,ijkl->ij', W, IK0)
        ke = (1.0/(4.0*area)) * np.tensordot(W, IK0, axes=([0,1],[2,3]))  # (6,6)

        return me, ke, bz, by, area

    @staticmethod
    @njit(cache=True, fastmath=True)
    def form_t6_numba_flat(xyz, M0, IJ0, IK0f):
        y, z = xyz
        y1,y2,y3 = y; z1,z2,z3 = z
        z12, z23, z31 = z1-z2, z2-z3, z3-z1
        y32, y13, y21 = y3-y2, y1-y3, y2-y1

        A = abs(0.5*((y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)))

        b = np.array([-y32, -y13,  y21])
        c = np.array([-z23, -z31, -z12])

        me = A * M0

        # gy = (1/(2A)) * (b_k * A * IJ0[:,:,k]) = 0.5 * Σ_k b_k IJ0[:,:,k]
        gy = 0.5 * (b[0]*IJ0[:,:,0] + b[1]*IJ0[:,:,1] + b[2]*IJ0[:,:,2])

        # gz = 0.5 * Σ_k c_k IJ0[:,:,k]
        gz = 0.5 * (c[0]*IJ0[:,:,0] + c[1]*IJ0[:,:,1] + c[2]*IJ0[:,:,2])

        W  = (np.outer(b,b) + np.outer(c,c)).reshape(9)
        ke = (1.0/(4.0*A)) * np.dot(W, IK0f).reshape(6,6)

        return me, ke, gy, gz, A