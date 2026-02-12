import numpy as np
from shps.solve.poisson import poisson_neumann, HilbertLoad, PoissonLoad, BurgersSource


class WarpingAnalysis:
    def __init__(self, model, materials=None):
        self.model = model
        self.nodes = model.nodes
        self.elems = model.elems
        self._materials = materials

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
        return WarpingAnalysis(self.model.translate(vect))

    def section_tensor(self, twist=None, shear=None):

        External, Constrained, Uniform, Neglected = range(4)

        def constraints(shear, twist):
            nwm = 3
            P  = np.eye(6+2*nwm)
            Pa = np.zeros((3,3))
            Pe = np.zeros((3,6))

            if twist == Constrained or twist == Uniform:
                Pa[0,0] = Pe[0,3] = 1

            if shear == Constrained or shear == Uniform:
                Pa[1,1] = Pe[1,1] = 1
                Pa[2,2] = Pe[2,2] = 1


            P[9:12,0:6] = Pe
            P[9:12,9:12] = np.eye(3) - Pa
            P = P[:, ~(P == 0).all(axis=0)][~(P == 0).all(axis=1),:]

            return P

        cnn = self.cnn()
        cmm = self.cmm()
        cnm = self.cnm()
        cnw = self.cnw()
        cnv = self.cnv()
        cmw = self.cmw()
        cmv = self.cmv()
        cvv = self.cvv()
        cww = self.cww()

        owv =  np.zeros((3,3))
        ACA =  np.block([[cnn  , cnm,   cnw,   cnv],
                         [cnm.T, cmm,   cmw,   cmv],
                         [cnw.T, cmw.T, cww,   owv],
                         [cnv.T, cmv.T, owv.T, cvv]])
        
        if twist is None:
            twist = External
        if shear is None:
            shear = External
        P = constraints(shear, twist)
        BCBP = (ACA@P)[:, ~(P == 0).all(axis=0)]
        PBCBP = (P@BCBP)[~(P == 0).all(axis=1),:]
        return PBCBP

    def cnn(self):
        if self._nn is not None:
            return self._nn
        e = np.ones(len(self.model.nodes))
        EA = self.model.inertia(e,e, weight="e")
        GA = self.model.inertia(e,e, weight="g")
        self._nn = np.array([[EA,  0,  0],
                             [ 0, GA,  0],
                             [ 0,  0, GA]])
        return self._nn

    def cmm(self):
        if self._mm is not None:
            return self._mm 

        y,z = self.model.nodes.T
        izy = self.model.inertia(z,y, weight="g")
        izz = self.model.inertia(y,y, weight="e")
        iyy = self.model.inertia(z,z, weight="e")
        self._mm = np.array([[izz+iyy,   0,    0],
                             [   0   , iyy, -izy],
                             [   0   ,-izy,  izz]])
        return self._mm

    def cww(self):
        """
        \\int \\varphi \\otimes \\varphi
        """
        if self._ww is None:
            w = self.solve_twist()
            Iw = self.model.inertia(w, w, weight="e")
            self._ww = np.array([[Iw, 0, 0],
                                 [ 0, 0, 0],
                                 [ 0, 0, 0]])
        return self._ww

    def cvv(self, v=None):
        # w = self.warping()
        if v is None:
            w = self.solve_twist()
            Iww = self.model.energy(w, w, weight="g")
        else:
            Iww = 0

        vv = np.array([[Iww, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        if v is not None:
            vv[1,1] = self.model.energy(v[0], v[0], weight="g")
            vv[2,2] = self.model.energy(v[1], v[1], weight="g")
            vv[1,2] = self.model.energy(v[0], v[1], weight="g")
            vv[2,1] = self.model.energy(v[1], v[0], weight="g")
        return vv


    def cnm(self):
        if self._nm is not None:
            return self._nm
        y,z = self.model.nodes.T
        e  = np.ones(len(self.model.nodes))
        EQy = self.model.inertia(e,z, weight="e")
        EQz = self.model.inertia(e,y, weight="e")
        GQy = self.model.inertia(e,z, weight="g")
        GQz = self.model.inertia(e,y, weight="g")
        self._nm = np.array([[ 0,  EQy, -EQz],
                             [-GQy,  0,    0],
                             [ GQz,  0,    0]])
        return self._nm


    def cmw(self):
        if self._mw is not None:
            return self._mw
        y,z =  self.model.nodes.T
        w = self.solve_twist()
        iwy = self.model.inertia(w,z)
        iwz = self.model.inertia(w,y)
        self._mw = np.array([[  0,  0, 0],
                             [ iwy, 0, 0],
                             [-iwz, 0, 0]])
        return self._mw

    def cmv(self):
        if self._mv is not None:
            return self._mv

        w = self.solve_twist()

        yz  = self.model.nodes
        cxx = self.model.curl(yz, w)
        self._mv = np.array([[cxx, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
        return self._mv


    def shear_factor(self, nu=None, v=None):
        if v is None:
            v = self.shear_warping(nu)

        r = self.model.nodes.T

        ca = nu/2
        a1, a2, c = self._shear_coeffs(nu)
        ix = np.array([[0.0,-1.0],[1.0,0.0]])
        a = r*(ca*np.sum(r**2, axis=0))

        cnn = self.cnn()[1:,1:]
        cmm = self.cmm()[1:,1:]
        D =   -np.linalg.solve(ix@cmm@ix, cnn)/c
        u =  D.T@(v + a) - r

        # for i in range(2):
        #     cnw = self.cnw(u[i])[0,0]
        #     u[i] -= cnw/cnn[0,0]

        cnv = self.cnv(v=u)[1:,1:]
        cvv = self.cvv(v=u)[1:,1:]
        print(np.linalg.solve(cnn,cvv/(6*c))) #cvv, (c-1)*cnn/(3*c)))

        # cs = np.linalg.inv(cnn[1:,1:])@cnv[1:,1:] + np.eye(2)
        if False:
            ann = cnn + (cnv+cnv.T)/(6*c) + cnn/(3*c)
            cs = np.linalg.solve(cnn,ann)
            return np.linalg.diagonal(cs)
        else:
            ann = cnn + (cnv+cnv.T) + cvv
            cs = np.linalg.solve(cnn,ann)
            return 1/np.linalg.eigvals(cs)


    def shear_factor_romano(self, u=None, nu=None):
        if nu is None and u is None:
            nu = 0
            u = self.shear_warping(nu)
        
        a1, a2, c = self._shear_coeffs(nu)
        c1 =  -nu/2
        c2 =   nu/2
        c3 =  0

        r = self.model.nodes.T
        y,z = r

        cmm = self.cmm()[1:,1:]
        cnn = self.cnn()[1:,1:]
        ix = np.array([[0.0,-1.0],[1.0,0.0]])
        # D = -np.linalg.solve(ix@cmm@ix, cnn)/c
        Ji = np.linalg.inv(-ix@cmm@ix)


        # cvv = self.cvv(v=u)[1:,1:]
        _,cvv = self.model._assemble_K1_K2(*u, weight="g")


        brg = self.model.burgers
        iQV = np.array([
            [ brg(z,z,u[0],0)-brg(z,y,u[0],1),  brg(z,z,u[1],0)-brg(z,y,u[1],1) ],
            [ brg(y,y,u[0],1)-brg(y,z,u[0],0),  brg(y,y,u[1],1)-brg(y,z,u[1],0) ]
        ])
        iRV = np.array([
            [ brg(y,y,u[0],0)+brg(y,z,u[0],1),  brg(y,y,u[1],0)+brg(y,z,u[1],1) ],
            [ brg(z,z,u[0],1)+brg(z,y,u[0],0),  brg(z,z,u[1],1)+brg(z,y,u[1],0) ]
        ])

        kae = c1*iQV + c2*iRV  #+  c1*cmm - c2*ix@cmm@ix + c3*cnv

        qr = self.model.quartic
        iQQ = np.array([
            [ qr(y,y,z,z)+qr(z,z,z,z),  -qr(y,y,y,z)-qr(y,z,z,z)],
            [-qr(y,y,y,z)-qr(y,z,z,z),   qr(y,y,y,y)+qr(y,y,z,z)]
        ])
        iRR = np.array([
            [ qr(y,y,z,z)+qr(y,y,y,y),   qr(y,y,y,z)+qr(y,z,z,z)],
            [ qr(y,y,y,z)+qr(y,z,z,z),   qr(z,z,z,z)+qr(y,y,z,z)]
        ])
        kaa = (c1**2 + c3)*iQQ + (c2**2 + c3)*iRR + c3**2*cnn


        FF = kae  + kae.T + kaa + cvv

        ann = Ji.T@(FF*cnn[0,0]/(c**2))@Ji

        k = np.linalg.eigvals(np.linalg.inv(ann))
        return k, ann

    def fiber_shear(self, nu=0):
        u = self.shear_warping(nu)

        a1, a2, c = self._shear_coeffs(nu)
        r = self.model.nodes.T

        cmm = self.cmm()[1:,1:]
        cnn = self.cnn()[1:,1:]
        ix = np.array([[0.0,-1.0],[1.0,0.0]])
        knn, xnn = self.shear_factor_romano(u, nu)
        snn = np.linalg.solve(xnn, cnn)
        D = -np.linalg.solve(ix@cmm@ix, snn)/c

        ca = nu/2 # -nu/(1+nu) # -nu/2
        a = D@r*(ca*np.sum(r**2, axis=0))
        # for i in range(2):
        #     a[i] -= self.cnw(a[i])[0,0]/cnn[0,0]
        u =  D.T@(u + a) - r

        for i in range(2):
            cnw = self.cnw(u[i])[0,0]
            u[i] -= cnw/cnn[0,0]
        return u

    def shear_enhance(self, u, nu):

        a1, a2, c = self._shear_coeffs(nu)
        # c1 = a1
        # c2 = a1 + a2 
        # c3 =  c
        # c3 =   1.0#/c #/c #/c # -1.0 # 1.0 #2*(1+v)
        # c1 *=  1.0 #/c
        # c2 *=  1.0 #/c
        # c2 = -1.0 #/c
        # c1 = -1.0
        # c3 = 0.0
        c1 =   0 # nu/2 # 1 # c
        c2 =   0 # 5/4*nu #nu/2 # 1.5 #c**2 #1/c # c
        c3 =   0 # c
        # c1 =   nu/2
        # c2 =  -nu/2
        c2 = -(5/4)*nu
        c3 =  0
        print(f"{c1 = }, {c2 = }, {c3 = }")
        r = self.model.nodes.T
        y,z = r

        cmm = self.cmm()[1:,1:]
        cnn = self.cnn()[1:,1:]
        ix = np.array([[0.0,-1.0],[1.0,0.0]])
        knn, xnn = self.shear_factor_romano(u, nu)
        snn = np.linalg.solve(xnn, cnn)
        D = -np.linalg.solve(ix@cmm@ix, snn)/c

        ca = nu/2 # -nu/(1+nu) # -nu/2
        a = D@r*(ca*np.sum(r**2, axis=0))
        # for i in range(2):
        #     a[i] -= self.cnw(a[i])[0,0]/cnn[0,0]
        u =  D.T@(u + a) - r

        for i in range(2):
            cnw = self.cnw(u[i])[0,0]
            u[i] -= cnw/cnn[0,0]

        cnv = self.cnv(v=u)[1:,1:]
        cvv = self.cvv(v=u)[1:,1:]
        # _,cvv = self.model._assemble_K1_K2(*u, weight="g")

    
        brg = self.model.burgers
        iQV = np.array([
            [ brg(z,z,u[0],0)-brg(z,y,u[0],1),  brg(z,z,u[1],0)-brg(z,y,u[1],1) ],
            [ brg(y,y,u[0],1)-brg(y,z,u[0],0),  brg(y,y,u[1],1)-brg(y,z,u[1],0) ]
        ])
        iRV = np.array([
            [ brg(y,y,u[0],0)+brg(y,z,u[0],1),  brg(y,y,u[1],0)+brg(y,z,u[1],1) ],
            [ brg(z,z,u[0],1)+brg(z,y,u[0],0),  brg(z,z,u[1],1)+brg(z,y,u[1],0) ]
        ])

        kae = c1*iQV + c2*iRV  +  c1*cmm - c2*ix@cmm@ix + c3*cnv

        qr = self.model.quartic
        iQQ = np.array([
            [ qr(y,y,z,z)+qr(z,z,z,z),  -qr(y,y,y,z)-qr(y,z,z,z)],
            [-qr(y,y,y,z)-qr(y,z,z,z),   qr(y,y,y,y)+qr(y,y,z,z)]
        ])
        iRR = np.array([
            [ qr(y,y,z,z)+qr(y,y,y,y),   qr(y,y,y,z)+qr(y,z,z,z)],
            [ qr(y,y,y,z)+qr(y,z,z,z),   qr(z,z,z,z)+qr(y,y,z,z)]
        ])
        kaa = (c1**2 + c3)*iQQ + (c2**2 + c3)*iRR + c3**2*cnn

        T = iRR
        H = c2**2 * D.T@T@D
        # import scipy.linalg
        # H2 = np.linalg.inv(scipy.linalg.sqrtm(H))
        kaa = H
        kae = kae@D


        C = 1 #nu/(1+nu) #1 #nu/(1+nu) #2*nu/4
        ann = cnn*1 + (cnv + cnv.T) + cvv
        # print(np.linalg.diagonal(np.linalg.solve(cnn,ann)))
        # if c1 != 0 or c2 != 0 or c3 != 0:
        #     aee = kae.T@np.linalg.solve(kaa,kae)        # print(kaa)\
        #     ann -= aee # Ji.T@aee@Ji #/c**2 #@D #np.linalg.solve(D,aee)

        k = np.linalg.eigvals(np.linalg.solve(cnn,ann))
        print(f"Enhanced eigenvalues: {k}")
        return k


    def shear_strain(self, q):
        pass

    def shear_stress(self, q):
        pass

    def _shear_coeffs(self, nu):
        G = 1.0
        E = G*2*(1 + nu)
        v = 2.0*nu*G/E

        # for Ap 
        c = E/G # 2*(1+v)
        a1 =  nu/2 # Q
        a2 = -nu/2 # R
        # a1 =  -c*(1-3*v)/8.0
        # a2 =  -c*(1+v)/4.0

        # a1 =   (1-3*v)/8
        # a2 =   (1+v)/4
        # a1 =  (1+3*v)/8  # (r . r) 1
        # a2 = -(1+v)/4.   #  r o r

        # a1 = -((1+nu)/4 - nu*3/4) /(E/2)
        # a2 = -((1+nu)*3/4 - nu/4) /(E/2)

        return a1, a2, c


    def shear_warping(self, nu=0.28):
        from shps.solve.boundary import find_boundaries, Monomial2D, integrate_boundary
        r = self.model.nodes.T
        y,z = r


        c1, c2, c = self._shear_coeffs(nu)


        cnn = self.cnn()[1:,1:]
        cmm = self.cmm()[1:,1:]

        f = np.zeros((2,len(self.model.nodes)))
        conn = np.array([elem.nodes for elem in self.model.elems])
        boundary_nodes = set()
        for boundary in find_boundaries(conn):
            boundary_nodes.update(boundary)
            # d' = e_y
            f[0] += integrate_boundary(self.model.nodes, boundary,
                                gy_terms=[
                                    # z^2 - y^2
                                    Monomial2D(0,2,  nu/2), Monomial2D(2,0, -nu/2)
                                ],
                                gz_terms=[
                                    Monomial2D(1,1,-nu)
                                ])
            # d' = e_z
            f[1] += integrate_boundary(self.model.nodes, boundary,
                                gz_terms=[
                                    # y^2 - z^2
                                    Monomial2D(0,2, -nu/2), Monomial2D(2,0,  nu/2)
                                ],
                                gy_terms=[
                                    Monomial2D(1,1,-nu)
                                ])

        # Find an interior node to fix
        fix_node = 0
        for i in range(len(self.model.nodes)):
            if i not in boundary_nodes:
                fix_node = i
                break

        u = np.array([
            poisson_neumann(
                self.model.nodes,
                self.model.elems,
                fix_node=fix_node,
                force=f[i],
                loads=[
                    HilbertLoad(r[i]*(2*nu-c)), #  Note: (2*nu-c) == -2.0
                ]
            ) for i in (0,1)
        ])


        for i in range(2):
            cnw = self.cnw(u[i])[0,0]
            u[i] -= cnw/cnn[0,0]

        # return u
        # ca = nu/2
        # ix = np.array([[0.0,-1.0],[1.0,0.0]])
        # a = r*(ca*np.sum(r**2, axis=0))
        # D =  np.linalg.solve(ix@cmm@ix, cnn)/c
        # u =  D.T@(u + a) - r

        # for i in range(2):
        #     cnw = self.cnw(u[i])[0,0]
        #     u[i] -= cnw/(cnn[0,0])
        return u


    def shear_warping1(self, nu=0.28):
        r = self.model.nodes.T
        y,z = r


        c1, c2, c = self._shear_coeffs(nu)


        Cn = self.cnn()[1:,1:]
        Cm = self.cmm()[1:,1:]
        ix = np.array([[0.0,-1.0],[1.0,0.0]])

        u = [
            poisson_neumann(
                self.model.nodes,
                self.model.elems,
                loads=[
                    HilbertLoad(r[i]*c),
                    BurgersSource(y,y,    i=(i,), c=-c1),
                    BurgersSource(z,z,    i=(i,), c=-c1),
                    BurgersSource(y,r[i], i=(0,), c=-c2),
                    BurgersSource(z,r[i], i=(1,), c=-c2)
                ]
            ) for i in (0,1)
        ]

        for i in range(2):
            cnw = self.cnw(u[i])[0,0]
            u[i] -= cnw/(Cn[0,0])

        D = -np.linalg.solve(ix@Cm@ix, Cn)/c
        return D.T@u # + r


    def shear_warping0(self, nu=0.24):

        r = self.model.nodes.T

        G = 1
        E = G*2*(1 + nu)
        u = [
            poisson_neumann(
                self.model.nodes,
                self.model.elems,
                loads=[
                    HilbertLoad(r[i]),
                    BurgersSource(r[ i],r[ i], i=(i,), c= G*nu/2),
                    BurgersSource(r[ i],r[~i], i=(int(not i),), c=G*nu),
                    BurgersSource(r[~i],r[~i], i=(i,), c=-G*nu/2),
                ]
            ) for i in (0,1)
        ]  #+ r/E  #np.ones_like(r)

        Cn = self.cnn()[1:,1:]
        for i in range(2):
            cnw = self.cnw(u[i])[0,0]
            u[i] -= cnw/Cn[0,0]

        Cm = self.cmm()[1:,1:]
        # Cn += self.cnv(shear=True,v=u)[1:,1:]

        ix = np.array([[0.0,-1.0],[1.0,0.0]])

        D = (np.linalg.inv(ix@Cm@ix)@Cn)*G/E
        return D.T@u #-r ##+ r/E #(u - r)
    

    def css(self):
        w = self.solve_twist()

        u = poisson_neumann(
            self.model.nodes,
            self.model.elems,
            loads = [HilbertLoad(w)]
        )
        return self.model.inertia(u,w)

    def cnv(self, shear=False, v=None):
        # if self._nv is not None:
        #     return self._nv
        w = self.solve_twist()

        i = np.zeros_like(self.model.nodes)
        i[:,1] = -1
        # i[:,0] = 1
        cxy = self.model.curl(i, w)
        # i[:,1] = 1
        # i[:,0] = 0
        i[:,0] = 1
        i[:,1] = 0
        cxz = self.model.curl(i, w)

        if shear or (v is not None):
            if v is None:
                v = self.shear_warping()
            o = np.zeros_like(v[0])
            syy = self.model.poisson(v[0],o)
            szz = self.model.poisson(o,v[1])
            syz = self.model.poisson(o,v[0])
            szy = self.model.poisson(v[1],o)
            # syy = self.model.poisson(v[0],v[0])
            # szz = self.model.poisson(v[1],v[1])
            # syz = self.model.poisson(v[0],v[1])
            # szy = self.model.poisson(v[1],v[0])
        else:
            syy = 0
            szz = 0
            syz = 0
            szy = 0

        return np.array([[0.0, 0.0, 0.0],
                         [cxy, syy, szy],
                         [cxz, syz, szz]])
        # return self._nv


    def cnw(self, ua=None)->float:
        # Normalizing Constant = -warpIntegral / A
        c = 0.0

        if ua is not None:
            for i,elem in enumerate(self.model.elems):
                area = self.model.cell_area(i)
                c += sum(ua[elem.nodes])/3.0 * area

        return np.array([[ c , 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])

    def solution(self):
        return self.solve_twist()

    def solve_twist(self):
        """
        We should have 
          self.model.inertia(np.ones(nf), warp) ~ 0.0
        """
        if self._solution is None:
            y, z = self.model.nodes.T
            self._solution = poisson_neumann(
                self.model.nodes,
                self.model.elems,
                loads=[PoissonLoad( z, -y)]
                # loads=[PoissonLoad(-z, y)]
            )
            cnw = self.cnw(self._solution)[0,0]
            cnn = self.cnn()[0,0]
            self._solution -= cnw/cnn

        return self._solution
    

    def centroid(self):
        A = self.cnn()[0,0]
        cnm = self.cnm()
        Qy = cnm[0,1] # int z
        Qz = cnm[2,0] # int y
        return np.array((float(Qz/A), float(Qy/A)))

    def shear_center(self):
        # if self._shear_center is not None:
        #     return self._shear_center
        xc  = self.centroid()
        cmm = self.translate(-xc).cmm()
        # cmm = self.cmm()

        I = np.array([[ cmm[1,1],  cmm[1,2]],
                      [ cmm[2,1],  cmm[2,2]]])

        _, iwy, iwz = self.cmw()[:,0]
        # _, iwz, iwy = -cen.cmw()
        ysc, zsc = np.linalg.solve(I, [iwy, iwz])
        self._shear_center = np.array((
            float(ysc), #-c[0,0], 
            float(zsc), #+c[1,0]
        )) # self.centroid()
        return self._shear_center

    def warping(self):
        if self._warping is not None:
            return self._warping

        w = self.solve_twist() 
        # w = self.translate(-self.centroid()).solution()

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

