"""
This module implements mappings from a parent (natural) element
to a child in the problem domain via the isoparametric map.
"""
import numpy as np
from shps import plane

class IsoparametricMap:

    def __init__(self, parent, nodes=None):
        if nodes is None:
            nodes = parent.nodes
        self.parent = parent
        self.nodes = nodes

        self.ndf = self.ndm = 2


    def interp(self, xi, vals)->np.array:
        shapes = self.parent.basis
        ans = np.zeros(self.ndf)
        for k,shape in shapes.items():
            for df in range(self.ndf):
                ans[df] += shape(*xi)*vals[k][df]

        return ans

    def coord(self, xi):
        shapes = self.parent.basis
        crd = np.zeros(len(next(iter(self.nodes.values()))))
        X = 0.
        Y = 0.
        I4 = IsoparametricMap(plane.Q4, nodes=self.nodes)
        for k,shape in shapes.items():
            shp = shape(*xi)
            if k not in self.nodes:
                crd += shp*np.array(I4.coord(self.parent.nodes[k]))
            #   x,y = I4.coord(self.parent.nodes[k])
            else:
            #   x,y = self.nodes[k]
                crd += shp*np.array(self.nodes[k])

            # X += shp*x
            # Y += shp*y
        return crd #X,Y

    def deriv(self, xi):

        local_shape_derivs = {
            i: [dN[j](*xi) for j in range(self.ndm)]
            for i, dN in self.parent.deriv.items()
        }

        jacinv = self.jacinv(xi)

        global_shape_derivs = {
            i: [
                sum(dN[k]*jacinv[k,j] for k in range(self.ndm))
                    for j in range(self.ndm)
            ] for i,dN in local_shape_derivs.items()
        }
        return global_shape_derivs

    def jac(self, xi):
        return [[
            sum(dN[j](*xi)*x[i]
                for dN, x in zip(self.parent.deriv.values(), self.nodes.values()))
            for j in range(self.ndm)
            ] for i in range(self.ndm)
        ]

    def jacinv(self, xi):
        jac = np.atleast_2d(self.jac(xi))
        det = self.jacdet(jac=jac)
        return np.array([[ jac[1,1],-jac[0,1]],
                         [-jac[1,0], jac[0,0]]])/det

    def jacdet(self, xi=None, jac=None):
        if xi is not None:
            jac = self.jac(xi)
        jac = np.atleast_2d(jac)
        return jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]

    def inverse(self, x, xi=(0., 0.), maxiter=10, tol=1e-10):
        xi = np.atleast_1d(xi)
        x = np.atleast_1d(x)
        for i in range(maxiter):
            f = self.coord(xi) - x
            xi -= self.jacinv(xi)@f

            if np.linalg.norm(f) < tol:
                return xi

        else:
            raise ValueError("Failed to converge")

    def coords(self):
        "symbolic isoparametric map"
        shapes = self.parent.shapes()
        X = 0.
        Y = 0.
        for k,shape in shapes.items():
            if k not in self.nodes:
                x,y = self.parent.nodes[k]
            else:
                x,y = self.nodes[k]
            X += shape*x
            Y += shape*y
        return X,Y

    def jacs(self, xi):
        "symbolic"
        return [[
            sum(dN[j](*xi)*x[i]
                for dN, x in zip(self.parent.derivs.values(), self.nodes.values()))
            for j in range(self.ndm)
            ] for i in range(self.ndm)
        ]

if __name__ == "__main__":
    import plane
    nodes = {
        1: (-2,-2),
        2: ( 4,-3),
        3: ( 2, 3),
        4: (-2, 2)
    }

    test = IsoparametricMap(plane.Lagrange(1), nodes)
    # print(test.jac((0.0, 0.0)))



    # displ = {
    #   1 : ( 0.005 , -0.003),
    #   2 : ( 0.000 ,  0.002),
    #   3 : ( 0.004 ,  0.000),
    #   4 : (-0.005 ,  0.001)
    # }
    # test.trial(displ)
    # print(test.deriv((0.0 ,0.0)))
    # print(test.strain((0.0 ,0.0)))
    print(test.inverse((1.,1.)))


