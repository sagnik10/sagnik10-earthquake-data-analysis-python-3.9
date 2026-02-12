import numpy as np

class Hermite1D:
    def __init__(self, degree: int):
        self.degree = degree
    @property
    def monomials(self):
        return lambda x: [eval(t, dict(x=x)) for t in self.terms]

    @property
    def terms(self):
        sip = self.degree + 1
        return [f"x**{i}" for i in range(sip)]

    def vandermonde(self):
        sip = self.degree + 1

        nn  = int(np.ceil(sip/2))    # no. of auxialiry nodes

        # create and re-order xn to set the ends node as the first 
        # two nodes and to put the node with one dof at the end
        xn  = np.linspace(-1,1,nn).T
#       xn  = np.linspace(0,1,nn).T
        xn  = xn [np.array([ 0,-1, *range(1, nn-2) ])]

        # if "sip" is even, all the node have two dofs; if "sip" is odd, all the
        # nodes have two dofs except the last node, which has one dof
        DV = nn if not sip%2 else nn-1

        # compute matrix to be inverted
        DIM = nn + DV;
        C = np.zeros((DIM,DIM));
        for n in range(DIM):
            e = DIM - (n + 1);
            C[:nn, n] = xn**e;
            if n != DIM-1:
                C[nn:DIM,n] = (xn[:DV]**(e-1))*e;
        return C



def hermite(degree:int, deriv:int, xi):
#HERMITE Hermite interpolation polynomials in interval -1<xi<1
#  HP = HERMITE (DEGREE,DERIV,XI)
#  the function determines the values of Hermite interpolation polynomials of degree DEGREE
#  and derivative order DERIV at integration points in vector XI;
#  the values are returned in array HP with rows representing the different Hermite
#  polynomials of degree DEGREE and columns representing the values at points XI
#  NOTE: XI need to be supplied in the interval -1<xi<1
#  EXAMPLE: Hermite(3,2,xi) returns the second derivative of cubic Hermite polynomials at xi
#
#  If degree is even, one node of the equispaced grid used to evaluate the
#  polynomials considers only the value of ordinate, without the
#  derivative. This node is always the last node of the grid, considering
#  that the end nodes are located in the first two positions.
#
#  To go from the interval [-1;+1] to the interval [0;L]:
#     Jac = 0.5*L;    xP = Jac.*(1.+xi);
#     hp(1:2:size(hp,1),:) = hp(1:2:size(hp,1),:)./(Jac^deriv);
#     hp(2:2:size(hp,1),:) = hp(2:2:size(hp,1),:)./(Jac^(deriv-1));
    xi  = np.array(xi)
    hp  = np.zeros((degree + 1,len(xi)))

    for i,p in enumerate(hermite_polynomial(degree,deriv)):
        # create the basis derivative
        poly = np.polyder(p, deriv)

        # evaluate polynomial
        hp[i-1,:] = np.polyval(poly, xi.T);

    return hp


def hermite_polynomial(degree: int, deriv: int):

    sip = degree + 1

    # obtain auxiliary point locations
    nn  = int(np.ceil(sip/2))    # no. of auxialiry nodes

    # create and re-order xn to set the ends node as the first 
    # two nodes and to put the node with one dof at the end
    xn  = np.linspace(-1,1,nn).T
    xn  = xn [np.array([ 0,-1, *range(1, nn-2) ])]

    # if "sip" is even, all the node have two dofs; if "sip" is odd, all the
    # nodes have two dofs except the last node, which has one dof

    DV = nn if not sip%2 else nn-1

    # compute matrix C to be inverted
    DIM = nn + DV;
    C = np.zeros((DIM,DIM));
    for n in range(DIM):
        e = DIM - (n + 1);
        C[:nn, n] = xn**e;
        if n != DIM-1:
            C[nn:DIM,n] = (xn[:DV]**(e-1))*e;


    pp = []
    # compute interpolation function
    for i in range(1,sip+1):
        # set equal to 1 the ordinate of the current point
        n  = int(np.ceil(i/2)) - 1;      # current point
        y  = np.zeros((nn,1))
        dy = np.zeros((DV,1))
        #
        if not sip % 2:
            # if "sip" is even, all the node have two dofs
            # for the first dof of the node
            if i%2 != 0:
                y[n]  = 1;
                dy[n] = 0;
            # for the second dof of the node
            else:
                y[n]  = 0
                dy[n] = 1

        else:
            # if "sip" is odd, all the node have two dofs except the last node, 
            # which has one dof for the last dof (except the end nodes)
            if i == sip:
                y[n] = 1;
            # for the first dof of the node
            elif i%2 != 0:
                y[n]  = 1
                dy[n] = 0
            # for the second dof of the node
            else:
                y[n]  = 0
                dy[n] = 1

        # evaluate polynomial coefficient for current point
        pp.append(np.linalg.solve(C, np.vstack([y,  dy])).flatten())
    return pp

print(hermite(3, 2, [-0.1, 0.1, 0.15, 0.2]))

from shps.shape import stringify, make_shapes
import sympy as sp
print(stringify(Hermite1D(3), xi=[sp.symbols("xi")], latex=True))
# h = Hermite1D(3)
# print(
#     "\n".join(
#         sp.latex(e) for e in make_shapes(h.vandermonde(), h.monomials, xi=[sp.symbols("xi")]).values()
#     )
# )
#print(np.linalg.inv(Hermite1D(3).vandermonde()))

