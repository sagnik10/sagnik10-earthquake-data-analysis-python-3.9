import numpy as np

import operator
from functools import reduce


def _lagrange_basis(x, x_values):

    if isinstance(x_values, dict):
        keys = [k for k in x_values.keys()]
    else:
        keys = range(len(x_values))

    def _basis(j):
        p = [(x - x_values[m])/(x_values[j] - x_values[m])
             for m in keys if m != j]
        return reduce(operator.mul, p)

    if isinstance(x_values, dict):
        return {k: _basis(k) for k in keys}
    else:
        return [_basis(k) for k in keys]

def _lagrange(x, x_values, y_values, ndm: int = 1):

    assert len(x_values) != 0 and (len(x_values) == len(y_values)), \
            'x and y cannot be empty and must have the same length'

    bases = _lagrange_basis(x, x_values)

    if isinstance(x_values, dict):
        keys = x_values.keys()
    else:
        keys = range(len(x_values))

    return sum(bases[j]*y_values[j] for j in keys)

def _lagrange_polynomial_(
    root: int,
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    roots = np.zeros(shape=(degree+1))
    vals = np.ones(shape=(len(eval_pts)))
    for m in range(degree+1):
        roots[m] = 2*m / degree - 1
    for m in range(degree+1):
        if root != m:
            vals *= (eval_pts - roots[m]) / (roots[root] - roots[m])
    return vals

def lagrange_polynomial(
    degree: int,
    eval_pts
) -> np.ndarray:
    try:
        N = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            N[j] = _lagrange_polynomial_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return N
    except TypeError:
        N = np.zeros(shape=(degree+1))
        for j in range(degree+1):
            N[j] = _lagrange_polynomial_(
                root=j,
                degree=degree,
                eval_pts=[eval_pts]
            )[0]
        return N

def _lagrange_polynomial_derivative_(
    root: int,
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    roots = np.zeros(shape=(degree+1))
    vals = np.zeros(shape=(len(eval_pts)))
    for m in range(degree+1):
        roots[m] = 2*m / degree - 1
    for i in range(degree+1):
        if i != root:
            mvals = np.ones(len(eval_pts))
            for m in range(degree+1):
                if root != m and i != m:
                    mvals *= (eval_pts - roots[m]) / (roots[root] - roots[m])
            vals += 1 / (roots[root] - roots[i]) * mvals
    return vals

def lagrange_polynomial_derivative(
    degree: int,
    eval_pts
) -> np.ndarray:
    try:
        dN = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            dN[j] = _lagrange_polynomial_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return dN
    except TypeError:
        dN = np.zeros(degree+1)
        for j in range(degree+1):
            dN[j] = _lagrange_polynomial_derivative_(
                root=j,
                degree=degree,
                eval_pts=[eval_pts]
            )[0]
        return dN

def _lagrange_polynomial_2_derivative_(
    root: int,
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    roots = np.zeros(shape=(degree+1))
    vals = np.zeros(shape=(len(eval_pts)))
    for m in range(degree+1):
        roots[m] = 2*m / degree - 1
    for i in range(degree+1):
        if i != root:
            mvals = np.zeros(shape=(len(eval_pts)))
            for m in range(degree+1):
                if root != m and i != m:
                    lvals = np.ones(shape=(len(eval_pts)))
                    for l in range(degree+1):
                        if root != l and i != l and m != l:
                            lvals *= (
                                (eval_pts - roots[l]) /
                                (roots[root] - roots[l])
                            )
                    mvals += 1 / (roots[root] - roots[m]) * lvals
            vals += mvals / (roots[root] - roots[i])
    return vals

def lagrange_polynomial_2_derivative(
    degree: int,
    eval_pts
) -> np.ndarray:
    try:
        ddN = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            ddN[j] = _lagrange_polynomial_2_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return ddN
    except TypeError:
        ddN = np.zeros(degree+1)
        for j in range(degree+1):
            ddN[j] = _lagrange_polynomial_2_derivative_(
                root=j,
                degree=degree,
                eval_pts=[eval_pts]
            )[0]
        return ddN

def _lagrange_polynomial_3_derivative_(
    root: int,
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    j = root
    roots = np.zeros(shape=(degree+1))
    vals = np.zeros(shape=(len(eval_pts)))
    for m in range(degree+1):
        roots[m] = 2*m / degree - 1
    for i in range(degree+1):
        if i != j:
            lvals = np.zeros(shape=(len(eval_pts)))
            for l in range(degree+1):
                if l != j and l != i:
                    nvals = np.zeros(shape=(len(eval_pts)))
                    for n in range(degree+1):
                        if n !=j and n!= i and n!= l:
                            mvals = np.ones(shape=(len(eval_pts)))
                            for m in range(degree+1):
                                if m != j and m != i and m != l and m != n:
                                    mvals *= (
                                        (eval_pts - roots[m]) /
                                        (roots[j] - roots[m])
                                    )
                            nvals += 1 / (roots[j] - roots[n]) * mvals
                    lvals += 1 / (roots[j] - roots[l]) * nvals
            vals += lvals / (roots[j] - roots[i])
    return vals

def lagrange_polynomial_3_derivative(
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    try:
        d3N = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            d3N[j] = _lagrange_polynomial_3_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return d3N
    except TypeError:
        d3N = np.zeros(degree+1)
        for j in range(degree+1):
            d3N[j] = _lagrange_polynomial_3_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts[0]
            )[0]
        return d3N

def dual_basis_function(degree, eval_pts):
    n_integration_points = 2 * (degree + 1) - 1
    (sg, wg) = np.polynomial.legendre.leggauss(n_integration_points)
    phi = lagrange_polynomial(degree, sg)
    a = np.zeros((degree+1, degree+1))
    for j in range(degree+1):
        L = np.zeros((degree+1, degree+1))
        R = np.zeros(degree+1)
        for g in range(n_integration_points):
            L += wg[g] * np.outer(phi[:,g], phi[:,g])
            R[j] += wg[g] * phi[j,g]
        a[j] = np.linalg.solve(L, R)
    return a @ lagrange_polynomial(degree, eval_pts)


def displace(coord, u, nsub=20):
    nn = coord.shape[0] # Number of nodes
    xyz = np.array(coord) + np.array(u)

    s = np.linspace(-1, 1, nsub)

    return np.stack([_lagrange(xi, np.linspace(-1, 1, nn), xyz) for xi in s]).T


def lagrange(xyz, nsub=20):
    nn = xyz.shape[0] # Number of nodes

    s = np.linspace(-1, 1, nsub)

    return np.stack([_lagrange(xi, np.linspace(-1, 1, nn), xyz) for xi in s]).T


class Curve:
    def __init__(self, order: int, nodes: dict = None, ndm=1):
        self.order = order
        self.n = order + 1
        self.ndm = ndm
        self._shape = None
        self._shapes = None
        self._coord_symbols = [sp.Symbol("xi")]

        if ndm == 1:
            _x = lambda x: x
        else:
            _x = lambda x: np.array([x, *[0.0]*(ndm-1)])

        if nodes is None:
            nodes = {
                i: _x(x) for i, x in np.linspace(-1.0, 1.0, self.n)
            }

        self.nodes = nodes

    def shapes(self, symbols=False):
        if self._shapes is None:
            xi = self._coord_symbols[0]
            self._shapes = _lagrange_basis(xi, self.nodes, ndm=self.ndm)
        return self._shapes

    @property
    def shapef(self):
        if self._shape is None:
            self._shape = [
                eval(f"lambda xi, eta: {shape}")
                for shape in self.shapes().values()
            ]
        return [x for x in self._shape]


    @property
    def inner(self)->dict:
        return {k: coord for k, coord in self.nodes.items()
                if coord not in (1.0, -1.0)}

    @property
    def outer(self)->dict:
        return {k: coord for k, coord in self.nodes.items()
                if coord in (1.0, -1.0)}

    def shape(self) -> "Basis": ...
    def value(self): ...
    def deriv(self): ...
    def nabla(self): ...
    def print(self): ...


class Lagrange(Curve):
    pass


#
# Meshing
#

def _node_id(ele_id, n_nodes_per_ele):
    return np.array([
        n_nodes_per_ele*ele_id+j for j in range(n_nodes_per_ele+1)
    ], dtype=int)

def line_mesh(A, B, n_elements, order, material, reference_vector, starting_node_index=0,
#              consider_contact_jacobian=False, dual_basis_functions=True, n_contact_integration_points=None
):
    """
    Create line mesh from coordinate A to B.
    """
    n_ele = n_elements
    n_nod = order * n_ele + 1

    coordinates = np.zeros((3,n_nod))
    for i in range(3):
        coordinates[i,:] = np.linspace(A[i], B[i], n_nod)

    beam = []
    for i in range(n_ele):
        element = elmt.SimoBeam(
            nodes=starting_node_index+_node_id(i, order),
            ndf=7,
            yornt=reference_vector,
            coordinates=coordinates[:,_node_id(i, order)],
            **material
        )
        beam.append(element)
    return (coordinates, beam)



def n_point_mesh(points, n_elements, order, material,
        reference_vector, starting_node_index=0
#       consider_contact_jacobian=False,
#       dual_basis_functions=True
        ):
    """
    Create a mesh from a list of points by connecting them in a sequence
    (P1 -- P2 -- P3 -- ... -- PN).

    # Parameters:
    points ...................... points in 3D
    n_elements .................. a list containing the number of elements for each segment
    order ....................... element order (polynomial interpolation order)
    material .................... dictionary with material properties
    reference_vector ............ a vector to define the orientation of the cross-section
    dual_basis_functions ........ a boolean saying if the Lagrange multiplier field should be interpolated with dual shape functions or with Lagrange polynomials
    """

    assert points.shape[1] == len(n_elements) + 1, 'Number of points should be one greater then the length of n_elements list.'
    n_ele = np.array(n_elements)
    n_nod = order * np.sum(n_ele) + 1
    coordinates = np.zeros((3,n_nod))
    for i in range(len(n_ele)):
        n1 = order*np.sum(n_ele[:i])
        n2 = order*np.sum(n_ele[:i])+order*n_ele[i]
        for j in range(n1, n2):
            for k in range(3):
                coordinates[k,j] = points[k,i] + (points[k,i+1] - points[k,i]) * (j - n1) / (n2 - n1)
    coordinates[:,-1] = points[:,-1]

    beam = []
    for i in range(np.sum(n_ele)):
        element = elmt.SimoBeam(
            nodes=starting_node_index+_node_id(i, order),
            ndf=7,
            yornt=reference_vector,
            coordinates=coordinates[:,_node_id(i, order)],
            **material
        )
        beam.append(element)
    return (coordinates, beam)
