# functions for building interpolation routines
#
# build.py or 
# forge.py
#
from itertools import product
import sympy as sp
import shps.gauss
from shps.plane import Lagrange

class Shape: ...

def tokens(shape: Shape):
    pass

def latex(shape: Shape):
    pass

def ccode(shape: Shape):
    pass

"""
int
poisson_{form}{nodes}_G{quad}_tangent(struct {form}* data, double restrict xyz[{nen}][{ndm}], double restrict k[{nst}][{nst}])
{{
{dvol}
{stif}
}}
"""

def elem():
    nip = 4
    nen = 4
    ndm = 2
    el = Lagrange(1)
    shapes = list(el.shapes().values())
    xi, eta = sp.symbols("xi eta")
    shaped = [[s.diff(xi), s.diff(eta)] for s in shapes]


    quad = shps.gauss.Legendre(3)
    # weights = quad.weights
    print([zip(quad.points, quad.weights), zip(quad.points, quad.weights)])
    points = list(product(quad.points, quad.points))
    weights = [i*j for i,j in product(quad.weights, quad.weights)]


    #points,weights = zip(*product(zip(quad.points, quad.weights), zip(quad.points, quad.weights))) #[[-1, -1], [1, -1], [1, 1], [-1, 1]]
    print(points)
    print(weights)

    def dshp(l):
        return [
            [str(dsdx).replace("xi", str(points[l][0])).replace("eta", str(points[l][1]))  for dsdx in dsh]
            for dsh in shaped
        ]

    def ijac(l):
        return [["a", "b"], ["c", "d"]]

    return "\n\n".join(
        ";\n".join(
            f"  k[{i}][{j}] = k[{j}][{i}] = k[{i}][{j}] + " +
            f"{weights[l]}*({' + '.join(f'({ijac(l)[m][k]})*({ijac(l)[m][n]})*({dshp(l)[i][k]})*({dshp(l)[j][n]})' for m in range(2) for n in range(2) for k in range(2))})"
            for i in range(nen)  for j in range(nen)
        ) for l in range(nip)
    )


if __name__ == "__main__":
    print(elem())

