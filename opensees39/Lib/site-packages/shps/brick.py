#!/usr/bin/env python
# Claudio Perez
# Fall 2022

import numpy as np
import sympy as sp
from functools import cached_property

from .shape import make_shapes
from sympy.printing.c import ccode as _ccode

ccode = lambda *args: _ccode(*args, standard='C89')


def spiral(n):
    """Generate natural node coordinates in a spiral."""
    dx,dy = 1,0            # Starting increments
    x,y = 0,0              # Starting location
    output = [[None]* n for j in range(n)]
    for i in range(n**2):

        yield (-1.+x*2./(n-1),  -1.+y*2./(n-1)), i

        output[x][y] = i + 1
        nx,ny = x+dx, y+dy
        if 0<=nx<n and 0<=ny<n and output[nx][ny] is None:
            x,y = nx,ny
        else:
            dx,dy = -dy,dx
            x,y = x+dx, y+dy

class Brick:
    def __init__(self, order=None, nodes=None):
        self.order = order
        self.n = self.order + 1
        self._shapes = None
        self._shape = None
        self._nodes = nodes
        self._deriv  = None
        self._derivs = None
        self._coord_symbols = sp.symbols("xi eta")

    @cached_property
    def inner(self)->dict:
        "Interior nodes"
        return {k: coord for k,coord in self.nodes.items()
                if 1.0 not in coord and -1.0 not in coord}
    @cached_property
    def outer(self)->dict:
        "Exterior nodes"
        return {k: coord for k,coord in self.nodes.items()
                if 1.0 in coord or -1.0 in coord}

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = {}
            for j,(xy,i) in enumerate(spiral(self.n)):
                if (0 < i < 4*(self.n-1)):
                    if not i%(self.n-1):
                        i //= self.n-1
                    else:
                        i += 3 - i//(self.n-1)
                self._nodes.update({i+1: xy})
        return self._nodes

    @property
    def basis(self):
        return {k: v for k,v in zip(self.nodes.keys(), self.shapef)}

    @property
    def shapef(self):
        if self._shape is None:
            self._shape = [
                eval(f"lambda xi, eta: {shape}")
                for shape in self.shapes().values()
            ]
        return [x for x in self._shape]

    @property
    def derivs(self):
        xi = self._coord_symbols
        if self._derivs is None:
            self._derivs = {
                k: [shape.diff(x) for x in xi]
                for k,shape in self.shapes().items()
            }
        return self._derivs

    @property
    def deriv(self):
        xi = self._coord_symbols
        if self._deriv is None:
            self._deriv = {
                k: [eval(f"lambda xi, eta: {shape.diff(x)}") for x in xi]
                for k,shape in self.shapes().items()
            }
        return self._deriv

    def coord(self, xi):
        return xi

    def interp(self, xi, vals):
        basis = self.basis
        ndf = len(next(iter(vals.items())))
        ans = np.zeros(ndf)
        for k,shape in basis.items():
            for df in range(ndf):
                ans[df] += shape(*xi)*vals[k][df]
        return ans


    def vandermonde(self):
        f = lambda x,y: [eval(t,dict(x=x,y=y)) for t in self.terms]
        return [f(*x) for x in self.nodes.values()]


    def imaps(self, nodes):
        "symbolic isoparametric map"
        shapes = self.shapes()
        X = 0.
        Y = 0.
        for k,shape in shapes.items():
            if k not in nodes:
                x,y = self.nodes[k]
            else:
                x,y = nodes[k]
            X += shape*x
            Y += shape*y
        return X,Y


    def plot_nodes(self, ax=None):
        import matplotlib.pyplot as plt
        # plt.style.use("steel")
        if ax is None:
            _, ax = plt.subplots()

    def plot(self, i):
        import matplotlib.pyplot as plt
        plt.style.use("steel")
        _,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
        nat = np.array(list(self.nodes.values())).T
        N = self.shapef[i]

        x = np.linspace(-1.0, 1.0,20)
        y = np.linspace(-1.0, 1.0,20)
        xx, yy = np.meshgrid(x,y)

        z = np.array([[N(xj,yj) for xj,yj in zip(xi,yi)] for xi, yi in zip(xx,yy) ])
        ax.plot_surface(xx,yy,z,alpha=0.5)
        boun = np.array(((-1,-1),(1,-1),(1,1),(-1,1),(-1,-1))).T
        ax.scatter(*nat, marker="s")
        ax.plot(*boun,'-s')
        plt.show()

    def shapes(self, symbols=False):
        if self._shapes is None:
            A = self.vandermonde()
            f = lambda x,y: [eval(t,dict(x=x,y=y)) for t in self.terms]
            self._shapes = make_shapes(A, f, keys=list(self.nodes.keys()))
        return self._shapes

#   def print_shape(self, latex=True):
#       A = self.vandermonde()
#       monos = lambda x,y: [eval(t,dict(x=x,y=y)) for t in self.terms]
#       return _stringify(A, monos, keys=list(self.nodes.keys()), latex=latex)

class ShapelessBrick(Brick):
    def shapes(self, symbols=False):
        xi, eta = sp.symbols("xi eta")
        if self._shapes is None:
            self._shapes = {i: sp.Function(f"N_{i}")(xi, eta) for i in self.nodes}
        return self._shapes

class Lagrange(Brick):
    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = {}
            for j,(xy,i) in enumerate(spiral(self.n)):
                if (0 < i < 4*(self.n-1)):
                    if not i%(self.n-1):
                        i //= self.n-1
                    else:
                        i += 3 - i//(self.n-1)

                self._nodes.update({i+1: xy})

        return self._nodes

    @property
    def terms(self):
        terms = []
        for i in range(self.order+1):
            for j in range(i):
                terms.append(f"x**{i}*y**{j}")

            terms.append(f"x**{i}*y**{i}")

            for j in reversed(range(i)):
                terms.append(f"x**{j}*y**{i}")
        return terms


class Legendre(Lagrange):
    "Q1 + Hierarchical Legendre Interpolation"
    def __init__(self, order, corder=1):
        super().__init__(corder)
        super().shapes()
        self.corder = corder
        p = self.order = order

        #xi, eta = sp.symbols("xi eta")
        _xi = sp.IndexedBase("xi", shape=(2,))
        xi, eta = _xi[0], _xi[1]

        self._shapes.update({
            4+p: self.legendres(p,xi)*self.legendres(p,eta)
            for p in range(self.corder, self.order)
        })

#   def print_shape(self, symbols=False, latex=True):
#       print("\n".join(ccode(i) for i in self._shapes.values()))

    def legendres(self, p, x):
        from math import factorial
        return ((x**2-1)**p).diff(x, p-1)/(factorial(p-1)*2**(p-1))

class Serendipity(Brick):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = {}
            for j,(xy,i) in enumerate(spiral(self.n)):
                if (0 < i < 4*(self.n-1)):
                    if not i%(self.n-1):
                        i //= self.n-1
                    else:
                        i += 3 - i//(self.n-1)
                else:
                    if i != 0:
                        break
                self._nodes.update({i+1: xy})
            if self.order <= 3:
                return self._nodes
            elif self.order == 4:
                self._nodes[17] = (0.0, 0.0)
            else:
                raise ValueError("Unsupported order {self.order} for Serendipity quadrilateral")
        return self._nodes

    @property
    def terms(self):
        if self.order == 3:
            return ["1",
                    "x", "y",
                    "x**2", "x*y", "y**2",
                    "x**3", "x**2*y", "y**2*x", "y**3",
                    "x**3*y", "x*y**3"]

        elif self.order == 2:
            return ["1",
                    "x", "y",
                    "x**2", "x*y", "y**2",
                    "x**2*y", "y**2*x"]

        elif self.order == 1:
            return ["1", "x", "y", "x*y"]

        else:
            raise Exception("Unimplemented order")

if __name__ == "__main__":

    order = 2

    terms = []
    for i in range(order+1):
        for j in range(i):
            for k in range(j):
                print(f"x**{i}*y**{j}*z**{k}")


            for k in reversed(range(j)):
                print(f"x**{k}*y**{j}*z**{i}")

        print(f"x**{i}*y**{i}")

        for j in reversed(range(i)):
           #print(f"x**{j}*y**{i}")
            for k in range(j):
                print(f"x**{i}*y**{j}")

            for k in reversed(range(j)):
                print(f"x**{i}*y**{j}")
    return terms
