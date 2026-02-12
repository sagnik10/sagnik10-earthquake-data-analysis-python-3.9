#!/usr/bin/env python
# Claudio Perez
# Fall 2022

import numpy as np
import itertools
from math import sqrt
from collections import namedtuple

Basis = namedtuple("Basis", ("val", "tan"))

Quad   = [
#   ((0.0, 0.0), 1.0)
    ((-1/sqrt(3), -1/sqrt(3)), 1.0),
    (( 1/sqrt(3), -1/sqrt(3)), 1.0),
    (( 1/sqrt(3),  1/sqrt(3)), 1.0),
    ((-1/sqrt(3),  1/sqrt(3)), 1.0),
]


def tangent(xyz, u, dofs, k):
    nen = 4
    (xi,yi), _, (xj, yj), __ = xyz
    A = (yj - yi)*(xj - xi)
    shp = [
        #    |              val                 |                tan                     |
        Basis(lambda x,y:  1/A*(x - xj)*(y - yj), lambda x,y,i: [(y-yj)/A,  (x-xj)/A][i]),
        Basis(lambda x,y: -1/A*(x - xi)*(y - yj), lambda x,y,i: [(yj-y)/A,  (xi-x)/A][i]),
        Basis(lambda x,y:  1/A*(x - xi)*(y - yi), lambda x,y,i: [(y-yi)/A,  (x-xi)/A][i]),
        Basis(lambda x,y: -1/A*(x - xj)*(y - yi), lambda x,y,i: [(yi-y)/A,  (xj-x)/A][i])
    ]

    for x,w in Quad:
        ij = itertools.product(*[dofs]*2)
        x = xi+(xj-xi)*(x[0]+1)/2, yi+(yj-yi)*(x[1]+1)/2
        B = [[shp[i].tan(*x,j) for j in range(2)] for i in range(nen)]
        for i in range(nen):
            for j in range(nen):
                ii,jj = next(ij)
                k[ii,jj] += (B[i][0]*B[j][0] + B[i][1]*B[j][1])*(w*0.25*A)


def number(nodes, cells, boundary):
    ndf = 1
    nt = len(nodes)*ndf
    # Initialize counters
    free, fixed = 0, len(nodes)*ndf

    # dofs = np.arange(len(nodes)*ndf).reshape(len(nodes),ndf)
    # dofs = defaultdict(lambda: [None for i in range(ndf)])
    dofs = np.zeros((len(nodes), ndf), dtype=int) + fixed**2
    U,P = np.zeros((nt,2)).T
    for i,node in enumerate(nodes.values()):
        for j,u in enumerate(boundary(*node)):
            if u is not None:
                fixed -= 1
                U[fixed] = u
                dofs[i,j] = fixed
            else:
                # P[free] = 0.0
                dofs[i,j] = free
                free += 1

    return nodes, cells, dofs, P, U, free


def assemble(nodes, elems, dofs, P, U, free):
    ndm = 2
    ndf = 1
    nn = len(nodes)
    nt = nn*ndf
    K = np.zeros((nt, nt))
    for elem in elems.values():
        xyz = [nodes[i] for i in elem]
        df = dofs[np.array(elem)-1].flatten()
        tangent(xyz, None, df, K)

    return K, P, U, dofs, free, nodes, elems


def solve(K, P, U, dofs, free, *args, **kwds):
    n = free  #max(dofs[dofs > -1]) + 1
    U[:n] = np.linalg.solve(K[:n, :n], P[:n] - K[:n,n:]@U[n:])

    solution = U[dofs.flatten()]
    return solution, args


def block(n: int, L: float, H: float):
    x,y = np.linspace([0,0], [L,H], n+1).T
    nodes = {
        i+1: (xi, yi) for i,(xi,yi) in enumerate(itertools.product(x,y))
    }
    cells = {
        k: (1+j+i*(n+1), 1+n+1+j+i*(n+1), 1+n+2+j+i*(n+1), 2+j+i*(n+1))
        for k,(i,j) in enumerate(itertools.product(*[range(n)]*2))
    }
    return nodes, cells


if __name__ == "__main__":

    from test_heat import plot_analytic

    def boundary(x,y):
        L = 10.0
        if y == 0.0:
            return x*(L - x),
        elif x == 0.0 or x == L or y == L:
            return 0.0,
        else:
            return None,

    # nodes, elems, dofs, P, U, free = number(*block(3, 10, 10), boundary)
    # plt = Plotter()
    # plt.nodes({k: v for i,(k,v) in enumerate(nodes.items()) if dofs[i,0]<free})
    # plt.nodes({k: v for i,(k,v) in enumerate(nodes.items()) if dofs[i,0]>=free})
    # plt.show()

    # nodes, elems, dofs, P, U, free = number(*block(4, 10, 10), boundary)
    # plt = Plotter()
    # plt.nodes({k: v for i,(k,v) in enumerate(nodes.items()) if dofs[i,0]<free})
    # plt.nodes({k: v for i,(k,v) in enumerate(nodes.items()) if dofs[i,0]>=free})
    # plt.show()

    from sees.plane import render

    for n in 2,4, 8, 32: # 64:
        solution, (nodes, elems) = \
                solve(*assemble(*number(*block(n,10.,10.), boundary)))

        ax = render((nodes, elems), solution)
        plot_analytic(ax).set_title(fr"${n}\times{n}$")
        import matplotlib.pyplot as plt
        plt.show()
        #plt.gcf().savefig(f"p5-{n}.png")


