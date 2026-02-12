import numpy as np

#
# Interpolation
#
def _Interpolant(docs, fwd, grad):
    fwd.grad = grad
    return fwd


Q4 = _Interpolant("",
    lambda r, s: 0.25*np.array([
             (r-1)*(s-1), 
            -(r+1)*(s-1), 
             (r+1)*(s+1), 
            -(r-1)*(s+1)
    ]),
    lambda r, s: np.array([
            [s-1, -(s-1), (s+1), -(s+1)],
            [r-1, -(r+1), (r+1), -(r-1)]
    ])
)

T6 = _Interpolant(
    """
    Quadratic Lagrange polynomial interpolation over a triangle.
    """,
    lambda r,s: np.array([
             (1 - r - s) - 2*r*(1 - r - s) - 2*s*(1 - r - s),
             r - 2*r*(1 - r - s) - 2*r*s,
             s - 2*r*s - 2*s*(1-r-s),
             4*r*(1 - r - s), #6
             4*r*s,
             4*s*(1 - r - s )
        ]),

    lambda r,s: np.array([
            [4*r + 4*s - 3, 4*r - 1,       0, -8*r - 4*s + 4, 4*s,           -4*s],
            [4*r + 4*s - 3,       0, 4*s - 1,           -4*r, 4*r, -4*r - 8*s + 4]])
)

