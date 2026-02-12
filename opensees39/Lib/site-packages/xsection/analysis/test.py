import numpy as np
import math
from collections import defaultdict

def compute_Q0_T6_exact():
    """
    Computes the reference tensors for a 6-node quadratic triangle (T6) using
    exact barycentric integration.

    This avoids numerical quadrature by representing all shape functions as
    polynomials of the area coordinates (L0, L1, L2) and integrating them
    term-by-term using the exact formula:
    integral(L0^a * L1^b * L2^c) = a! * b! * c! / (a + b + c + 2)!

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two 6x6x6 tensors,
                                       (Q0_xi, Q0_eta).
    """

    # --- 1. Define Shape Functions as Polynomials ---
    # A polynomial is a list of tuples: [(coefficient, (p0, p1, p2)), ...],
    # where p0, p1, p2 are the powers of L0, L1, L2.
    
    # Node ordering: Vertices 0,1,2; Mid-nodes 3(0-1), 4(1-2), 5(2-0)
    N_poly = [
        [(2.0, (2, 0, 0)), (-1.0, (1, 0, 0))],  # N0 = 2*L0^2 - L0
        [(2.0, (0, 2, 0)), (-1.0, (0, 1, 0))],  # N1 = 2*L1^2 - L1
        [(2.0, (0, 0, 2)), (-1.0, (0, 0, 1))],  # N2 = 2*L2^2 - L2
        [(4.0, (1, 1, 0))],                      # N3 = 4*L0*L1
        [(4.0, (0, 1, 1))],                      # N4 = 4*L1*L2
        [(4.0, (1, 0, 1))],                      # N5 = 4*L2*L0
    ]

    # Derivatives of shape functions with respect to L0, L1, L2
    dN_poly_dL = [
        # dN0/dL0, dN0/dL1, dN0/dL2
        [[(4.0, (1, 0, 0)), (-1.0, (0, 0, 0))], [], []],
        # dN1/dL0, dN1/dL1, dN1/dL2
        [[], [(4.0, (0, 1, 0)), (-1.0, (0, 0, 0))], []],
        # dN2/dL0, dN2/dL1, dN2/dL2
        [[], [], [(4.0, (0, 0, 1)), (-1.0, (0, 0, 0))]],
        # dN3/dL0, dN3/dL1, dN3/dL2
        [[(4.0, (0, 1, 0))], [(4.0, (1, 0, 0))], []],
        # dN4/dL0, dN4/dL1, dN4/dL2
        [[], [(4.0, (0, 0, 1))], [(4.0, (0, 1, 0))]],
        # dN5/dL0, dN5/dL1, dN5/dL2
        [[(4.0, (0, 0, 1))], [], [(4.0, (1, 0, 0))]],
    ]

    # --- 2. Helper Functions for Polynomial Math ---

    def poly_mult(poly1, poly2):
        """Multiplies two polynomials in our defined format."""
        # Use a dictionary to automatically handle combining like terms
        result_dict = defaultdict(float)
        if not poly1 or not poly2:
            return []
        for c1, p1 in poly1:
            for c2, p2 in poly2:
                new_coeff = c1 * c2
                new_powers = (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])
                result_dict[new_powers] += new_coeff
        # Convert back to list of tuples format
        return [(coeff, powers) for powers, coeff in result_dict.items()]

    def exact_integral(powers):
        """Computes integral of L0^a * L1^b * L2^c over reference element."""
        a, b, c = powers
        return math.factorial(a) * math.factorial(b) * math.factorial(c) / \
               math.factorial(a + b + c + 2)

    # --- 3. Compute the Base Tensors (derivatives w.r.t. L0, L1, L2) ---
    
    Q0_L = [np.zeros((6, 6, 6)) for _ in range(3)] # Q0_L0, Q0_L1, Q0_L2

    for j in range(6):  # Index for the derivative term dN/dL
        for k in range(6):
            for l in range(6):
                # Get the polynomial for N_k * N_l
                Nk_Nl_poly = poly_mult(N_poly[k], N_poly[l])
                
                for m in range(3): # Corresponds to d/dLm
                    # Get the polynomial for the derivative d(N_j)/d(L_m)
                    dN_poly = dN_poly_dL[j][m]
                    if not dN_poly:
                        continue # Derivative is zero, so integral is zero
                    
                    # Form the full integrand
                    integrand_poly = poly_mult(dN_poly, Nk_Nl_poly)

                    # Integrate term by term
                    total_integral = sum(
                        coeff * exact_integral(powers)
                        for coeff, powers in integrand_poly
                    )
                    Q0_L[m][j, k, l] = total_integral

    return tuple(Q0_L)

def _I0(a, b, c):
    return 2.0 * math.factorial(a)*math.factorial(b)*math.factorial(c) / math.factorial(a+b+c+2)

def v2():
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

if __name__ == "__main__":
    Q0_T6_exact = compute_Q0_T6_exact()
    print("Q0_L0:\n", Q0_T6_exact[0])
    print("Q0_L1:\n", Q0_T6_exact[1])
    print("Q0_L2:\n", Q0_T6_exact[2])

    print("\n\n--- Using v2() ---\n")
    Q0_T6_exact = v2()
    print("Q0_L0:\n", Q0_T6_exact[0])
    print("Q0_L1:\n", Q0_T6_exact[1])
    print("Q0_L2:\n", Q0_T6_exact[2])