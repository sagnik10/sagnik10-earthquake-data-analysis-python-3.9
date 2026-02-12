import numpy as np
import sympy as sp
from sympy.printing.c import ccode as _ccode

ccode = lambda *args: _ccode(*args, standard='C89')

def stringify(shape, keys=None, sym=False, latex=True, xi=None):
    A    = shape.vandermonde()
    if hasattr(shape, "monomials"):
        terms = shape.monomials
    else:
        terms = lambda x,y: [eval(t,dict(x=x,y=y)) for t in shape.terms]

    if keys is None:
        keys = list(range(1, len(A)+1))

    if xi is None:
        if latex:
            xi = sp.symbols("xi eta")

        else:
            xi = sp.IndexedBase("xi", shape=(2,))

    output = ""
    for i,exp in enumerate(make_shapes(A, terms, keys=keys, sym=sym, xi=xi).values()):
#   for i in range(len(A)):
#       b[:] = 0.0
#       b[i] = 1.0
#       if Ainv is not None:
#           C = Ainv@b

#       else:
#           C = np.linalg.solve(A,b)


#       exp = sum(sp.nsimplify(c, rational=True, full=True, tolerance=1e-10)*t
#                 for c,t in zip(C, shape(x,y)))
        _coef, expr = sp.factor(exp).as_coeff_Mul()

#       _coef, expr = exp.as_coeff_Mul()

        if _coef == 1:
            coef = ""

        else:
            coef = sp.latex(_coef) if latex else ccode(_coef) + "*"

        if latex:
            output = "\n".join((output, f"N_{{{keys[i]}}} &= {coef} {sp.latex(_simplify(expr))} \\\\"))

        else:
            expr.replace(
                 lambda x: x.is_Pow and x.exp > 0,
                 lambda x: sp.Mul(*[x.base]*x.exp, evaluate=False))
            output = "\n".join((output, f"    shp[{i}][0] = {' '*bool(_coef>0)}{coef}{ccode((expr))};"))
            for j,x in enumerate(xi):
                output = "\n".join((output, f"    shp[{i}][{j+1}] = {' '*bool(_coef>0)}{coef}({ccode((expr.diff(x)))});"))
#           output = "\n".join((output, f"    shp[{i}][2] = {' '*bool(_coef>0)}{coef}({ccode((expr.diff(y)))});"))

    if latex:
        return f"$$\\begin{{aligned}}{output}\n\\end{{aligned}}$$"

    else:
        return output

def make_shapes(A, terms, keys=None, sym=False, xi=None):
    import sympy as sp

    if keys is None:
        keys = list(range(1, len(A)+1))

    if xi is None:
        xi = sp.symbols("xi eta")

    b = np.zeros(len(A))

    if sym:
        Ainv = sp.Matrix(A).inv()
    else:
        Ainv = None

    output = {}
    for i in range(len(A)):
        b[:] = 0.0
        b[i] = 1.0

        if Ainv is not None:
            C = Ainv@b

        else:
            C = np.linalg.solve(A,b)

        exp = sum(sp.nsimplify(c, rational=True, full=True, tolerance=1e-10)*t
                  for c,t in zip(C, terms(*xi)))
       #output.update({keys[i]: _simplify(sp.factor(exp))})
        output.update({keys[i]: sp.factor(exp)})

    return output


def _stringify(shape, keys=None, symbolic=False, latex=True):
    """
    """
    A    = shape.vandermonde()
    func = lambda x,y: [eval(t,dict(x=x,y=y)) for t in shape.terms]

    if keys is None:
        keys = list(shape.nodes.keys())
#       keys = list(range(1, len(A)+1))


    b = np.zeros(len(A))

    output = ""

    try:
        Ainv = np.linalg.inv(A)
        Ainv = None

    except:
        Ainv = sp.Matrix(A).inv()


    if latex:
        x,y = sp.symbols("xi eta")

    else:
        _xi = sp.IndexedBase("xi", shape=(2,))
        x,y = _xi[0], _xi[1]
        # x,y = sp.symbols(" ".join(f"xi_{i}" for i in range(2)))

    for i in range(len(A)):
        b[:] = 0.0
        b[i] = 1.0
        if Ainv is not None:
            C = Ainv@b

        else:
            C = np.linalg.solve(A,b)


        exp = sum(sp.nsimplify(c, rational=True, full=True, tolerance=1e-10)*t
                  for c,t in zip(C, func(x,y)))

        _coef, expr = sp.factor(exp).as_coeff_Mul()

        if _coef == 1:
            coef = ""

        else:
            coef = sp.latex(_coef) if latex else ccode(_coef) + "*"

        if latex:
            output = "\n".join((output, f"N_{{{keys[i]}}} &= {coef} {sp.latex(_simplify(expr))} \\\\"))

        else:
            expr.replace(
                 lambda x: x.is_Pow and x.exp > 0,
                 lambda x: sp.Mul(*[x.base]*x.exp, evaluate=False))
            output = "\n".join((output, f"    shp[{i}][0] = {' '*bool(_coef>0)}{coef}{ccode((expr))};"))
            output = "\n".join((output, f"    shp[{i}][1] = {' '*bool(_coef>0)}{coef}({ccode((expr.diff(x)))});"))
            output = "\n".join((output, f"    shp[{i}][2] = {' '*bool(_coef>0)}{coef}({ccode((expr.diff(y)))});"))

    if latex:
        return f"$$\\begin{{aligned}}{output}\n\\end{{aligned}}$$"

    else:
        return output

def _simplify(shape, variables=None):
    import sympy as sp

    if variables is None:
        variables = "xi eta"

    variables = sp.symbols(variables)

    linear_sums = {}
    simple_terms = []

    for term in shape.args:
        if term.is_Add and len(term.free_symbols) == 1:
            var = next(iter(term.free_symbols))
            if var in variables and var in linear_sums:
                # print(term, var) 
                for i,ls in enumerate(linear_sums[var]):
                    c, x = ls.args
                    if ((x+c)*term).simplify() == (x**2 - c**2):
                        linear_sums[var][i] = x**2 - c**2
                        break
                else:
                    linear_sums[var].append(term)
                    continue

            elif var in variables:
                linear_sums[var] = [term]
        else:
            simple_terms.append(term)

    out = sp.core.Mul(
        *(
            *simple_terms,
            *[sp.core.Mul(*x, evaluate=False) for x in linear_sums.values()]
        ),
        evaluate=False
    )

    assert sp.simplify(out - shape) == 0, (out, shape)
    return out


