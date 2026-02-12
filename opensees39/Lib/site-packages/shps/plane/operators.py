class Operator:
    pass

class Identity(Operator):
    # used to form mass for transient terms
    pass

class Tangent(Operator):
    pass

class Gradient(Operator):
    def tan(self):
        pass

    def val(self):
        pass

class SymmetricGradient(Operator):
    pass


class Divergence(Operator):
    def compose(self, other: Operator):
        pass

    def tan(self):
        pass

    def val(self):
        pass



Laplacian = lambda ndm: Divergence(Gradient(ndm))


class BubnovGalerkin:
    def __init__(self, operator, shape):
        pass

    def stiffness(self, xyz, u=None, dofs=None, K=None):
        operator = self.operator

        for basis in self.shape:
            Ba = operator(basis)

if __name__ == "__main__":
#   element = Divergence(Gradient(LagrangeQ4))

    element = BubnovGalerkin(Laplacian(ndm=2), LagrangeQ4, GaussLegendre(4))


    dyn_ele = IdentityBGT3(c=rho)
    element = LaplacianBGT3()
    element = LaplacianBGQ4()


    u  = solve(mesh, element, load)

    #         Laplacian(LagrangeQ4)


