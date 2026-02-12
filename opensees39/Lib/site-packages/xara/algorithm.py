#===----------------------------------------------------------------------===//
#
#                                   xara
#                              https://xara.so
#
#===----------------------------------------------------------------------===//
#
# Copyright (c) 2025, OpenSees/Xara Developers
# All rights reserved.  No warranty, explicit or implicit, is provided.
#
# This source code is licensed under the BSD 2-Clause License.
# See LICENSE file or https://opensource.org/licenses/BSD-2-Clause
#===----------------------------------------------------------------------===//

class Newton:
    """Newton()
    Use the Newton-Raphson method for solving nonlinear equations.
    """
    def __init__(self):
        pass

    def _setup(self, model):
        model.algorithm("Newton")


class NewtonLineSearch:
    """
    Use the Newton-Raphson method with line search for solving nonlinear equations.
    
    Parameters
    ----------
    tolerance : float
        The line search tolerance. Default is 0.8.
    """
    def __init__(self, tolerance=0.8):
        self.tolerance = tolerance

    def _setup(self, model):
        model.algorithm("NewtonLineSearch", self.tolerance)

class ModifiedNewton:
    def __init__(self, initial=False):
        self.initial = initial

    def _setup(self, model):
        if self.initial:
            model.algorithm("ModifiedNewton", initial=True)
        else:
            model.algorithm("ModifiedNewton")


class BFGS:
    def __init__(self):
        pass

    def _setup(self, model):
        model.algorithm("BFGS")

