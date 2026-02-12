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
try:
    from opensees.openseespy import Model
except:
    Model = None

successful = 0


from ._analysis import StaticAnalysis, EigenAnalysis, DynamicAnalysis
from .load import NodalLoad

try:
    from jax import tree_util
#   import opensees.openseespy as ops
# 1) Tell JAX that ops.Model is an opaque leaf
    def _flatten_model(model):
        # no children, carry the model object in aux_data
        return (), model

    def _unflatten_model(aux_data, _children):
        # aux_data *is* the original model
        return aux_data

    tree_util.register_pytree_node(Model, _flatten_model, _unflatten_model)
except:
    pass



def solve(model, loading,
          output=None,
          analysis=None,
          algorithm=None, scale=1.0,
          test=None):
    """
    Solve a static equilibrium problem for the model with the given loading.

    Parameters
    ----------
    model : :py:class:`Model`
        The structural model to analyze.
    loading : Load or list of Loads
        The loads to apply during the analysis.
    algorithm : Algorithm, optional
        The solution algorithm to use. If None, the :ref:`Newton` algorithm is used.
    constraints : ConstraintHandler, optional
        The constraint handler to manage boundary conditions. Default is :ref:`"Plain" <PlainConstraints>`.
    """
    analysis_ = StaticAnalysis(loading, algorithm=algorithm, test=test)
    if analysis_.analyze(model) != successful:
        message = "Analysis did not complete successfully."
        message = model._openseespy._interp.read_error()
        raise RuntimeError(message)

    if output is not None:
        node, dof = output
        return model.nodeDisp(node, dof)
    return None


def eigen(model, modes, solver=None, output=None, problem="Frequency"):
    """
    Perform eigenvalue analysis on the model.

    Parameters
    ----------
    model : :py:class:`Model`
        The structural model to analyze.
    modes : int
        The number of eigenmodes to compute.
    solver : EigenSolver, optional
        The eigenvalue solver to use. If None, a default solver is used.
    problem : str, optional
        The type of eigenvalue problem to solve. Options are "Frequency" or "Buckling".

    Returns
    -------
    eigenvalues : list
        The computed eigenvalues.
    """
    model.eigen(modes)


def trace(model, loading, output=None, algorithm=None, method=None):
    """
    Perform a static analysis while tracing the response of the model.
    """
    analysis = StaticAnalysis(loading, algorithm=algorithm)
    if analysis.analyze(model) != successful:
        raise RuntimeError("Analysis did not complete successfully.")

    if output is not None:
        node, dof = output
        return model.nodeDisp(node, dof)
    return None



def integrate(model, loading, dt, steps, output=None, algorithm=None, method=None):
    """
    integrate the dynamic response of the model.
    """
    if output is None:
        output = {"displacement": model.getNodeTags()}
        output["velocity"] = output["displacement"]
        output["acceleration"] = output["displacement"]

    analysis = DynamicAnalysis(loading, dt, algorithm=algorithm)

    analysis._start_equilibrium(model)
    from collections import defaultdict
    result = defaultdict(list)
    for step in range(steps):
        if analysis.analyze(model, dt, 1) != successful:
            message = model._openseespy._interp.read_error()
            raise RuntimeError(f"Analysis did not complete successfully: {message}")
        if output is not None:
            for key, nodes in output.items():
                if key == "displacement":
                    for node in nodes:
                        result["U"].append(model.nodeDisp(node))
                elif key == "acceleration":
                    for node in nodes:
                        result["A"].append(model.nodeAccel(node))
                elif key == "velocity":
                    for node in nodes:
                        result["V"].append(model.nodeVel(node))
    return dict(result)


class Integrator:
    pass

class _LoadControl:
    pass

class ArcLengthControl(_LoadControl):
    pass

class DisplacementControl(_LoadControl):
    pass

class EnergyControl(_LoadControl):
    pass

class LoadControl(_LoadControl):
    pass
