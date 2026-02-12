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
import jax
import inspect
from functools import wraps


def NodalValue(create_model, output, steps=1):
    node, dof = output

    from xara.para import _Parameter

    pnames = list(inspect.signature(create_model).parameters.keys())
    @jax.custom_vjp
    @wraps(create_model)
    def f(*params):
        m = create_model(*(
            _Parameter(p, index=i, name=n) for (i, p), n in zip(enumerate(params), pnames)
        ))
        m.analyze(steps)
        return m.nodeDisp(node, dof)

    # Forward
    def f_fwd(*params):
        m = create_model(*(
            _Parameter(p, index=i, name=n) for (i, p), n in zip(enumerate(params), pnames)
        ))
        m.analyze(steps)
        y = m.nodeDisp(node, dof)
        return y, m


    # Backward
    def f_bwd(m, cotangent):
        return tuple(
            m.sensNodeDisp(node, dof, tag) * cotangent
            for tag in m._model._openseespy._interp._tcl.call("getParamTags")
        )

    f.defvjp(f_fwd, f_bwd)
    return f

