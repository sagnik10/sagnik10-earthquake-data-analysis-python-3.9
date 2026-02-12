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

class Pattern:
    _persistent: bool

    def _activate(self, model):
        pass

    def _deactivate(self, model):
        tag = self.tag
        if self._persistent:
            model.loadConst(tag=tag)
        else:
            model.removePattern(tag=tag)


class AmbientAcceleration(Pattern):
    def __init__(self, dof, series, dt=None, time=None, persist=False, tag=None):
        self.tag = tag
        self._dof = dof
        if hasattr(series, "tolist"):
            series = series.tolist()
        self._data = series
        self._dt = dt
        # super().__init__(self, persist=persist)


    def _activate(self, model, pattern=None):
        if pattern is None:
            pattern = 1

        series_tag = 1 + 6*(pattern - 1)

        if pattern not in model._patterns:
            model.timeSeries("Path", series_tag,
                             dt=self._dt, values=self._data)
            model.pattern("UniformExcitation",
                          pattern, self._dof, accel=series_tag)
            series_tag += 1
            # model._patterns.append(pattern)
        else:
            model.activatePattern(tag=pattern)


