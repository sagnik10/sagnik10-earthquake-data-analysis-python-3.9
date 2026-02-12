import numpy as np

class Shape:
    """
    Base class for cross-section calculations.
    """
    def linspace(self, start, stop, num, endpoint=True, center=None, **kwds):
        """
        Create ``num`` copies of this section with centroids linearly aranged from ``start`` to ``stop``.
        """
        if center is None:
            for x in np.linspace(start, stop, num=num, endpoint=endpoint):
                yield self.translate(x-self.centroid, **kwds)
            return
        
        # else:
        #     radius = np.linalg.norm(center - self.centroid)
        #     for x in np.linspace(start, stop, num=num, endpoint=endpoint):
        #         loc = radius*np.sin()

        # ensure inputs are 2D float arrays
        start = np.asarray(start, dtype=float)
        stop  = np.asarray(stop,  dtype=float)
        current_centroid = np.asarray(self.centroid, dtype=float)

        if center is None:
            # straight-line interpolation of positions
            for pos in np.linspace(start, stop, num=num, endpoint=endpoint):
                delta = pos - current_centroid
                yield self.translate(delta, **kwds)

        else:
            center = np.asarray(center, dtype=float)

            r_start = np.linalg.norm(start - center)
            r_stop  = np.linalg.norm(stop  - center)
            if not np.isclose(r_start, r_stop):
                raise ValueError(
                    "start and stop must lie on the same circle around {}".format(center)
                )
            radius = 0.5 * (r_start + r_stop)

            angle0 = np.arctan2(start[1] - center[1],
                                start[0] - center[0])
            angle1 = np.arctan2(stop[1]  - center[1],
                                stop[0]  - center[0])

            if angle1 <= angle0:
                angle1 += 2 * np.pi

            so = self.translate(-start, **kwds)
            # interpolate angles
            for angle in np.linspace(angle0-angle0, angle1-angle0, num=num, endpoint=endpoint):
                pos   = center + radius * np.array([np.cos(angle), np.sin(angle)])
                delta = pos # - current_centroid
                yield so.rotate(angle)
