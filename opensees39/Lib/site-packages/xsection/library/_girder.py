#
#                                ^ y
#                                |
# _  |_______________________________________________________|
#    |_____  _______________ _________ _______________  _____|
#          \ \             | |       | |             / /
#           \ \            | |   |   | |            / /
#            \ \___________| |_______| |___________/ /
# _           \__________________+__________________/  ---> x
#             |                                     |


from xsection.polygon import PolygonSection
import numpy as np

class GirderSection(PolygonSection):
    def __init__(self,
        thickness_top  : float,
        thickness_bot  : float,
        height         : float,
        width_top      : float,
        width_webs     : list,
        web_spacing    : float,
        web_slope      : float = 0.0,
        overhang       : float = None
        ):
        self._thickness_top  = thickness_top
        self._thickness_bot  = thickness_bot
        self._height         = height
        self._width_top      = width_top
        self._width_webs     = width_webs
        self._web_spacing    = web_spacing
        self._web_slope      = web_slope
        self._overhang       = overhang

        mesh_size = (min(thickness_bot, thickness_top, *width_webs))**2

        t_top = self._thickness_top
        t_bot = self._thickness_bot
        H     = self._height
        Wt    = self._width_top
        tw    = self._width_webs
        sp    = self._web_spacing
        s     = self._web_slope
        oh    = self._overhang

        if len(tw) < 2:
            raise ValueError("need at least 2 webs")

        # clear height between flanges
        Ho = H - t_top
        Hi = Ho - t_bot
        # bottom‐flange width
        if oh is not None:
            Wb = Wt - 2*(oh + s*Ho)
        else:
            # fallback: make bottom flange same as top
            Wb = Wt

        # outer webs:
        x0 = -Wb/2 + tw[ 0]/2
        xN = +Wb/2 - tw[-1]/2

        # 1) exterior ring (ccw)
        exterior = np.array([
            # top flange
            (-Wt/2,         H),
            (+Wt/2,         H),
            (+Wt/2,         Ho),
            # Right web
            (xN + tw[-1]/2 + s*Ho,     Ho),  # down the right sloped web
            # (+Wb/2+t_bot*s,   t_bot),
            (+Wb/2,         0.0),
            (-Wb/2,         0.0),
            # (-Wb/2,         t_bot),
            (x0 - tw[ 0]/2 - s*Ho,     Ho),  # up the left sloped web
            (-Wt/2,         Ho),
        ])

        # compute web‐center x‐locations
        nweb = len(tw)
        if nweb > 2:
            # evenly spaced internal webs
            intern = np.linspace(x0 + sp, xN - sp, nweb-2)
            centers = np.hstack(([x0], intern, [xN]))
        else:
            centers = np.array([x0, xN])
        
        # 2) holes between webs
        interiors = []
        m = len(centers)-1
        for i in range(m):
            left  = centers[i]   + tw[i]/2
            right = centers[i+1] - tw[i+1]/2
            if right <= left:
                continue

            y0, y1 = t_bot, t_bot+Hi

            if i == 0:
                # leftmost gap: left side sloped
                hole = np.array([
                    ( left,         y0       ),
                    ( right,        y0       ),
                    ( right,        y1       ),
                    ( left - s*Hi,  y1       ),
                ], float)

            elif i == m-1:
                # rightmost gap: right side sloped
                hole = np.array([
                    ( left,         y0       ),
                    ( right,        y0       ),
                    ( right + s*Hi, y1       ),
                    ( left,         y1       ),
                ], float)

            else:
                # pure rectangle
                hole = np.array([
                    ( left, y0 ),
                    ( right, y0 ),
                    ( right, y1 ),
                    ( left, y1 ),
                ])

            interiors.append(hole)



        super().__init__(exterior, interiors, mesh_size=mesh_size)
    
