import numpy as np 

class Material: ... 


class _ElasticAnnulus:
    def __init__(self, section):
        self.J = 0.5 * section.area * section.outer_radius ** 2


class Annulus:
    _composite = False
    _is_fiber = True
    _is_shape = True
    _is_model = True

    """
    Create a circular patch.

    ## Examples

    >>> patch.circ()
    """
    _signature = [
        "radius", "sector", "divs", "quadrature", "material",
        "center"
    ]
    _img  = "circPatch.svg"
    _args = [
      Ref("material", type=Material, about="tag of previously defined material ("\
                          "`UniaxialMaterial` tag for a `FiberSection` "\
                          "or `NDMaterial` tag for use in an `NDFiberSection`)"),
      Grp("divs", args=[
          Int("circ", about="number of subdivisions (fibers) in "\
                            "the circumferential direction (number of wedges)"),
          Int("rad",  about="number of subdivisions (fibers) in the radial direction (number of rings)"),
      ]),
      Grp("center",    args=[Num("y"), Num("z")],
          about="$y$ & $z$-coordinates of the center of the circle", default=[0.0, 0.0]),
      Num("inner_radius",   about="internal radius", default=0.0),
      Num("outer_radius",   about="external radius"),
      Num("start_angle", about="starting angle", default=0.0),
      Num("endAng",   about="ending angle", default=np.pi*2),
    ]
    def __init__(self, inner_radius, outer_radius, 
                 mmaterial=None, *args, **kwds):

        self._moic = None
        self._moig = None
        self._area = None
        self._fibers = None
        if "diameter" in kwds:
            self.outer_radius = kwds["diameter"]/2

    def __contains__(self, point):
        origin = np.asarray(self.center)
        vect = np.asarray(point) - origin
        size = np.linalg.norm(vect)
        angl = (np.arctan2(*reversed(vect))+2*np.pi) % 2*np.pi
        inside = [
            (size < self.outer_radius),
            (size > self.inner_radius),
            (angl > self.start_angle),
            (angl < self.endAng),
        ]
        return all(inside)


    def create_fibers(self):
        """
        - https://stackoverflow.com/questions/33510979/generator-of-evenly-spaced-points-in-a-circle-in-python
        - https://mathworld.wolfram.com/CircleLatticePoints.html
        - Sunflower algorithm: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        - Sunflower alpha algorithm : https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
        - Partition of the circle in cells of equal area and shape: https://orbi.uliege.be/bitstream/2268/91953/1/masset_isocell_orbi.pdf
        """
        if self._fibers is None:
            if self.divs is None:
                self._fibers = []

            elif self.kwds.get("rule", "mid") == "mid":
                ri, ro = self.inner_radius, self.outer_radius
                dr = (self.outer_radius - self.inner_radius)/self.divs[1]
                dt = (self.endAng - self.start_angle)/self.divs[0]
                areas = (0.5*dt*((ri+(i+1)*dr)**2 - (ri+i*dr)**2) for i in range(self.divs[1]))
                self._fibers = [
                    _Fiber([r*np.cos(theta), r*np.sin(theta)], area, self.material)
                        for r,area in zip(np.linspace(self.inner_radius+dr/2, self.outer_radius-dr/2, self.divs[1]), areas)
                        #for theta in np.linspace(self.start_angle+dt/2, self.endAng-dt/2, self.divs[0])
                            for theta in np.arange(self.start_angle, self.endAng, dt)
                ]

            elif self.kwds.get("rule",None) == "uniform":
                self._fibers = [
                    _Fiber([r*np.cos(theta), r*np.sin(theta)], area, self.material)
                    for r,theta,area in _rtuniform(n=self.divs[1], rmax=self.outer_radius, m=self.divs[0], rmin=self.inner_radius)
                ]

            elif self.kwds.get("rule",None) == "uniform-2":
                self._fibers = [
                    _Fiber([r*np.cos(theta), r*np.sin(theta)], area, self.material)
                    for r, theta, area in _rtuniform(n=self.divs[1], rmax=self.outer_radius, m=self.divs[0])
                    if (r*np.cos(theta), r*np.sin(theta)) in self
                ]
            elif self.kwds.get("rule",None) == "uniform-3":
                self._fibers = [
                    _Fiber([r*np.cos(theta), r*np.sin(theta)], area, self.material)
                    for r, theta, area in _rtpairs(
                        np.linspace(self.inner_radius, self.outer_radius, self.divs[2]), 
                        np.arange(self.divs[1], self.divs[2]+self.divs[1]+1)*self.divs[0]
                    )
                    if (r*np.cos(theta), r*np.sin(theta)) in self
                ]
            elif self.kwds.get("rule",None) == "uniform-4":
                self._fibers = [
                    _Fiber([r*np.cos(theta), r*np.sin(theta)], area, self.material)
                    for r, theta, area in _rtpairs(
                        np.linspace(0.0, self.outer_radius, self.divs[2]),
                        np.arange(self.divs[1], self.divs[2]+self.divs[1]+1)*self.divs[0]
                    )
                    if (r*np.cos(theta), r*np.sin(theta)) in self
                ]

            elif self.kwds.get("rule",None) == "sunflower":
                area = self.area / self.divs[0]
                self._fibers = [
                    _Fiber([x,y], area, self.material)
                    for x,y in _sunflower(self.divs[0], self.outer_radius)
                    if (x,y) in self
                ]
            else:
                raise ValueError(f"Unknown quadrature rule, '{self.kwds['rule']}'.")
        return self._fibers

    @property
    def moic(self):
        if self._moic is None:
            r2,r1 = self.outer_radius, self.inner_radius
            a2,a1 = self.endAng, self.start_angle
            dsin  = np.sin(2*a2) - np.sin(2*a1)
            self._moic = np.array([
                [(a2-a1 - 0.5*dsin), 0],
                [0, (a2-a1 + 0.5*dsin)],
            ])*(r2**4 - r1**4)*0.125 + self.area*self.centroid**2
        return self._moic

    @property
    def area(self):
        r2,r1 = self.outer_radius, self.inner_radius
        a2,a1 = self.endAng, self.start_angle
        return 0.5*(r2**2 - r1**2)*(a2 - a1)

    @property
    def centroid(self):
        r2,r1 = self.outer_radius, self.inner_radius
        a2,a1 = self.endAng, self.start_angle
        return np.array([
            np.sin(a2) - np.sin(a1),
            - np.cos(a2) + np.cos(a1)
        ])*(r2**3 - r1**3)/(3*self.area)


    @property
    def ixc(self):
        return self.moic[0,0]

    @property
    def iyc(self):
        return self.moic[1,1]


def _rtpairs(R,N):
    """
    R - list of radii
    N - list of points per radius

    Takes two list arguments containing the desired radii
    and the number of equally spread points per radii respectively.
    The generator, when iterated, will return radius-angle polar
    coordinate pairs, in metres and radians, which can be used 
    to plot shapes, e.g. a disc in the x-y plane.
    """
    for i in range(len(R)-1):
        theta = 0.
        dTheta = 2*np.pi/N[i]
        for j in range(N[i]):
            theta = j*dTheta
            area = 0.5*dTheta*(R[i+1]**2 - R[i]**2)
            yield (R[i+1]+R[i])/2, theta, area


def _rtuniform(n,rmax,m,rmin=0.0):
    """
    n - number of radii
    rmax - maximum radius
    m - scaling of points with radius

    This generator will return a disc of radius rmax, 
    with equally spread out points within it. The number 
    of points within the disc depends on the n and m parameters.
    """
    if not isinstance(n,int):
        n0, n = n
    else:
        n0 = 1

    R = [rmin]
    N = [n0]
    rmax_f = float(rmax)
    for i in range(int(n)):
        ri = rmin + (i+1)*((rmax_f-rmin)/int(n))
        ni = int(m)*(i+1)
        R.append(ri)
        N.append(ni)

    return _rtpairs(R,N)


def sunflower(n, rad, alpha=0, geodesic=False):
    from math import sqrt, sin, cos, pi
    phi = (1 + sqrt(5)) / 2  # golden ratio
    def radius(k, n, b):
        if k > n - b:
            return 1.0
        else:
            return sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
    points = []
    angle_stride = 360 * phi if geodesic else 2 * pi / phi ** 2
    b = round(alpha * sqrt(n))  # number of boundary points
    for k in range(1, n + 1):
        r = rad*radius(k, n, b)
        theta = k * angle_stride
        points.append((r * cos(theta), r * sin(theta)))
    return points

