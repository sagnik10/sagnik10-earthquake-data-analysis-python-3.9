import numpy as np
from typing import Union
import numpy as np

class Section: pass
class Mesh: pass

def mesh_y(sect: Section,
           quad: Union[int,dict,list]=None):
    Z,Y,DZ,DY = zip(*(
        [*p.centroid,*(np.subtract(p.vertices[2],p.vertices[0]))]
         for p in sect.patches
        )
    )
    nr = len(Y) # number of rectangles
 
    if isinstance(quad, dict): quad = [quad]*nr
    u,du = zip(*(quad_points(**q) for q in quad))
    nIPs = [len(ui) for ui in u]
    nIP = sum(nIPs)
    
    # Centroid
    A  = sum(dY*dZ for dY,dZ in zip(DY,DZ))
    yc = sum(y*dY*dZ for y,dY,dZ in zip(Y,DY,DZ))/A

    DU = [sum(dui) for dui in du]
    
    yi = [float(Y[i]-yc + DY[i]/DU[i] * u[i][j])
            for i in range(nr) for j in range(nIPs[i])]

    dy = [float(DY[i]/DU[i] * du[i][j])
            for i in range(nr) for j in range(nIPs[i])]

    zi = [ Z[i] for i in range(nr) for j in range(nIPs[i])]
    
    dz = [DZ[i] for i in range(nr) for j in range(nIPs[i])]
    
    # reorder points
    y, z, dy, dz = map(list, zip(*sorted(zip(yi, zi, dy, dz))))

    da = [ dy[i]*dz[i] for i in range(nIP)]

    
    Izr =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i])**2 
               for i in range(nr))
    
    Izc =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i]-yc)**2 
               for i in range(nr))
    
    A = sum(da)
    ke = np.array([[A,0.0], [0.0, Izc]])

    SectData = {
        'da': da, 'y': y, 'z': z, 'x': z,
        # 'Qm':Qm, 
        "ke": ke,
        'yc':yc,
        "shape": nIP,
        "bounds": sees2shapely(sect).bounds,
        'Izr':Izr,'Izc':Izc}
    
    return SectData


def _yield_point(sect: Union[Section, Mesh], 
                phi:  float, 
                dy:   float,
                bounds: tuple
    ):
    epsy = 1.0
    a = np.array([[1, 0],[dy, 1]])
    if isinstance(sect,dict):
        ke = a@sect["ke"]@a.T
        d = bounds[3] - bounds[1]
        # distance from bottom fiber to centroid.
        # yc is dist to centroid from user's origin
        yr = sect["yc"] - bounds[1]
        
    else:
        ke = np.array([[sect.area, 0.],
                       [0.0, sect.ixc]])
        d = bounds[3] - bounds[1]
        yr = sect.centroid[1] - bounds[1]
    
    if phi > 0.75:
        e1 = -epsy
        e2 = -epsy + 2*epsy*(phi-0.75)/0.25    
    elif phi > 0.50:
        e2 = -epsy
        e1 =  epsy - 2*epsy*(phi-0.50)/0.25    
    elif phi > 0.25:
        e1 =  epsy
        e2 =  epsy - 2*epsy*(phi-0.25)/0.25
    else:
        e2 =  epsy
        e1 = -epsy + 2*epsy*phi/0.25
    
    kap = -(e1-e2)/d
    return e1, e2, ke@np.array([e2-kap*yr, kap])


def elastic_limit(section: Section,
                  bounds,
                  shift=0.0):
    yc   = section.centroid[1]
    area = section.area
    d = bounds[3] - bounds[1]
    # distance from bottom fiber to centroid.
    # yc is dist to centroid from user's origin
    yr = yc - bounds[1]

    n1 = area
    m1 = -area*shift

    kap2 = 2/d
    eps2 = (1 - kap2*yr)
    n2 = area*eps2 # - 
    m2 = (section.ixc + shift**2*area)*kap2

    return np.array([
        [m1,m2,-m1,-m2,m1],[n1,n2,-n1,-n2,n1]
    ])#/np.array([[section.ixc],[area]])


def plastic_profile(n, psi):
    i = int(psi/n > 1.0)
    a,c = (-1)**i, (-1)**(not i)
    y = abs(psi/n - 1)
    return np.array([[a,a,c,c],[0,y,y+1/n,1]])

def plastic_profiles(nIP, fy=1.0):   
    return    [[fy*(-1)**(j>i) for i in range(1,nIP+1)] for j in range(1,1+nIP)] + \
              [[fy*(-1)**(j<i) for i in range(1,nIP+1)] for j in range(0,  nIP)]

def plastic_limit(sect, y, da, fy=1., fp = None, yref=0., norm=False, scale=None, **kwds):
    to_float = True
    
    Qm = np.array([[*da], [-y*a for y,a in zip(y,da)]])
    nIP = len(Qm[0,:])
    f_n = fp or plastic_profiles(nIP, fy)
    
    N,M = map(list,zip(*[Qm@f for f in f_n]))
    N.append(N[0])
    M.append(M[0])
    
    if norm:
        # return np.array([M,N])/np.array([[sect.ixc],[sect.area]])
        if scale is None:
            Np_pos = N[0]
            Np_neg = N[nIP]

            Mp_pos = M[nIP*3//2]
            Mp_neg = M[nIP//2]
        else:
            Np_pos = scale[1]
            Np_neg = -Np_pos
            Mp_pos = scale[0]
            Mp_neg = -Mp_pos
        
        M[ : nIP] =  [-Mi / Mp_neg for Mi in M[:nIP]]
        M[nIP : ] =  [ Mi / Mp_pos for Mi in M[nIP:]]
        
        N[:nIP//2] = [Ni / Np_pos for Ni in N[0:nIP//2] ]
        N[nIP//2 : nIP*3//2+1] =  [-Ni/Np_neg for Ni in N[nIP//2 : nIP*3//2+1]]
        
        N[nIP*3//2 : ] = [ Ni / Np_pos  for Ni in N[nIP*3//2 : ]]
    
    if to_float: 
        N = [float(Ni) for Ni in N]
        M = [float(Mi) for Mi in M]
    return M,N

class PlasticLocus:
    def __init__(self, sect, **kwds):
        super().__init__(**kwds)
        self.sect = sect
        fibers = list(sect.fibers())
        mesh = dict(
            x  = np.array([fiber.location[0] for fiber in fibers]),
            y  = np.array([fiber.location[1] for fiber in fibers]),
            da = np.array([fiber.area for fiber in fibers])
        )
        self.mesh = mesh
        self._mesh_dirty = False

        try:
            bounds = mesh["bounds"]
        except:
            bounds = min(mesh["x"]), min(mesh["y"]), max(mesh["x"]), max(mesh["y"])
        self.d = bounds[3] - bounds[1]
        self.bounds = bounds
        self.yield_prof = None
        self.plast_prof = None

    @staticmethod
    def plastic_profiles(nIP, fy=1.0):
        return [[fy*(-1)**(j>i) for i in range(1,nIP+1)] for j in range(1,1+nIP)] + \
               [[fy*(-1)**(j<i) for i in range(1,nIP+1)] for j in range(0,  nIP)]

    @staticmethod
    def elastic_limit(section, bounds, shift=0.0):
        zc   = section.torsion.centroid()[1]
        area = section.torsion.model.cell_area()

        d = bounds[3] - bounds[1]
        # distance from bottom fiber to centroid.
        # yc is dist to centroid from user's origin
        yr = zc - bounds[1]

        n1 = area
        m1 = -area*shift

        kap2 = 2/d
        eps2 = (1 - kap2*yr)
        n2 = area*eps2 # - 
        m2 = (section.torsion.cmm()[1,1] + shift**2*area)*kap2

        return np.array([
            [m1,m2,-m1,-m2,m1],[n1,n2,-n1,-n2,n1]
        ])#/np.array([[section.ixc],[area]])

    @staticmethod
    def plastic_profile(n, psi):
        i = int(psi/n > 1.0)
        a,c = (-1)**i, (-1)**(not i)
        y = abs(psi/n - 1)
        return np.array([[a,a,c,c],[0,y,y+1/n,1]])


    @staticmethod
    def plastic_limit(sect, y, da, fy=1., fp = None, scale=None, **kwds):
        Qm = np.array([
            [*da], [-y*a for y,a in zip(y,da)]
        ])
        nIP = len(Qm[0,:])
        f_n = fp or PlasticLocus.plastic_profiles(nIP, fy)

        N,M = map(list,zip(*[Qm@f for f in f_n]))
        N.append(N[0])
        M.append(M[0])
        return M,N

    def plot_plast(self, ax, nip, ip=1):
        d = self.d
        x,y = self.plastic_profile(nip, ip)
        ax.fill_betweenx(y*d, x, [0]*4, alpha=0.2, color="b")
        if self.yield_prof is None:
            self.plast_point = None
        return x,y

    def plot_yield(self, ax, phi, dy):
        d = self.d
        f1, f2, NM = _yield_point(self.sect, phi, dy, bounds=self.bounds)
        ax.fill_betweenx([0, d], [f2, f1], [0, 0], alpha=0.2, color="red")
        if self.yield_prof is None:
            self.yield_prof = ax.plot([f2, f1], [0, d], "r")[0]
        else:
            self.yield_prof.set_xdata([f2, f1])
        return NM

    def plot(self, phi=0.3, ip=3, fy=1.0):
        import matplotlib.pyplot as plt

        sect, mesh = self.sect, self.mesh
        nip = len(mesh["y"])
        ax = [None, None, None]
        fig = plt.figure()
        grid = fig.add_gridspec(nrows = 1, ncols = 3, 
          hspace = 0, wspace = 0, width_ratios = [6, 2, 4], 
          height_ratios = [6]
        )
        ax[0] = fig.add_subplot(grid[0,0])
        ax[1] = fig.add_subplot(grid[0,1])#, sharey = ax[0])
        ax[2] = fig.add_subplot(grid[0,2])

        # SECTION GEOMETRY
        ax[0].set_autoscale_on(True)
        # veux.section(sect, ax=ax[0], fix_axes=False)
        yc = sect.torsion.centroid()[1]
        ax[0].scatter(mesh["x"], mesh["y"]+yc, marker=".", color="b", alpha=0.5, s=0.3)

        ax[0].set_title("Geometry")
        ax[0].set_xlabel("$z$")
        ax[0].set_ylabel("$y$")


        # STRAIN PROFILE
        f,_ = self.plot_plast(ax[1], nip, ip=1)
        NM = self.plot_yield(ax[1], phi, 0.0)
        ax[1].set_xlabel(r"$\frac{\sigma}{f_y}$")
        ax[1].set_title("Stress")
        ax[1].set_yticks([], [])

        # SURFACE PLOTS
        #NM  = self.plot_yield(ax[1], phi,  0.0)
        #self.point = ax[2].scatter(NM[1], NM[0], marker="x", color="r")
        # self.plast_point = ax[2].scatter(PP[1], PP[0], marker="o", color="k")
        # PP = self.mesh["Qm"]@f
        ax[2].plot(*self.elastic_limit(sect, self.bounds), "r")
        ax[2].plot(*self.plastic_limit(sect, **mesh, norm=False), "b")
        ax[2].yaxis.tick_right()
        ax[2].yaxis.set_label_position("right")
        ax[2].set_xlabel("M")
        ax[2].set_ylabel("N")
        ax[2].axvline(0.0)
        ax[2].axhline(0.0)
        ax[2].grid("on")

        def update(yref=4.0, phi=(0.,1,0.05), ip=(0,2*nip,1)):
            for item in list(ax[1].collections):
                item.remove()
            # ax[1].collections.clear()
            # YP = self.plot_yield(ax[1], phi,0.0, bounds=self.bounds)
            f,y = self.plot_plast(ax[1], nip, ip)
            # PP = self.mesh["Qm"]@f
            # self.point.set_offsets([YP[1], YP[0]])
            # self.plast_point.set_offsets([PP[1], PP[0]])
            fig.canvas.draw_idle()
        return update

