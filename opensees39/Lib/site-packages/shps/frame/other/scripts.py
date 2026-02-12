# External libraries
import matplotlib.pyplot as plt
import numpy as np

# My functions
from quad import quad_points

## SECTION ANALYSIS
def epsi(y,epsa,kappa):  return epsa - y*kappa;
ei = epsi


def My_abd(k, yref, ymf, d, bf, tf, tw, fy, E):
    r"""
    Yield moment 
    
    M_u(\kappa) = -\int _c^{d-h}\left(f-Ek\left(x+h\right)\right)xdx 
                = -\frac{-2d^3Ek+2c^3Ek-Ekh^3+3d^2Ekh+3c^2Ekh+3d^2f-3c^2f+3fh^2-6dfh}{6}
    
    M_b = -\int _{-h}^{d-h}\left(f-Ek\left(x+h\right)\right)xdx 
        = -\frac{-2d^3Ek+3d^2Ekh+3d^2f-6dfh}{6}
    
    """
    h = d/2+yref
    c = (ymf-yref)-tf/2
    assert ymf + d/2 + tf/2 == d
    Mu =-(bf-tw)*((-6*d*fy*h + 3*(fy*h**2) - 3*c**2*fy + 3*(d**2*fy) \
                   + 3*(c**2*(E*(h*k))) + 3*(d**2*(E*(h*k))) - E*h**3*k \
                   + 2*(c**3*(E*k)) - 2*d**3*(E*k))/6)
    Mb =-    tw *(-6*d*fy*h + 3*(d**2*fy) + 3*(d**2*(E*(h*k))) - 2*d**3*(E*k))/6
    return Mu + Mb


def Ny_abd(k, yref, ymf, d, bf, tf, tw, fy,E):
    """

    N_b(\kappa,h) = b_b \int _{-h}^{d-h} f_y - E k \left(x+h\right)dx 
                  = b_b (d f_y-\kappa E\left(h d+\frac{\left(-h+d\right)^2}{2}
                    -\frac{h^2}{2}\right))

    N_u(\kappa,h) = b_u \int _c^{d-h} f_y - E k \left(x+h\right)dx 
                  = b_u (f_y\left(-h+d\right)-f_y c-\kappa E\left(h\left(-h+d\right)
                    -hc+\frac{\left(-h+d\right)^2}{2}-\frac{c^2}{2}\right))

    """
    h = d/2+yref
    c = (ymf-yref)-tf/2
    assert ymf + d/2 + tf/2 == d

    Nb =     tw *(d*fy - k*E*(-h**2/2 + d*h + (d - h)**2/2))
    Nu = (bf-tw)*(-k*E*(-c**2/2 + (d - h)**2/2 - c*h + h*(d - h)) - c*fy + fy*(d - h))
    return Nb + Nu


def _yield_points_T(yref, ymf, d, bf,  tw, fy, E,**kwds):
    n_abd = 2
    tf = 2*(d/2-ymf)
    
    kap = np.linspace(0., 2.*fy/E/d, n_abd)
    
    N = [Ny_abd(kap[i], yref, ymf, d, bf, tf, tw, fy, E) for i in range(n_abd)]
    M = [My_abd(kap[i], yref, ymf, d, bf, tf, tw, fy, E) for i in range(n_abd)]
    
    N.extend([-Ny_abd(kap[i], yref, ymf, d, bf, tf, tw, fy, E) for i in range(n_abd)])
    M.extend([-My_abd(kap[i], yref, ymf, d, bf, tf, tw, fy, E) for i in range(n_abd)])
    
    N.append(N[0])
    M.append(M[0])
    
    return M,N

def Sr(yi, dA, nIP, elems,**kwds):
    resp = elems['resp']
    State = elems['State']

    def ei(y,epsa,kappa): return epsa - y*kappa
    
    def srm(e,*state): # e = [ eps_a,  kappa ]
        Δϵ = [epsi(yi[i],*e) - state[i][0] for i in range(nIP)]
        state_n = [State(*resp(Δϵ[i], *state[i])) for i in range(nIP)]
        s = np.array([
            sum([state_n[i][2]*dA[i]       for i in range(nIP)]),
           -sum([state_n[i][2]*dA[i]*yi[i] for i in range(nIP)]) ])
        return s, state_n
    
    def ks(e,*state): # e = [ eps_a,  kappa ]
        Δϵ = [epsi(yi[i],*e) - state[i][0] for i in range(nIP)]
        
        st = [State(*resp(Δϵ[i], *state[i])) for i in range(nIP)]
        # st = state
        
        return np.array([
        [ sum(st[i].Et*dA[i] for i in range(nIP)),
         -sum(st[i].Et*yi[i]*dA[i] for i in range(nIP))],
        
        [-sum(st[i].Et*yi[i]*dA[i] for i in range(nIP)),
          sum(st[i].Et*yi[i]**2*dA[i] for i in range(nIP))] ]), st
    
    return srm,ks


def load_hist(s_ref,steps):
    diff = np.diff(s_ref,axis=0)
    s = np.array([n/stepi*diff[i]+s_ref[i] for i,stepi in enumerate(steps) for n in range(1,stepi+1)])
    return s

def Composite_Section(Y, DY, DZ, quad, y_shift = 0.0, MatData=None):
    nr = len(Y) # number of rectangles
    u,du = [],[]
    for i in range(nr):
        loc, wght = quad_points(**quad[i])
        u.append(loc)
        du.append(wght)

    nIPs = [len(ui) for ui in u]
    nIP = sum(nIPs)
    
    DU = [sum(du[i]) for i in range(nr)]
    
    yi = [float(    Y[i] + DY[i]/DU[i] * u[i][j]       )
            for i in range(nr) for j in range(nIPs[i])]

    dy = [float(    DY[i]/DU[i] * du[i][j]      )
            for i in range(nr) for j in range(nIPs[i])]
    dz = [DZ[i] for i in range(nr) for j in range(nIPs[i])]
    
    yi, dy, dz = map(list, zip( *sorted(zip(yi, dy, dz) )))

    dA = [ dy[i]*dz[i] for i in range(nIP)]

    Qm = np.array([[*dA], [-y*da for y,da in zip(yi,dA)]])

    ## Properties
    yrc = sum(y*dY*dZ for y,dY,dZ in zip(Y,DY,DZ))/sum(dY*dZ for dY,dZ in zip(DY,DZ))
    
    Izrq = sum(yi[i]**2*dA[i] for i in range(nIP)) # I   w.r.t  z @ yref using quadrature
    
    Izr =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i])**2 for i in range(nr))
    
    Izc =  sum(DZ[i]*DY[i]**3/12 + DZ[i]*DY[i]*(Y[i]+yrc)**2 for i in range(nr))
    
    A = sum(dA)
    ke = np.array([[A,-A*yrc], [-A*yrc, Izc]])

    SectData = {
        'nIP': nIP,'dA': dA, 'yi': yi,'Qm':Qm,
        "ke": ke,
        'yrc':yrc,
        'Izrq':Izrq,'Izr':Izr,'Izc':Izc,'MatData':MatData}
    
    return SectData

def I_Sect(b,d,alpha,beta,quad, yref=0.0, MatData=None):
    tf = (1 - alpha*d)*0.5
    bf = b 
    tw = beta*b
    
    Yref = -yref

    Y  = [Yref, 0.5*(d-tf) + Yref, -0.5*(d-tf) + Yref]
    
    DY = (tf,d-2*tf, tf)
    DZ = [bf, tw, bf]
    
    SectData = Composite_Section(Y,DY,DZ,quad, MatData=MatData)
    
    return SectData

def R_Sect(b, d, quad, yref=0.0,MatData=None):
    Y = [-yref]
    
    DY = [d]
    DZ = [b]
    
    SectData = Composite_Section(Y,DY,DZ,quad, MatData=MatData)

    # Properties
    A = b*d
    Z = 1/4*b*d**2
    I = 1/12*b*d**3
    S = I/d*2
    SectData['prop'] = dict(A=A,I=I,S=S,Z=Z)
        
    return SectData

def TC_Sect(d, bf, tw, quad, yref=0.0, tf=None, ymf=None, **kwds):
    if tf is None:
        tf = 2*(d/2-ymf)
    else:
        ymf = (d-tf)/2

    Yref = -yref
    Y  = [ Yref,  ymf + Yref, ymf + Yref]
    
    DY = [d, tf, tf]
    DZ = [ tw , (bf-tw)/2, (bf-tw)/2]

    SectData = Composite_Section(Y,DY,DZ,quad,MatData=kwds)
    SectData["d"] = d
    SectData["yref"] = yref
    
    return SectData

def plastic_points(Qm, fy, yref=0., norm=False, scale=None, **kwds):
    to_float = True
    nIP = len(Qm[0,:])
    
    f_n = [[fy*(-1)**(j>i) for i in range(1,nIP+1)] for j in range(1,1+nIP)] + \
          [[fy*(-1)**(j<i) for i in range(1,nIP+1)] for j in range(0,  nIP)]

    
    N, M = [], []
    for f in f_n:
        sri = Qm@f
        N.append(sri[0])
        M.append(sri[1])
    N.append(N[0])
    M.append(M[0])
    
    if norm:
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
        
        M[0 : nIP] = [-Mi / Mp_neg for Mi in M[0:nIP]]
        M[nIP : ] =  [ Mi / Mp_pos for Mi in M[nIP:]]
        
        N[0:nIP//2] = [Ni / Np_pos for Ni in N[0:nIP//2] ]
        N[nIP//2 : nIP*3//2+1] =  [-Ni / Np_neg for Ni in N[nIP//2 : nIP*3//2+1]]
        
        N[nIP*3//2 : ] = [ Ni / Np_pos  for Ni in N[nIP*3//2 : ]]
    
    if to_float: 
        N = [float(Ni) for Ni in N]
        M = [float(Mi) for Mi in M]
    return M,N


def plot_surface_T(yref, ymf, d, bf, tw, fy, E, quad, ax=None,mpl={}):
    
    tf = 2*(d/2-ymf)
        
    if ax is None: 
        fig, ax = plt.subplots()
    else:
        fig, ax = ax
    
    SectData = TC_Sect(d, bf, tw, quad, yref,ymf=ymf)

    Qm = SectData['Qm']
    
    Mp,Np = plastic_points(Qm, fy, yref=0., norm=False, scale=None)
    
    My,Ny = _yield_points_T(yref, ymf, d, bf, tw, fy, E)
    
    plastic_surface = ax.plot(Mp, Np, label='p')[0]
    yield_surface   = ax.plot(My, Ny,'r:',label='y')[0]
    
    return plastic_surface, yield_surface


def vary_strain(sect, dy, phi):
    epsy = sect["MatData"]["fy"]
    a = np.array([[1,0],[dy,1]])
    ke = a@sect["ke"]@a.T
    
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
    
    d = sect["d"]
    yr = sect["yrc"]
    # f = [e2 + (e1-e2)*(y+yr+d/2)/d for y in sect["yi"]]
    return e1, e2, ke@np.array([e2+(e1-e2)*(2*yr+d/2), (e1-e2)/d]) #, sect["Qm"]@f
    

def plot_strain(sect, dy, phi, ax):
    e1, e2, nm = vary_strain(sect, dy, phi)
    n, m = nm
    fy = sect["MatData"]["fy"]

    ax[0].set_title("Section Loci")
    ax[0].axhline(y=0, color='k')
    ax[0].axvline(x=0, color='k')

    ax[0].axhline(y= 24.0, color='grey',linewidth=1)
    ax[0].axhline(y=-24.0, color='grey',linewidth=1)
    
    ax[1].set_title("Stress Profile")
    ax[1].set_xlim([-fy, fy])
    return ax[1].plot([e1, e2], [1, 0])[0], ax[0].scatter(m, n)

