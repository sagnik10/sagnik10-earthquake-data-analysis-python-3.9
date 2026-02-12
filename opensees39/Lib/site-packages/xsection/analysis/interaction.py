import xara
import numpy as np
# test_stress = lambda s: max(s) < 0.9*50*ksi
# test_stress = lambda s: min(s) > -0.9*50*ksi
# test_strain = lambda s: max(s) < 0.9*0.003
# test_strain = lambda s: min(s) > -0.9*0.003

class _SectionInterface:
    def __init__(self, section, shape, materials):

        if isinstance(materials, dict):
            materials = [materials]

        self._model = None
        self._materials = materials
        self._section = section
        self._shape = shape

        self._is_shear = "shear" in self._section.lower()


    def initialize(self):
        if self._model is not None:
            self._model.wipe()

        self._model = _create_model(self._section, self._shape, self._materials)
        model = self._model

        self._model.invoke("section", 1, ["update 0 0 0 0 0 0;"])


    def getStressResultant(self, e, commit=True):
        eps, kap = map(float, e)
        stress = self._model.invoke("section", 1, [
                        f"update  {eps} 0 0 0 0 {kap};",
                         "stress"
        ] + (["commit"] if commit else []))
        if self._is_shear:
            return np.array(stress)[[0, 5]]
        else:
            return np.array(stress)[[0, 3]]


    def getSectionTangent(self):
        tangent = self._model.invoke("section", 1, [
                        "tangent"
        ])

        n = int(np.sqrt(len(tangent)))
        Ks = np.array(tangent).reshape(n,n)
        return Ks


def _solve_eps(sect, kap, axial: float, eps0, tol=1e-8, maxiter=15, time=0.0):
    # Newton-Raphson iteration
    eps = eps0
    s = sect.getStressResultant([eps, kap], False)
    for i in range(maxiter):
        if abs(s[0] - axial) < tol:
            return eps
        s = sect.getStressResultant([eps, kap], False)
        eps -= (s[0] - axial)/sect.getSectionTangent()[0,0]
    
    print(f"Warning: {maxiter} iterations reached, r = {s[0] - axial}, {time = }")
    return None

    return eps


def _analyze(s, P, dkap, nstep):
    s.initialize()
    k0 = 0.0

    kap = 0
    if eo := _solve_eps(s,  k0,  P,  0.0):
        e0 = _solve_eps(s,  k0,  P,  eo)
    else:
        e0 = 0.0

    PM = [
        s.getStressResultant([e0, k0], True)
    ]
    if e1 := _solve_eps(s, k0+dkap, P, e0):
        PM.append(s.getStressResultant([e1, k0+dkap], True))

        e = e0
        kap = 2*dkap
        for _ in range(nstep):
            # if abs(PM[-1][1]) < 0.995*abs(PM[-2][1]):
            #     break
            e = _solve_eps(s, kap, P, e)
            if e is None:
                break
            PM.append(s.getStressResultant([e, kap], True))
            kap += dkap
    return PM, kap




def _create_model(section, shape, materials, boundary=None):

    model = xara.Model(ndm=3, ndf=6)

    if boundary is None:
        boundary = (0, 1, 1,  1, 1, 0)

    if isinstance(materials, dict):
        materials = [materials]

    shear = "shear" in section.lower()
    for i,mat in enumerate(materials):
        if shear:
            m = mat
            model.nDMaterial(m["type"], i+1, **{k: v for k, v in m.items() if k not in {"name", "type"}})
        else:
            m = mat
            model.uniaxialMaterial(m["type"], i+1, **{k: v for k, v in m.items() if k not in {"name", "type"}})

    model.section(section, 1, GJ=1e3)
    for i in range(len(materials)):
        for fiber in shape.create_fibers(warp=shear, group=i):
            model.fiber(**fiber, material=i+1, section=1)

    # Define two nodes at (0,0)
    model.node(1, (0, 0, 0))
    model.node(2, (0, 0, 0))

    # Fix all degrees of freedom except axial and bending
    model.fix(1, (1, 1, 1,  1, 1, 1))
    model.fix(2, boundary)

    # Create element
    model.element("zeroLengthSection", 1, (1, 2), 1)

    return model


def _analyze_direction(model,
                       direction,
                       maxK, numIncr,
                       initial=None
                       ):
    """
    Arguments
       axialLoad -- axial load applied to section (negative is compression)
       maxK      -- maximum curvature reached during analysis
       numIncr   -- number of increments used to reach maxK (default 100)
    """
    if initial is not None:
        # Define constant axial load
        model.pattern("Plain", 1, "Constant", loads={2: initial})

        # Define analysis
        model.system("BandGeneral")
        model.numberer("Plain")
        model.constraints("Plain")
        model.test("NormUnbalance", 1.0e-8, 20, 0)
        model.algorithm("Newton")
        model.integrator("LoadControl", 0.0)
        model.analysis("Static")

        # Do one analysis for constant axial load
        model.analyze(1)

    # Define reference moment
    model.pattern("Plain", 2, "Linear")
    model.load(2, direction, pattern=2)

    # Compute curvature increment
    dK = maxK/numIncr

    # Use displacement control at node 2 for section analysis
    model.integrator("DisplacementControl", 2, 6, dK, 1, dK, dK)

    MK = []
    for i in range(numIncr):

        # Get moment and curvature
        M = model.eleResponse(1, "force")[5]
        k = model.nodeDisp(2, 6)

        MK.append([-M, k])

        # Evaluate step
        if model.analyze(1) != 0:
            break

    return MK


class SectionInteraction:
    def __init__(self, section, axial, limit=None):
        self.axial = axial
        self._section = section
        if limit is None:
            pass
        self._limit = limit


    def moment_curvature(self):

        # test = lambda m: m.eleResponse(1, "section", "fiber", (0.0, 0.0), "stress")[0] < 0.9*50*ksi

        for N in self.axial:
            model = _create_model(*self._section)
            M, k = zip(*_analyze_direction(model,
                                           (0,0,0,  0,0,1), 
                                           0.005, 200, 
                                           initial=[N,0,0,   0,0,0]))
            yield N, M, k

    def create_model(self):
        return _create_model(*self._section)


    def surface2(self, nstep = 30, incr=5e-6):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(1,2, sharey=True, constrained_layout=True)
        sect = _SectionInterface(*self._section)
        axial = self.axial

        # Curvature increment
        dkap = incr
        s = sect
        for P in axial:
            PM, kmax = _analyze(s, P, dkap, nstep)

            p, m = zip(*PM)

            ax[0].scatter(np.linspace(0.0, kmax, len(m)), m, s=0.2)

            ax[1].scatter(p, m, s=0.2)

        ax[0].set_xlabel("Curvature, $\\kappa$")
        ax[0].set_ylabel("Moment, $M(\\varepsilon, \\kappa)$")
        ax[1].set_xlabel("Axial force, $P$")
        # ax[1].set_ylabel("Moment, $M$")

        # plt.show()


def _prep_control_load(control_dof, point):
    """Return a 6-vector load for node 2 with only `control_dof` nonzero."""
    v = [0,0,0, 0,0,0]
    v[control_dof-1] = point
    return v

def _linearity_break(r0, r1, r, k, tol):
    """
    r0, r1, r are 2-vectors (responses at steps 0,1,k). 
    We take r0 as the origin (usually [0,0]) and test deviation of r from the
    extrapolated line through r1 by a factor k.
    """
    # predicted linear response at step k
    n1 = np.linalg.norm(r1)
    if n1 < 1e-16:
        return False
    r_pred = (k)*np.array(r1)  # since r0 ~ 0, slope ~ r1 per unit step
    err = np.linalg.norm(np.array(r) - r_pred) / (np.linalg.norm(r_pred) + 1e-16)
    return err > tol

def _analyze_in_plane(model, control_vec, direction_vec,
                      max_scale, numIncr,
                      resp_node, dof_i, dof_j, linear_tol):
    """
    Push along a 2-DOF direction using proportional load control and
    return the last 'linear' response (ri, rj). If it never breaks, return the last step.
    """
    # optional: apply control (e.g., axial force or a fixed displacement as a load)
    if control_vec is not None and any(abs(x) > 0 for x in control_vec):
        model.pattern("Plain", 1, "Linear", loads={2: control_vec})
        model.system("BandGeneral")
        model.numberer("Plain")
        model.constraints("Plain")
        model.test("NormUnbalance", 1.0e-10, 20, 0)
        model.algorithm("Newton")
        model.integrator("LoadControl", 0.1)  # keep pattern 1 active without stepping it
        model.analysis("Static")
        if model.analyze(10) != 0:
            return -1, (0,0)
        model.loadConst(time=0)
    # proportional load in (dof_i, dof_j)

    model.pattern("Plain", 2, "Linear")
    model.load(2, direction_vec, pattern=2)

    dlam = max_scale / max(1, numIncr)

    # analysis setup
    model.system("BandGeneral")
    model.numberer("Plain")
    model.constraints("Plain")
    model.test("NormUnbalance", 1.0e-8, 15, 0)
    model.algorithm("Newton")
    model.integrator("LoadControl", dlam)
    model.analysis("Static")

    # responses
    hist = []
    last_ok = (0.0, 0.0)

    # for k in range(1, numIncr+1):
    k = 0
    max_step = 7000
    while True:
        k += 1
        if k > max_step:
            k = None
            break

        if model.analyze(1) != 0:
            # lost convergence -> last_ok is the limit
            break

        ri = model.eleResponse(1, "force")[dof_i-1]
        rj = model.eleResponse(1, "force")[dof_j-1]
        hist.append((ri, rj))

        last_ok = (ri, rj)

        if len(hist) >= 2:
            if model.numIter() > 1: #_linearity_break(np.zeros(2), r1, rk, k, linear_tol):
                # return the previous step (last linear)
                if len(hist) >= 1:
                    last_ok = tuple(hist[-1])
                break

    print(k, last_ok)
    return k, last_ok

def limit_surface(section_tuple,
                  control_dof, control_points,
                  dof_i, dof_j,
                  nr=72,
                  max_scale=1.0,  # scale factor for the proportional load path
                  numIncr=200,
                  linear_tol=0.02,
                  resp_node=2):
    """
    Build a set of (control_value, angle, ri, rj) samples approximating the limit surface
    in the (dof_i, dof_j) response plane for each control point.

    Parameters
    ----------
    section_tuple: tuple
      ("Fiber", shape, materials)
    control_dof: int
      node DOF (at node=2) whose value we hold fixed via a Constant load vector
    control_points: 
      iterable of scalar control values to sweep
    dof_i, dof_j: int 
      the two response DOFs (node=2) that define the plane
    nr: int 
      number of direction angles in [0, 2 pi)
    max_scale: float 
      total proportional-load scale in the (dof_i, dof_j) plane
    numIncr: int
      steps along each ray
    linear_tol: float
      relative deviation from linearity at which we declare the “limit”
    resp_node: int
      node ID for responses (default 2)
    """
    samples = []  # list of dicts: {"control": c, "angle": th, "ri": ri, "rj": rj}
    boundary = [1,1,1,  1,1,1]
    boundary[control_dof-1] = 0 
    boundary[dof_i-1] = 0
    boundary[dof_j-1] = 0

    boundary = tuple(boundary)

    for c in control_points:
        # build a fresh model per control value to avoid path-history carryover

        control_vec = _prep_control_load(control_dof, c)

        for th in np.linspace(0.0, 2*np.pi, nr, endpoint=False):
            di, dj = np.cos(th), np.sin(th)

            # proportional load direction vector as a 6-vector
            dir6 = [0,0,0, 0,0,0]
            dir6[dof_i-1] = float(di)
            dir6[dof_j-1] = float(dj)

            model = _create_model(*section_tuple, boundary)
            k, (ri, rj) = _analyze_in_plane(model,
                                       control_vec=tuple(control_vec),
                                       direction_vec=tuple(dir6),
                                       max_scale=max_scale,
                                       numIncr=numIncr,
                                       resp_node=resp_node,
                                       dof_i=dof_i, 
                                       dof_j=dof_j,
                                       linear_tol=linear_tol)
            
            # if k is not None and k < 10:
            #     continue
            if k is None:
                continue

            samples.append({"control": c,
                            "angle": float(th), "ri": float(ri), "rj": float(rj)})

        # clean up for the next control point
        try:
            model.wipe()
        except Exception:
            pass

    rimax = max((abs(s["ri"]) for s in samples))
    rjmax = max((abs(s["rj"]) for s in samples))
    cmax = max((abs(s["control"]) for s in samples))
    if rimax == 0 or rjmax == 0:
        # no valid samples, return empty
        return []
    
    # Normalize ri, rj to [0,1] range
    for s in samples:
        s["ri"] = float(s["ri"]/rimax)
        s["rj"] = float(s["rj"]/rjmax)
        s["control"] = float(s["control"]/cmax)

    return samples



def _group_rings(samples, eps=1e-12):
    # Group by control value (with tolerance) and sort each ring by angle
    arr = np.array([(s["control"], s["angle"], s["ri"], s["rj"]) for s in samples], float)
    if arr.size == 0:
        return []

    # Cluster control values within tolerance
    controls = arr[:,0]
    order = np.argsort(controls)
    arr = arr[order]

    rings = []
    start = 0
    for i in range(1, len(arr)+1):
        if i == len(arr) or abs(arr[i,0] - arr[start,0]) > eps:
            ring = arr[start:i]
            # sort by angle
            ring = ring[np.argsort(ring[:,1])]
            rings.append(ring)
            start = i
    return rings

def _build_strip_tris(rings, require_equal_counts=False):
    # Build triangles between ring r and r+1 by connecting angle bins
    triangles = []
    idx_map = []
    xs, ys, zs = [], [], []

    # Flatten while remembering indices
    base = 0
    Ns = [len(r) for r in rings]
    if require_equal_counts and len(set(Ns)) != 1:
        raise ValueError("Rings have different sample counts; re-run with uniform nr.")

    for r, ring in enumerate(rings):
        idx_map.append(np.arange(base, base + len(ring)))
        xs.extend(ring[:,2])
        ys.extend(ring[:,3])
        zs.extend(ring[:,0])
        base += len(ring)

    # Triangulate strips
    for r in range(len(rings)-1):
        n0 = len(rings[r])
        n1 = len(rings[r+1])
        if n0 < 2 or n1 < 2:
            continue

        # If counts differ, map by nearest angle indices
        if n0 == n1:
            for a in range(n0):
                a0 = idx_map[r][a]
                a1 = idx_map[r][(a+1) % n0]
                b0 = idx_map[r+1][a]
                b1 = idx_map[r+1][(a+1) % n1]
                triangles.append([a0, a1, b0])
                triangles.append([a1, b1, b0])
        else:
            # nearest-angle mapping
            ang0 = rings[r][:,1]
            ang1 = rings[r+1][:,1]
            # ensure periodicity [0, 2 pi)
            for a in range(n0):
                a_next = (a+1) % n0
                # find nearest indices in ring r+1 for angle a and a_next
                j  = int(np.argmin(np.abs(ang1 - ang0[a])))
                jn = int(np.argmin(np.abs(ang1 - ang0[a_next])))
                a0 = idx_map[r][a]
                a1 = idx_map[r][a_next]
                b0 = idx_map[r+1][j]
                b1 = idx_map[r+1][jn]
                if b0 == b1:  # degenerate pair; skip one tri
                    triangles.append([a0, a1, b0])
                else:
                    triangles.append([a0, a1, b0])
                    triangles.append([a1, b1, b0])

    return np.asarray(xs), np.asarray(ys), np.asarray(zs), np.asarray(triangles, dtype=int)



def triangulate(samples, elev=25, azim=-60, alpha=0.9, figure=None):
    if not samples:
        raise ValueError("samples is empty")

    rings = _group_rings(samples)
    # Drop rings that collapsed to a single unique (ri,rj)
    rings = [
        r for r in rings
        if len(r) >= 3 and np.linalg.norm(np.std(r[:,2:4], axis=0)) > 1e-14
    ]

    x, y, z, tris = _build_strip_tris(rings)
    return (x,y,z), tris


def plot_limit_surface(samples, elev=25, azim=-60, alpha=0.9, figure=None):
    import veux
    if not samples:
        raise ValueError("samples is empty")

    rings = _group_rings(samples)
    # Drop rings that collapsed to a single unique (ri,rj)
    rings = [
        r for r in rings
        if len(r) >= 3 and np.linalg.norm(np.std(r[:,2:4], axis=0)) > 1e-14
    ]

    x, y, z, tris = _build_strip_tris(rings)

    c = veux._create_canvas("gltf")
    c.plot_mesh(np.array([x, z, y]).T, tris)
    return c
