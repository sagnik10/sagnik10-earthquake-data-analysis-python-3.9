import numpy as np
from typing import Dict, Any, Callable


class CellField(Callable): ...

class NodeField(Callable): ...

class PatchRecovery:
    pass 

class MomentDiagram:
    pass

class Curvature:
    pass

class Resultants:
    pass


class NodalAverage(NodeField):
    def __init__(self, model, response: CellField):
        """
        Return a dictionary of stress components per node.
        For 2D: sxx, syy, sxy
        For 3D: sxx, syy, szz, sxy, syz, sxz
        Summed values from each element's node contribution are averaged at the end.

        Args:
            model: an OpenSees model-like object with the following interface:
                - getEleTags()
                - getEleClassTags(tag: int)
                - getNodeTags()
                - eleNodes(tag: int)
                - eleResponse(tag: int, key: str)
            response (str): a label if needed for e.g. 'stress', 'stressAtNodes'
            ndm (int): number of dimensions (2 or 3)

        Returns:
            dict[node_tag] = {
                "xx": float,
                "yy": float,
                ...
            }
        """
        ndm = model.getNDM()

        if isinstance(response, str):

            resp_func = lambda elem: model.eleResponse(elem, response)
        elif callable(response):
            resp_func = response
        else:
            raise ValueError("response must be a string or callable")

        # For a typical 2D element with 3 stress components: sxx, syy, sxy
        if ndm == 2:
            keys = "sxx", "syy", "sxy"
        else:
            # For a typical 3D element with 6 stress components: sxx, syy, szz, sxy, syz, sxz
            keys = "sxx", "syy", "szz", "sxy", "syz", "sxz"

        nrc = len(keys)

        output = {
            node: {
                "_count": 0,
                **{key: 0.0 for key in keys}
            } for node in model.getNodeTags()
        }


        # Assume each element’s response yields a flat array 
        # with length = nen * nrc.
        # Then we reshape it to (nen, nrc).
        for elem in model.getEleTags():
            elem_nodes = model.eleNodes(elem)
            nen = len(elem_nodes)

            se = resp_func(elem)
            assert not (len(se) % nen), f"Element {elem} response length mismatch: {len(se)} != {nen * nrc}"
            se = np.reshape(se, (nen, nrc))

            # Accumulate at element's nodes
            for i, node in enumerate(elem_nodes):
                for j, key in enumerate(keys):
                    output[node][key] += se[i][j]

                output[node]["_count"] += 1


        # Now do averaging (if a node belongs to multiple elements).
        for node in output:
            c = output[node]["_count"] or 1.0

            for key in keys:
                output[node][key] /= c


        # Clean up
        for node in output:
            output[node].pop("_count", None)


        self._values = output

    def __call__(self, node):
        return self._values[node]



class FrameMoments(CellField):
    def __init__(self, model,  artist):
        self._model = model
        self._artist = artist

    def __call__(self, tag):
        ndm = self._artist.model.ndm
        wy  = -10 if tag == 2 else (-5 if tag == 1 else 0)
        wz  =  -1 if tag == 1 else 0

        X = self._artist.model.cell_position(tag)
        L = np.linalg.norm(X[-1] - X[0])
        if self._artist.model.cell_matches(tag, "prism"):
            xe = np.linspace(0, 1, 20)
            q = self._model.eleResponse(tag, "basicForces")


            if ndm == 3:
                w  = np.array([   0 ,  wz ,  wy ])
                qi = np.array([-q[3], q[4], q[1]])
                qj = np.array([ q[3], q[5], q[2]])
            else:
                w  = np.array([0, 0,  wy ])
                qi = np.array([0, 0, q[1]])
                qj = np.array([0, 0, q[2]])

            ye = [ (qi*(x-1) + qj*x + (1-x)*x * w * L*L/2) for x in xe ]
            return xe, ye

        else:

            # xe = [x for x,_ in self._artist.model.cell_quadrature(tag)]
            xe = [x/L for x in self._model.eleResponse(tag, "integrationPoints")]
            ye = [self._model.eleResponse(tag, "section", i+1, "deformations")[3:6]
                  for i in range(len(xe))]
            return xe, ye


class NodalStress(NodalAverage):
    def __init__(self, model, component, method=None):
        import xara
        if not isinstance(model, xara.Model):
            model = model._model 

        self._model = model
        self._component = component
        self._method = method

        super().__init__(model, "stressAtNodes")


    def __call__(self, node):
        s = self._values[node]
        if self._component in s:
            return s[self._component]

        elif self._component.lower() == "j2":
            return np.sqrt(
                (s["sxx"] - s["syy"])**2 +
                (s["syy"] - s["szz"])**2 +
                (s.get("szz",0) - s["sxx"])**2 +
                6*(s["sxy"]**2 + s.get("syz",0)**2 + s.get("sxz",0)**2)
            )/np.sqrt(2)


def node_average(model, response: CellField, ndm=2, keys=None):
    """
    Return a dictionary of stress components per node.
    For 2D: sxx, syy, sxy
    For 3D: sxx, syy, szz, sxy, syz, sxz
    Summed values from each element's node contribution are averaged at the end.

    Args:
        model: an OpenSees model-like object with the following interface:
               - getEleTags()
               - getEleClassTags(tag: int)
               - getNodeTags()
               - eleNodes(tag: int)
               - eleResponse(tag: int, key: str)
        response (str): a label if needed for e.g. 'stress', 'stressAtNodes'
        ndm (int): number of dimensions (2 or 3)

    Returns:
        dict[node_tag] = {
            "xx": float,
            "yy": float,
            ...
        }
    """
    if isinstance(response, str):
        resp_func = lambda elem: model.eleResponse(elem, response)
    elif callable(response):
        resp_func = response
    else:
        raise ValueError("response must be a string or callable")


    # For a typical 2D element with 3 stress components: sxx, syy, sxy
    # For a typical 3D element with 6 stress components: sxx, syy, szz, sxy, syz, sxz
    # You may need to adapt this depending on your element type(s).
    if ndm == 2:
        keys = "sxx", "syy", "sxy"
    else:
        keys = "sxx", "syy", "szz", "sxy", "syz", "sxz"

    nrc = len(keys)

    # Initialize a dictionary to hold stress sums and a count of how many
    # times each node is encountered (to do averaging).
    output = {node: {"_count": 0, **{key: 0.0 for key in keys}} for node in model.getNodeTags()}


    # We assume each element’s response yields a flat array 
    # with length = nen * nrc.
    # Then we reshape it to (nen, nrc).
    for elem in model.getEleTags():
        elem_nodes = model.eleNodes(elem)
        nen = len(elem_nodes)

        se = resp_func(elem)
        assert not (len(se) % nen),  f"Element {elem} response length mismatch: {len(se)} != {nen * nrc}"
        se = np.reshape(se, (nen, nrc))

        # Accumulate at element's nodes
        for i, node in enumerate(elem_nodes):
            for j, key in enumerate(keys):
                output[node][key] += se[i][j]

            output[node]["_count"] += 1


    # Now do averaging (if a node belongs to multiple elements).
    for node in output:
        c = output[node]["_count"] or 1.0

        for key in keys:
            output[node][key] /= c


    # Clean up
    for node in output:
        output[node].pop("_count", None)

    return output


def _stress_spr(xyz,  ElemData, Stress) -> Dict[str, np.ndarray]:
    """
    Super-patch stress recovery for quadrilateral membrane elements.

    - Zienkiewicz and Zhu (1992) The superconvergent patch recovery and a posteriori error estimates. Part 1: The recovery technique DOI: https://doi.org/10.1002/nme.1620330702


    Parameters
    ----------
    xyz : (ndm, nen) ndarray
        Undeformed nodal coordinates.
    ElemState : (something) ndarray
        Element-level displacement array (layout compatible with `ExtrReshu`).
    ElemData : object
        Element-specific data structure.  Must expose at least:
            • nIP   – number of Gauss points per direction
            • nodix – local-to-global node index map
            • Geom  – geometric-transformation data structure
    Stress : callable
        Material routine.  Must accept the signature
        `Stress('post', ip_index)` and return an object
    ndm : int, default 2
        Spatial dimension (2 ⇒ membrane problems).
    """
    nen  = xyz.shape[1]            # number of nodes

    # Geometric transformations ---------------------------------------
    xyzl, _   = DefGeom_Quad(xyz)                         # local xyz

    # ------------------------------------------------------------------
    # Gauss integration
    # ------------------------------------------------------------------
    nstr = 4                                # σ-vector length (σ_x, σ_y, τ_xy, p)
    sigNd = np.zeros((nen, nstr))
    LSQM  = np.zeros((nen, nen))

    # Gauss points & weights (nIP × nIP rule) --------------------------
    nat, wIP = Gauss2d(ElemData.nIP)        # nat.shape = (nIP**2, 2)

    for i, (ξ, η) in enumerate(nat):
        # Shape functions & derivatives -------------------------------
        N, _, J = shape2d((ξ, η), xyzl[:2, :].T, ElemData.nodix)

        sig = Stress("post", i)

        # Accumulate recovery terms -----------------------------------
        detJ = float(np.linalg.det(J))
        weight = detJ * wIP[i]

        sigNd += np.outer(N, sig) * weight
        LSQM  += np.outer(N, N) * weight

    return {"sig": sigNd, "LSQM": LSQM}


def project_element_stress_to_nodes(xyz, ElemData, Stress) -> Dict[str, np.ndarray]:
    """
    Project Gauss-point stresses to the element’s nodes with a least-squares fit.

    Parameters
    ----------
    xyz : (ndm, nen) ndarray
        Undeformed nodal coordinates (global system).
    ElemData : object
        Element-specific data structure that exposes
            • nIP   – Gauss-rule order (nIP × nIP)
            • nodix – local-to-global node map (passed straight to `shape2d`)
    Stress : callable
        A routine that returns the (already *global*) stress vector at an
        integration point: `sig = Stress("post", ip_index)`
        `sig` must be length-4 (σₓ, σ_y, τ_xy, p).
    ndm : int, default 2
        Spatial dimension (membranes ⇒ 2).

    """
    nen = xyz.shape[1]                         # # nodes

    # local geometry (no displacement-based rotation needed here)
    xyzl, _ = DefGeom_Quad(xyz)                # (3, nen)

    # integration rule -------------------------------------------------
    nat, wIP = Gauss2d(ElemData.nIP)           # nat: (nIP**2, 2)

    r    = np.zeros((nen, 4))                  # RHS vector
    LSQM = np.zeros((nen, nen))                # least-squares matrix

    # ------------------------------------------------------------------
    # loop over Gauss points
    # ------------------------------------------------------------------
    for i, (xi, eta) in enumerate(nat):
        # shape functions & Jacobian -----------------------------------
        N, _dNdx, J = shape2d((xi, eta), xyzl[:2, :].T, ElemData.nodix)

        sig = Stress("post", i)                # (4,) stress in *global* coords

        detJ = float(np.linalg.det(J))
        w    = detJ * wIP[i]

        r    += np.outer(N, sig) * w
        LSQM += np.outer(N, N) * w

    # ------------------------------------------------------------------
    # nodal stresses: solve LSQM · sigNd = r
    # ------------------------------------------------------------------
    # (nen × nen) · (nen × 4) = (nen × 4)
    try:
        sigNd = np.linalg.solve(LSQM, r)
    except np.linalg.LinAlgError:
        # fall back to least-squares if LSQM is singular/ill-conditioned
        sigNd = np.linalg.lstsq(LSQM, r, rcond=None)[0]

    return {"sigNd": sigNd, "LSQM": LSQM}

