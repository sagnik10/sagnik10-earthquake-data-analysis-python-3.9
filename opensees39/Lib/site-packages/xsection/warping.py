from ._base import Shape
class Mesh:
    nodes: list
    elems: list

class Material:
    pass


class WarpingSection(Shape):

    def __init__(self,
                 model: "TriangleModel",
                 warp_twist=True,
                 warp_shear=True
        ):

        self._w_model = model

        self._warp_shear:bool = warp_shear
        self._warp_twist:bool = warp_twist

    #
    # Virtual
    #
    @property
    def model(self):
        if self._w_model is None:
            raise ValueError("Model not initialized")
        return self._w_model

    @property
    def _analysis(self):
        from xsection.analysis.warping import WarpingAnalysis as TorsionAnalysis
        
        if not hasattr(self, "_warp_analysis") or self._warp_analysis is None:
            self._warp_analysis = TorsionAnalysis(self.model)

        return self._warp_analysis 

    #
    # Final
    #
    @classmethod
    def from_meshio(cls, mesh, **kwds):
        from shps.frame.solvers import TriangleModel
        return WarpingSection(TriangleModel.from_meshio(mesh), **kwds)


    def export(self, file_name=None, format=None):
        pass
    
    def send_xara(self, model, type, tag,
                  material=None, warp=True, shear=False):

        if type == "Elastic":
            cmm = self.cmm()
            cnn = self.cnn()
            cnv = self.cnv()
            cnm = self.cnm()
            cmw = self.cmw()
            cww = self.cww()
            cvv = self.cvv()
            A = cnn[0,0]
            model.section("ElasticFrame", tag,
                            E=material["E"],
                            G=material["G"],
                            A=cnn[0,0],
                            # Ay=1*A,
                            # Az=1*A,
                            # Qy=cnm[0,1],
                            # Qz=cnm[2,0],
                            Iy=cmm[1,1],
                            Iz=cmm[2,2],
                            J =self._analysis.torsion_constant(),
                            Cw= cww[0,0],
                            Ry= cnv[1,0],
                            Rz= cnv[2,0],
                            Sy= cvv[1,1],
                            Sz= cvv[2,2]
            )
        else:
            if isinstance(material,int):
                mtag = material 
            else:
                mtag = 1
                if "name" in material:
                    mtype = material["name"]
                    del material["name"]
                else:
                    mtype = "ElasticIsotropic"

                model.material(mtype, mtag, **material)

            model.section("ShearFiber", tag, GJ=0)

            for fiber in self.create_fibers(warp=warp, shear=shear):
                model.fiber(**fiber, material=mtag, section=tag)

    
    def exterior(self):
        return self._w_model.exterior()

    def interior(self):
        return self._w_model.interior()
    
    @property
    def centroid(self):
        return self._analysis.centroid()
    
    @property 
    def depth(self):
        return max(self.model.nodes[:,1]) - min(self.model.nodes[:,1])

    def torsion_warping(self):
        return self._analysis.warping()
    
    def cnn(self):
        return self._analysis.cnn()
    def cnm(self):
        return self._analysis.cnm()
    def cnv(self):
        return self._analysis.cnv()
    def cmm(self):
        return self._analysis.cmm()
    def cmw(self):
        return self._analysis.cmw()
    def cmv(self):
        return self._analysis.cmv()
    def cww(self):
        return self._analysis.cww()
    def cvv(self):
        return self._analysis.cvv()
    def css(self):
        return self._analysis.css()

    def summary(self, shear=False):
        s = ""
        tol=1e-13
        A = self._analysis.cnn()[0,0]

        cnw = self._analysis.cnw()
        cnm = self._analysis.cnm()
        cmm = self._analysis.cmm()
        cmw = self._analysis.cmw()
        cnv = self._analysis.cnv(shear=shear)

        cmv = self._analysis.cmv()
        cvv = self._analysis.cvv()

        # Compute centroid
        Qy = cnm[0,1] # int z
        Qz = cnm[2,0] # int y
        cx, cy = float(Qz/A), float(Qy/A)
        cx, cy = map(lambda i: i if abs(i)>tol else 0.0, (cx, cy))

        # Irw = self.torsion.cmv()[0,0]

        sx, sy = self._analysis.shear_center()
        sx, sy = map(lambda i: i if abs(i)>tol else 0.0, (sx, sy))
        if shear:
            nu = 0
            ky, kz = self._analysis.shear_factor(v=self._analysis.shear_warping(nu), nu=nu)
            ky = f"{ ky :>10.6}"
            kz = f"{ kz :>10.6}"
        else:
            ky = f""
            kz = f""

        cww = self._analysis.cww()
        # Translate to shear center to get standard Iww
        Iww = self.translate([-sx, -sy])._analysis.cww()[0,0]

        Isv = self._analysis.torsion_constant()

        s += f"""
  Block                              x            y           z           yz
  [nn]    Area               {A          :>10.4}   {ky}, {kz}
  [nm]    Centroid           {0.0        :>10.4}   {cx         :>10.4}, {cy         :>10.4}
  [nw|v]                     {cnw[0,0]/A :>10.4} | {cnv[1,0]/A :>10.4}, {cnv[2,0]/A :>10.4}

  [mm]    Flexural moments   {cmm[0,0]   :>10.4}   {cmm[1,1]   :>10.4}, {cmm[2,2]   :>10.4}, {cmm[1,2] :>10.4}
  [mv|w]                     {cmv[0,0]   :>10.4} | {cmw[1,0]   :>10.4}, {cmw[2,0]   :>10.4}


  [ww]    Warping constant   {cww[0,0] :>10.4}  ({Iww      :>10.4} at S.C.)
  [vv]    Bishear            {cvv[0,0] :>10.4}


          Shear center       {0.0        :>10.4}   {sx         :>10.4}, {sy :>10.4}
          Torsion constant   {Isv :>10.4}
        """

        return s


    def translate(self, offset):
        return WarpingSection(self.model.translate(offset)) 

    def rotate(self, angle=None, principal=None):
        if angle is not None:
            return WarpingSection(self.model.rotate(angle)) 


    def _principal_rotation(self):
        from shps.rotor import log
        import numpy as np
        import numpy.linalg as la
        # P = [[0,1,0],[1,0,0],[1,0,0]]
        I = self._analysis.cmm()

        vals, vecs = la.eig(I[1:,1:])
        # sort = vals.argsort()
        # Q = vecs[:,sort]
        Q = np.eye(3)
        Q[1:,1:] = vecs
        # print(Q)
        # Q = np.array([q for q in reversed(Q)])
        theta = log(Q)
        assert np.isclose(theta[1], 0.0), theta
        assert np.isclose(theta[2], 0.0), theta
        return float(theta[0])

    @property
    def elastic(self):
        from xsection import ElasticConstants
        import numpy as np
        y, z = self.model.nodes.T
        e = np.ones(y.shape)
        k, _ = self._analysis.shear_factor_romano()
        A = self.model.inertia(e, e)
        return ElasticConstants(
            A  =A,
            Ay =float(k[0]*A),
            Az =float(k[1]*A),
            Iyz=self.model.inertia(y, z),
            Iy =self.model.inertia(z, z),
            Iz =self.model.inertia(y, y),
            J = float(self._analysis.torsion_constant()),
        )

    def integrate(self, f: callable):
        pass

    # def create_fibers(self, *args, **kwds):
    #     yield from self.fibers(*args, **kwds)

    def create_fibers(self, origin=None, center=None, types=None, 
                      warp=True, shear=False,
                      group=None, axes=None, exclude=None):
        """
        use material to force a homogeneous material
        """

        if axes is not None:
            shape = self.rotate(principal=axes)
            if len(axes) == 1: # axes == "z":
                # old z was turned into y
                exclude = axes
            yield from shape.create_fibers(origin=origin, 
                                           center=center, 
                                           warp=warp, 
                                           group=group, 
                                           exclude=exclude)
            return

        if origin is not None:
            if origin == "centroid":
                shape = self.translate(-self.centroid)
            elif origin == "shear-center":
                shape = self.translate(-self._analysis.shear_center())
            else:
                shape = self.translate(-origin)

            yield from shape.create_fibers(center=center,  origin=None, 
                                           warp=warp, shear=shear, 
                                           group=group, axes=axes, exclude=exclude)
            return

        model = self.model

        if center is None:
            twist = self._analysis
            w = self._analysis.solve_twist() #warping() # 
        elif not isinstance(center, str):
            twist = self.translate(center)._analysis
            w = twist.solve_twist()
        elif center == "centroid":
            twist = self.translate(-self.centroid)._analysis
            w = twist.solve_twist()
        elif center == "shear-center":
            w = self._analysis.warping()
            twist = self._analysis


        if shear:
            w_shear = self._analysis.fiber_shear()
        else:
            w_shear = None

        # if callable(self._warp_shear):
        #     psi = self._warp_shear
        # else:
        psi = lambda y,z: 0.0

        for i,elem in enumerate(self.model.elems):
            if group is not None and elem.group != group:
                continue
            # TODO: Assumes TriangleModel
            yz = sum(model.nodes[elem.nodes])/3
            fiber = dict(area=model.cell_area(i))
            
            fiber["y"] = float(yz[0])
            fiber["z"] = float(yz[1])

            # fiber["warn-2d-z"] = True
            if warp:
                fiber["warp"] = [
                    [twist.model.cell_solution(i, w), *map(float, twist.model.cell_gradient(i,  w))],
                    [0, 0, 0],
                    [0, 0, 0]
                ]
                if shear:
                    fiber["warp"][1] = [
                        twist.model.cell_solution(i, w_shear[0]), 
                        *map(float, model.cell_gradient(i,  w_shear[0]))  
                    ]
                    fiber["warp"][2] = [
                        twist.model.cell_solution(i, w_shear[1]), 
                        *map(float, model.cell_gradient(i,  w_shear[1]))  
                    ]

            yield fiber

    def _repr_html_(self):
        import veux
        from veux.viewer import Viewer
        m = self.model
        a = veux.create_artist((m.nodes, m.cells()), ndf=1)
        a.draw_surfaces()
        viewer = Viewer(a,hosted=False,standalone=False)
        html = viewer.get_html()
        return html

import numpy as np

def orthogonalize(v1, v2, v3, *, normalize=False, atol=1e-12):
    """
    Make three vectors mutually orthogonal.
    - normalize=True  → return orthonormal vectors
      normalize=False → keep original norms
    """
    V = np.column_stack([np.asarray(v1), np.asarray(v2), np.asarray(v3)])  # (n,3)

    # Householder-QR is stable and simple
    Q, R = np.linalg.qr(V, mode="reduced")  # Q: (n,3), R: (3,3)

    # Guard against (near) linear dependence
    if np.min(np.abs(np.diag(R))) < atol:
        raise ValueError("Input vectors are nearly linearly dependent; can't produce 3 independent orthogonal vectors.")

    if normalize:
        u1, u2, u3 = Q.T
    else:
        # Rescale to the original norms
        norms = np.linalg.norm(V, axis=0)
        Q = Q * norms
        u1, u2, u3 = Q.T

    return u1, u2, u3