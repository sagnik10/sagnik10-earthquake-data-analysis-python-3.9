import xara
from dataclasses import dataclass


def create_section(model, type, tag, shape, material, warp=True, shear=True, center=None):

    if type == "Elastic":
        cmm = shape.cmm()
        cnn = shape.cnn()
        cnv = shape.cnv()
        cnm = shape.cnm()
        cmw = shape.cmw()
        cww = shape.cww()
        cvv = shape.cvv()
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
                        J =shape._analysis.torsion_constant(),
                        Cw= cww[0,0],
                        Ry= cnv[1,0],
                        Rz= cnv[2,0],
                        Sy= cvv[1,1],
                        Sz= cvv[2,2]
        )

    elif type == "Uniaxial":
        pass

    else:
        if isinstance(material,int):
            mtag = material
        else:
            mtag = 1
            if material is None:
                material = {"name": "ElasticIsotropic", "E": 1, "G": 1}
            if "name" in material:
                mtype = material["name"]
                del material["name"]
            else:
                mtype = "ElasticIsotropic"

            model.material(mtype, mtag, **material)

        model.section("ShearFiber", tag, GJ=0)

        for fiber in shape.create_fibers(warp=warp, shear=shear):
            model.fiber(**fiber, material=mtag, section=tag)



@dataclass
class Prism:
    length: float
    shape: str
    material: dict
    boundary: tuple
    element: str = "PrismFrame"
    section: str = "Elastic"
    transform: str = "Linear"
    geometry : str = None
    vertical: int = 3
    rotation: tuple = None
    divisions: int = 1
    order: int = 1
    shear: int = 1
    shear_warp: bool = False
    warp: list = None
    mass: float = None
    joint_offset: dict=None


    def __repr__(self):
        return f"""\
        Prism(length={self.length}, 
              shape={self.shape},
              material={self.material}, 
              boundary={self.boundary}, 
              element={self.element}, 
              section={self.section}, 
              transform={self.transform}, 
              geometry={self.geometry}, 
              vertical={self.vertical}, 
              rotation={self.rotation}, 
              divisions={self.divisions}, 
              order={self.order}, 
              shear={self.shear}
              warp={self.warp})
        """


    def create_model(self,
                    geometry:  str = None,
                    echo_file=None):

        L  = self.length
        element = self.element
        shape = self.shape
        material = self.material
        boundary = self.boundary
        transform = self.transform
        vertical = self.vertical
        divisions = self.divisions
        rotation = self.rotation
        section = self.section
        order = self.order
        shear = self.shear
        geometry = self.geometry
        warp = self.warp
        if warp is None:
            warp = [0,0,0]




        # Number of elements discretizing the column
        ne = divisions

        nen = order + 1
        nn = ne*(nen-1)+1

        model = xara.Model(ndm=3, ndf=6+sum(warp), echo_file=echo_file)

        for i in range(1, nn+1):
            x = (i-1)*L/float(nn-1)
            location = (x, 0, 0)

            if rotation is not None:
                location = tuple(rotation@location)

            model.node(i, location)


        # Define boundary conditions
        model.fix( 1, boundary[0])
        model.fix(nn, boundary[1])

        #
        # Define cross-section 
        #
        if isinstance(shape, dict):
            model.section("ElasticFrame", 1, **shape, **material)
        elif callable(section):
            section(model, 1, shape, material)
        else:
            shape.send_xara(model,
                            section, 1,
                            material,
                            warp=warp or self.shear_warp,
                            shear=self.shear_warp)

        # Define geometric transformation
        if vertical == 3:
            orient = (0, 0, 1)
        if rotation is not None:
            orient = tuple(map(float, rotation@orient))

        if self.joint_offset is not None:
            model.geomTransf(transform, 1, orient, offset=self.joint_offset)
        else:
            model.geomTransf(transform, 1, *orient)


        # Define elements
        options = {}
        if self.mass is not None:
            options["mass"] = self.mass

        for i in range(ne):
            start = i * (nen - 1) + 1
            nodes = list(range(start, start + nen))
            if geometry is None or geometry == "Linear" or "Exact" in element:
                model.element(element, i+1, nodes,
                              section=1,
                              shear=shear,
                              transform=1, **options)
            else:
                model.element(element, i, nodes,
                            section=1,
                            shear=shear,
                            order={"Linear": 0, "delta": 1}[geometry],
                            transform=1, **options)

        return model

