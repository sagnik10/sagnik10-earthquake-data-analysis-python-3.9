import re
import sys
import json
import warnings 
from collections import namedtuple
from .part import Part
import numpy as np

_Element = namedtuple("Element", ["cell", "xara_name", "options"])

_Elements = {
    # trusses
    "T2D2":  _Element("line",   "Truss",   None),
    "T2D2H": _Element("line",   None,      None),
    "T2D3":  _Element("line3",  None,      None),
    "T2D3H": _Element("line3",  None,      None),
    "T3D2":  _Element("line",   "Truss",   None),
    "T3D2H": _Element("line",   None,      None),
    "T3D3":  _Element("line3",  None,      None),
    "T3D3H": _Element("line3",  None,      None),
    # beams
    "B21":      _Element("line",   "ForceFrame",  None),
    "B21H":     _Element("line",   "ForceFrame",  None),
    "B22":      _Element("line3",  None,  None),
    "B22H":     _Element("line3",  None,  None),
    "B31":      _Element("line",   "ExactFrame",  None),
    "B31H":     _Element("line",   None,  None),
    "B32":      _Element("line3",  None,  None),
    "B32H":     _Element("line3",  None,  None),
    "B33":      _Element("line3",  "CubicFrame",  None), # 2-node cubic beam
    "B33H":     _Element("line3",  None,  None),

    # Warping (1-DOF)
    "WARP2D4":  _Element("quad",       None,  None),
    "WARP2D3":  _Element("triangle",   None,  None),
    # Warping (3-DOF)
    "WARPF2D3": _Element("triangle",   None,  None),
    "WARPF2D4": _Element("quad",       None,  None),
    "WARPF2D6": _Element("triangle6",  None,  None),
    "WARPF2D8": _Element("quad8",  None,  None),

    # surfaces
    "CPS4":     _Element("quad",  None,  None),
    "CPS4R":    _Element("quad",  None,  None),

    "S4":       _Element("quad",  "ASDShellQ4",  None),
    "S4R":      _Element("quad",  "ASDShellQ4",  None),
    "S4RS":     _Element("quad",   None,         None),
    "S4RSW":    _Element("quad",   None,         None),
    "S4R5":     _Element("quad",   None,         None),
    "S8R":      _Element("quad8",  None,         None),
    "S8R5":     _Element("quad8",  None,         None),
    "S9R5":     _Element("quad9",  None,         None),
    #
    "CPS3":     _Element("triangle",  None,          None),
    "STRI3":    _Element("triangle",  None,          None),
    "S3":       _Element("triangle",  "ShellDKGT",  None),
    "S3R":      _Element("triangle",  "ShellDKGT",  None),
    "S3RS":     _Element("triangle",  None,  None),
    "R3D3":     _Element("triangle",  None,  None),
    #
    "STRI65":   _Element("triangle6",  None,  None),
    # 'TRISHELL6': 'triangle6',

    # volumes
    "C3D8":     _Element("hexahedron",  "stdBrick",     None),
    "C3D8H":    _Element("hexahedron",  "bbarBrickUP",  None),
    "C3D8I":    _Element("hexahedron",  "bbarBrick",    None), # Incompatible mode eight-node brick element, maybe Taylor, R.L, Beresford, P.J. and Wilson, E.L., A non-conforming element for stress analysis. Int. J. Num. Meth. Engng. 10 , 1211-1219 (1976).
    "C3D8IH":   _Element("hexahedron",  None,  None),
    "C3D8R":    _Element("hexahedron",  None,  None),
    "C3D8RH":   _Element("hexahedron",  None,  None),
    # "HEX9": _Element("hexahedron9",  None,  None),
    "C3D20":    _Element("hexahedron20",  None,  None),
    "C3D20H":   _Element("hexahedron20",  None,  None),
    "C3D20R":   _Element("hexahedron20",  None,  None),
    "C3D20RH":  _Element("hexahedron20",  None,  None),
    # "HEX27": _Element("hexahedron27",  None,  None),
    #
    "C3D4":     _Element("tetra",    "FourNodeTetrahedron",  None),
    "C3D4H":    _Element("tetra4",   None,  None),
    # "TETRA8": _Element("tetra8",   None,  None),
    "C3D10":    _Element("tetra10",  "TenNodeTetrahedron",   None),
    "C3D10H":   _Element("tetra10",  None,  None),
    "C3D10I":   _Element("tetra10",  None,  None),
    "C3D10M":   _Element("tetra10",  None,  None),
    "C3D10MH":  _Element("tetra10",  None,  None),
    # "TETRA14": _Element("tetra14",  None,  None),
    #
    # "PYRAMID": _Element("pyramid",  None,  None),
    "C3D6":  _Element("wedge",  None,  None),
    "C3D15": _Element("wedge15",  None,  None),
    #
    # 4-node bilinear displacement and pore pressure
    "CAX4P": _Element("quad",  None,  None),
    # 6-node quadratic
    "CPE6": _Element("triangle6",  None,  None),

    "CONN3D2": _Element("link",  None,  None),
}

_SolidElements = {
    "hexahedron", "hexahedron20", "hexahedron27", "hexahedron9",
    "tetra", "tetra10",
    "wedge", "pyramid",
}

class Instance:
    def __init__(self, 
                 model, part, tree,
                 context=None,
                 root=None,
                 mode = None,
                 name=None):
        self._name  = name
        self._part  = part
        self._model = model
        if mode is None:
            mode = "simulate"

        self._translation = None 
        self._rotation = None
        if len(tree.data) > 0:
            self._translation = tuple(float(i) for i in tree.data[0].split(","))
        if len(tree.data) > 1:
            self._rotation = tuple(map(float, tree.data[1].split(",")))


        _create_part(part, model, context, mode=mode, root=root, inst=self)

        if id(tree) != id(part._tree):
            _create_part(Part(tree), model, context, mode=mode, root=root, inst=self)


    def translate(self, vector):
        if self._translation is None:
            return vector
        return np.array(vector) + np.array(self._translation)

    def rotate(self, vector):
        if self._rotation is None:
            return vector

        a = np.array(self._rotation[:3])
        b = np.array(self._rotation[3:6])
        angle = np.deg2rad(self._rotation[6])

        # Direction vector of the axis (unit vector)
        axis = b - a
        axis /= np.linalg.norm(axis)

        # Translate so rotation axis passes through origin
        v = np.array(vector) - a

        # Rodrigues' rotation formula
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        dot = np.dot(axis, v)
        cross = np.cross(axis, v)
        v_rot = (
            v * cos_theta +
            cross * sin_theta +
            axis * dot * (1 - cos_theta)
        )

        # Translate back
        return v_rot + a



def create_materials(ast, model, conv, root=None):
    part = Part(ast, root=root or ast)
    for node in ast.find_all("Material"):
        m = part._find_material(node)
        if m is None:
            continue

        E = m.young_modulus
        nu = m.poisson_ratio
        # Create the material
        tag = conv.define("Material", "material", m.name)
        if m.yield_stress is not None:
            assert m.young_modulus is not None
            assert m.poisson_ratio is not None

            if m.hardening == "isotropic":
                model.nDMaterial("J2Simplified", tag, 
                                E=m.young_modulus, 
                                nu=m.poisson_ratio, 
                                Fy=m.yield_stress, 
                                Hiso=m.plastic_modulus or 0,
                                Hkin=0,
                                density=m.density or 0
                )

            elif m.hardening == "JOHNSON COOK":
                warnings.warn("JOHNSON COOK hardening not implemented")
                model.material("ElasticIsotropic", tag, E, nu)

        elif E is not None:
            if nu is not None:
                model.material("ElasticIsotropic", tag, E, nu)
            else:
                assert E is not None
                model.uniaxialMaterial("Elastic", tag, E)

def create_sections(ast, model, conv, root=None, part=None):
    if part is None:
        part = Part(ast, root=root or ast)

    for section in ast.find_all("Shell Section"):
        name = section.attributes.get("name", section.attributes.get("elset", None))
        if name is None:
            raise ValueError("Shell Section without name or elset")

        tag = conv.define("ShellSection", "section", name)

        mat = section.attributes.get("material")
        thickness = section.attributes.get("thickness", None)

        if mat is not None:
            material = part._find_material(ast.find_attr("Material", name=mat))
            if material is None:
                raise ValueError(f"Material {mat} not found")
            
            if thickness is None:
                thickness = float(section.data[0].split(",")[0])

            nip = section.data[0].split(",")
            # TODO: Handle nip
            model.section("ElasticShell", tag,
                          material.young_modulus,
                          material.poisson_ratio,
                          thickness or 1.0,
                        #   density=material.density or 1.0
            )


    for section in ast.find_all("Beam Section"):
        elset = section.attributes.get("elset")
        mat = ast.find_attr("Material", name=section.attributes.get("material"))
        if mat is None:
            mat = root.find_attr("Material", name=section.attributes.get("material"))
        if mat is None:
            raise ValueError(f"Material {section.attributes.get('material')} not found")
        ela = next(mat.find_all("Elastic"))
        E, nu = map(float, ela.data[0].split(","))

        shape_name = section.attributes.get("section", None)
        if shape_name is not None:
            shape_name = shape_name.upper()

        assert shape_name in {"RECT", "L", "T", "C", "I", "ARBITRARY", "BOX", "CIRC"}
        thickness = section.attributes.get("thickness", None)
        tag = conv.define("FrameSection", "section", elset)


        from xsection.library import Angle, WideFlange, Rectangle, Channel, HollowRectangle, Circle
        if shape_name == "RECT":
            a,b = map(float, section.data[0].split(","))
            shape = Rectangle(a,b)
        elif shape_name == "BOX":
            a, b, t1, t2, t3, t4 = map(float, section.data[0].split(","))
            shape = HollowRectangle(a, b, t1, t2)
        elif shape_name == "CIRC":
            r = float(section.data[0])
            shape = Circle(r)
        elif shape_name == "L":
            a, b, t1, t2 = map(float, section.data[0].split(","))
            shape = Angle(a, b, t1, t2)
        elif shape_name == "T":
            shape = WideFlange(thickness, thickness, 0.1*thickness)
        elif shape_name == "C":
            shape = Channel(thickness, thickness, 0.1*thickness)
        elif shape_name == "I":
            l, h, b1, b2, t1, t2, t3, t4 = map(float, section.data[0].split(","))
            shape = WideFlange(thickness, thickness, 0.1*thickness)

        model.section("FrameElastic", tag, 
                      E=E,
                      G=E/(2*(1 + nu)),
                      A=shape.elastic.A,
                      Iy=shape.elastic.Iy,
                      Iz=shape.elastic.Iz,
                      J=shape.elastic.J or (
                        shape.elastic.Iy + shape.elastic.Iz),
                      )

        n1 = tuple(map(float, re.split(",\\W*", section.data[1].strip())))
        model.geomTransf("Linear", tag, n1)
        # n2 = 


def _create_part(part, model, conv, root=None, verbose=False, mode=None, inst=None):
    if mode is None:
        mode = "simulate"

    transforms = {}

    # Create Materials

    ast = part._tree
    
    if mode == "visualize":
        # Create dummy materials to make the model work
        E = 29e3
        nu = 0.27
        density = 1.27

        mat = 1
        fsec = 1
        model.material("ElasticIsotropic", mat, E, nu)
        model.section("ElasticShell",        1, E, nu, 1.175, density)
        model.section("FrameElastic", fsec, E=E, G=E*0.6, A=1, Iy=1, Iz=1, J=1)

        model.geomTransf("Linear", 1, (0.0, 1.0, 0))



    for node in part.find_nodes(recurse=False):
        tag = conv.define("Node", "node",
                            node.id, group=part.name)
        try:
            model.node(tag, tuple(inst.rotate(inst.translate(node.location))))
        except Exception as e:
            print(f"Error creating node {node.id} (tag {tag}):")
            raise e

    # TODO: Nodegen


    if False:
        if node.keyword == "Load":
            for child in node.children:
                load_data = child.attributes.get("data").split(",")
                node_id = int(load_data[0])
                load_values = list(map(float, load_data[1:]))
                model.load(node_id, *load_values)

    #
    # Create elements
    #
    for cell in part.find_cells():
        nodes = tuple(
            conv.identify("Node", "node", node_id, group=part.name)
            for node_id in cell.nodes
        )
        tag = conv.define("Element", "element", cell.id, group=part.name)

        if cell.type not in _Elements and mode != "visualize":
            raise ValueError(f"Element type {cell.type} not supported")
        if cell.type not in _Elements:
            warnings.warn(f"Element type {cell.type} not supported, skipping")
            continue

        element = _Elements[cell.type]

        if element.xara_name is None:
            if element.cell in   {"line", "line2"}:
                name = "ForceFrame"
            elif element.cell in {"line3", "line4"}:
                name = "ExactFrame"
            elif element.cell in {"quad", "quad4"}:
                name = "ASDShellQ4"
            elif element.cell in {"triangle"}:
                name = "ShellDKGT"
            elif element.cell in {"hexahedron"}:
                name = "stdBrick"
            elif element.cell in {"tetra", "tetra4"}:
                name = "FourNodeTetrahedron"
            else: 
                if mode != "visualize":
                    raise ValueError(f"Element {cell.type} does not have a Xara name")
                else:
                    continue

            warnings.warn(f"Element {cell.type} does not have a Xara name, using default {name}")

        else:
            name = element.xara_name

        if element.cell not in _SolidElements:
            transform = None
            if cell.section is None:
                if mode == "visualize":
                    section = 1
                    # raise ValueError(f"Element {cell.type} does not have a section")
                    # continue
                else:
                    raise ValueError(f"Element {cell.type} does not have a section")

            elif "line" in element.cell:
                section = conv.identify("FrameSection", "section", cell.section.name, group=part.name)
                transform = section #conv.identify("FrameSection", "geomTransf", cell.section.name, group=part.name)
            elif "quad" in element.cell or "triangle" in element.cell:
                section = conv.identify("ShellSection", "section", cell.section.name, group=part.name)
                transform = None
            else:
                raise ValueError(f"Element type {cell.type} does not have a section")

            if section is None and mode != "visualize":
                raise ValueError(f"Section {cell.section} not found for element {cell.type}")
            elif section is None:
                warnings.warn(f"Section {cell.section} not found for element {cell.type}, using default")
                continue
        
            if transform is not None:
                model.element(name, 
                            tag,
                            nodes,
                            section=section,
                            transform=transform
                )
            else:
                model.element(name, 
                            tag,
                            nodes,
                            section=section
                )
            continue

        else:
            if cell.material is None:
                if mode == "visualize":
                    material = 1
                else:
                    raise ValueError(f"Element {cell.type} does not have a material")
            else:
                material = conv.identify("Material", "material", cell.material, group=part.name)
                if material is None and mode != "visualize":
                    raise ValueError(f"Material {cell.material} not found for element {cell.type}")

            model.element(name, 
                          tag,
                          nodes,
                          material
            )
            continue


    return model, conv


def _create_boundaries(ast, model, conv):

    for block in ast.find_all("Boundary"):
        for line in block.data:
            print(line)
            try:
                boundary_data = json.loads("["+line+"]") # line.split(",")
                dofs = tuple(map(int, boundary_data[1:]))
            except:
                print("WARNING ", line, file=sys.stderr)
                continue
            if len(dofs) > 1:
                dofs = tuple(range(dofs[0], dofs[1]+1))

            try:
                nodes = (int(boundary_data[0]), )
            except:
                # its a set name
                nodes = (
                    int(i)
                    for row in ast.find_attr("Nset", nset=boundary_data[0]).data
                    for i in row.split(",")
                )
            nodes = list(nodes)

            for node in nodes:
                for dof in dofs:
                    model.fix(node, dof=dof)
