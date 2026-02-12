
import re
import sys
import json
from functools import reduce
from dataclasses import dataclass
import numpy as np


class _Elset:
    def __init__(self, name, tree):
        self.name = name
        self._tree = tree
        if "generate" not in tree.attributes:
            self._tags = reduce(lambda x, y: x.union(y),
                (map(lambda i: int(i) if i else None, re.split(",\\W*", line.strip())) for line in tree.data), set())
        else:
            self._tags = None

    def __contains__(self, item):
        if self._tags is not None:
            return item in self._tags
        if "generate" in self._tree.attributes:
            return any(item in range(*map(int, re.split(",\\W*", line.strip()))) for line in self._tree.data)
        

    def __iter__(self):
        return iter(self.tags)

    def __len__(self):
        return len(self.tags)

    def __repr__(self):
        return f"Elset({self.name}, {self.tags})"

class Part:
    """
    """
    def __init__(self, ast, model=None, root=None):
        self._tree = ast
        self._root = root if root is not None else ast
        self._model = model
        self._elsets = {
            i.attributes.get("elset"): _Elset(i.attributes.get("elset"), i)
            for i in ast.find_all("Elset", recurse=False)
        }
        self._nsets = {
            i.attributes.get("nset"): reduce(
                lambda x, y: x.union(y),
                (map(lambda i: int(i) if i else None, re.split(",\\W*", line.strip())) for line in i.data), 
                set())
            for i in ast.find_all("Nset", recurse=False)
        }

    @property 
    def name(self):
        return self._tree.attributes.get("name", None)
    
    def find_nodes(self, recurse=False, **kwds):
        @dataclass
        class Node:
            id: int
            location: tuple

        for block in self._tree.find_all("Node", recurse=recurse):
            for line in block.data:
                node_data = line.split(",")
                node_id = int(node_data[0])
                coords = tuple(map(float, node_data[1:]))
                if len(coords) == 2:
                    coords = (coords[0], coords[1], 0.0)
                elif len(coords) == 1:
                    coords = (coords[0], 0.0, 0.0)
                yield Node(node_id, coords)
        

        if self._model is None:
            return

        for nodes in self._tree.find_all("Ngen"):
            nset = nodes.attributes.get("nset", None)

            i_tag, j_tag = map(int, nodes.data[0].split(","))

            i_x = self._model.nodeCoord(i_tag)
            j_x = self._model.nodeCoord(j_tag)

            n  = j_tag - i_tag - 1

            for node_id, coords in zip(range(i_tag+1, j_tag), np.linspace(i_x, j_x, n)):
                coords = tuple(coords.tolist())
                if len(coords) == 2:
                    coords = (coords[0], coords[1], 0.0)
                elif len(coords) == 1:
                    coords = (coords[0], 0.0, 0.0)
                yield Node(node_id, coords)

    def _find_material(self, node):
        """
        Find a material by name in the part.
        """
        @dataclass
        class Material:
            name: str
            type: str
            density: float = None
            young_modulus: float = None
            poisson_ratio: float = None
            hardening: str = None
            yield_stress: float = None
            plastic_modulus: float = None
            Ep = None  # Plastic modulus for isotropic hardening ?


        m = Material(
            name=node.attributes.get("name"),
            type=node.attributes.get("type", "Elastic")
        )
        for child in node.children:
            if child.keyword == "Density":
                m.density = float(child.data[0].split(",")[0])
                continue

            if child.keyword == "Elastic":
                properties = child.data[0].split(",")
                m.young_modulus = float(properties[0])
                try:
                    m.poisson_ratio = float(properties[1])
                except:
                    pass
                continue

            elif child.keyword == "Plastic":
                m.hardening = child.attributes.get("hardening", "isotropic")
                properties = child.data[0].split(",")
                if m.hardening == "isotropic":
                    m.yield_stress = float(properties[0])
                    m.Ep = float(properties[1])
                elif m.hardening == "JOHNSON COOK":
                    pass

            elif child.keyword == "Concrete":
                continue
                tag = conv.define("Material", "uniaxial", node.attributes.get("name"))
                properties = child.children[0].attributes.get("data").split(",")
                f_c = float(properties[0])  # Compressive strength
                f_t = float(properties[1])  # Tensile strength
                model.uniaxialMaterial("Concrete", tag, f_c, f_t)

        return m

    def _find_section(self, elset):
        @dataclass
        class Section:
            name: str
            type: str
            material: str = None
            integration: str = None
            orientation: tuple = None

        orient = None
        type = None
        material = None
        section = self._tree.find_attr("Shell Section", elset=elset) 
        if section is None:
            section = self._root.find_attr("Shell Section", elset=elset)
            type = "Shell" if section is not None else None
        else:
            type = "Shell"

        if section is None:
            section = self._root.find_attr("Beam Section", elset=elset)
            if section is None:
                return None
            orient = tuple(map(float, section.data[1].split(",")))


        mnode = self._root.find_attr("Material", name=section.attributes.get("material", None))
        if mnode is not None:
            material = self._find_material(mnode)

        name = section.attributes.get("name", section.attributes.get("elset", None))
        return Section(
          name = name,
          type = type,
          material = material,
          orientation = orient
        )

    def find_cells(self, **kwds):
        @dataclass
        class Cell:
            id: int
            nodes: tuple
            type: str
            material: str
            section: str
            orient: str = None
            tree: "AbaqusTable" = None


        for block in self._tree.find_all("Element"):
            elset = block.attributes.get("elset", None)
            material = None
            section  = self._find_section(elset)

            
            for tag, *nodes in _iter_nodes(block):
                if elset is None:
                    for elset, tags in self._elsets.items():
                        if int(tag) in tags:
                            elset = elset
                            break
                    else:
                        elset = None
                if section is None:
                    section = self._find_section(elset)

                yield Cell(tag, 
                           tuple(nodes), 
                           block.attributes.get("type", block.attributes.get("TYPE", None)), 
                           None, 
                           section,
                           None,
                           block)


def _iter_nodes(block, n=None):
    if n is None:
        for line in block.data:
            if line[-1] == ",":
                line = line[:-1]
            yield map(int, re.split(",\\W*", line.strip())) # line.split(","))
    else:
        lines = iter(block.data)
        nodes = []
        while True:
            if len(nodes) == n+1:
                yield tuple(nodes)
                nodes = []
            try:
                line = next(lines)
            except StopIteration:
                if len(nodes) > 0:
                    yield tuple(nodes)
                break
            if line[-1] == ",":
                line = line[:-1]
            nodes.extend(map(int, re.split(",\\W*", line.strip())))

