"""
job = xcae.read("a.inp")

job.run()

mod = xcae.Model(job.assembly)

"""
import sys
import xara
from .part import Part
from .instance import Instance, create_sections, create_materials
from .context import Context

def read(filename):
    pass


class Assembly:

    def __init__(self, root, tree, mode=None, **kwds):
        self._root = root
        self._tree = tree
        self._mode = mode
        self._instances = []
        self._model = self._create_model(**kwds)

    @property 
    def model(self):
        return self._model

    def _create_model(self, mode=None, export=None):
        if mode is None:
            mode = self._mode if self._mode is not None else "simulate"

        if export is not None:
            export_file = open(export, "w+")
        else:
            export_file = None
        model = xara.Model(ndm=3, ndf=6, echo_file=export_file)

        context = Context()

        model.eval("\n#")
        model.eval("# Materials")
        model.eval("#")
        create_materials(self._root, model, context)

        model.eval("\n#")
        model.eval("# Sections")
        model.eval("#")
        create_sections( self._root, model, context)

        # 1) Create Instances
        model.eval("\n#")
        model.eval("# Instances")
        model.eval("#")
        for instance in self._tree.find_all("Instance"):
            part_name = instance.attributes["part"]
            part = self._root.find_attr("Part", name=part_name)
            model.eval("")
            model.eval(f"# Part name={part_name}")
            inst = Instance(model, 
                            Part(part,root=self._root),
                            tree=instance,
                            context=context,
                            name=instance.attributes.get("name",None),
                            root=self._root,
                            mode=mode
                    )
            
            self._instances.append(inst)

        # 2) Add standalone part components
        if len(self._instances) == 0:
            self._instances.append(
                Instance(model,
                        Part(self._tree, root=self._root),
                        tree=self._tree,
                        context=context,
                        name=self._tree.attributes.get("name", None),
                        root=self._root,
                        mode=mode
                )
            )

        return model


class Step:
    def __init__(self, job, model):
        self._job = job

    def run(self):
        model = self._model


class Job:
    """
    A Job is a collection of parts, assemblies, and instances.
    It is used to run simulations or visualizations.
    """
    def __init__(self, tree, name=None, mode=None):
        self._mode = mode if mode is not None else "simulate"
        self._name = name
        self._tree = tree
        self._assemblies = []

    def assemble(self, export=None):
        try:
            atree = next((self._tree.find_all("Assembly")))
        except StopIteration:
            atree = self._tree

        asm = Assembly(root=self._tree, 
                       tree=atree,
                       mode=self._mode,
                       export=export
        )
        return asm 
    
    def run(self):
        # 1) Create the model
        asm = self.assemble()
        model = asm._model
        # 2) Run the steps
        for step in self._tree.find_all("Step"):
            step = Step(self, model)
            step.run()



if __name__ == "__main__":
    import veux

    from .parser import load

    if len(sys.argv) < 2:
        print("Usage: xcae job.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    tree = load(filename, verbose=False)

    job = Job(tree, mode="simulate")

    m = job.assemble()._model
    # artist = veux.create_artist(m, vertical=3)
    # artist.draw_surfaces()
    artist = veux.render(m, vertical=3)
    veux.serve(artist)
