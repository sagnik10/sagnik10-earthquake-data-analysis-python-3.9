import xara
from collections import defaultdict

class _Parameter(float):
    __slots__ = ("index", "name")

    def __new__(cls, value, index=0, name=None):
        obj = super().__new__(cls, value)
        obj.index = int(index)
        obj.name  = str(name)
        return obj

    def __repr__(self):
        return f"{{Parameter {self.index} {super().__repr__()}}}"



class ParaModel:
    """
    A wrapper around xara.Model that allows us to use JAX's vjp/jvp
    with the model's methods.
    """
    def __init__(self, *args, parameters=0, **kwargs):
        self._model = xara.Model(*args, **kwargs)
        for i in range(parameters):
            self._model.parameter(i)

        self._p = defaultdict(lambda: {
            "material": set(),
            "section":  set(),
            "element":  set()
        })


    def __getattr__(self, name):
        return getattr(self._model, name)


    def analyze(self, *args, **kwargs):
        self.sensitivityAlgorithm("-computeAtEachStep")
        for (index,name),param in self._p.items():
            for level in "element",:
                for tag in param[level]:
                    self._model.addToParameter(index, level, tag, name)

        return self._model.analyze(*args, **kwargs)


    def _pull_parameters(self, level, tag, args, kwds):
        new_args = []
        new_kwds = {}
        for arg in args:
            if isinstance(arg, _Parameter):
                self._p[(arg.index,arg.name)][level].add(tag)
                new_args.append(float(arg))
            else:
                new_args.append(arg)

        for key, arg in kwds.items():
            if isinstance(arg, _Parameter):
                self._p[(arg.index,arg.name)][level].add(tag)
                new_kwds[key] = float(arg)
            else:
                new_kwds[key] = arg

        if "material" in kwds:
            for param in self._p.values():
                if kwds["material"] in param["material"]:
                    param[level].add(tag)

        if "section" in kwds:
            for param in self._p.values():
                if kwds["section"] in param["section"]:
                    param[level].add(tag)

        return new_args, new_kwds

    def section(self, *args, **kwds):
        tag = args[1]
        args, kwds = self._pull_parameters("section", tag, args, kwds)
        return self._model.section(*args, **kwds)

    def material(self, *args, **kwds):
        tag = args[1]
        args, kwds = self._pull_parameters("material", tag, args, kwds)
        return self._model.material(*args, **kwds)
    
    def element(self, *args, **kwds):
        tag = args[1]
        args, kwds = self._pull_parameters("element", tag, args, kwds)
        return self._model.element(*args, **kwds)

try:
    from jax import tree_util
    def _flatten_model(model):
        return (), model

    def _unflatten_model(aux_data, _children):
        return aux_data

    tree_util.register_pytree_node(ParaModel, _flatten_model, _unflatten_model)
except ImportError:
    # JAX is not available, so we won't register the pytree node
    pass
