
import sys
from pathlib import Path
from itertools import chain

_PREAMBLE = r"""

"""


class Evaluation:
    def __init__(self, paths, exclusions=None):
        self._exclusions = exclusions if exclusions is not None else []
        self._paths = paths


    def run(self):
        for path in self._paths:
            if not path.is_file():
                continue
            print(path)
            try:
                TestCase(path, self).run()

            except Exception as e:
                print(f"Error running test case {path}: {e}")
                raise e
                continue


class TestCase:
    def __init__(self, file, evaluation=None):
        self._file = file
        self._evaluation = evaluation


    def run(self):
        from opensees import tcl
        with open(self._file, "r") as f:
            script = _PREAMBLE + f.read()

        interp = tcl.Interpreter()

        def verify(*args):
            # interp.eval("puts {Verifying: " + " ".join(str(arg) for arg in args) + "}")
            objects = {
                "element": set(),
                "section": set(),
                "material": set()
            }

            if args[1] == "value":
                try:
                    model = interp.serialize()["StructuralAnalysisModel"]
                    for element in model["geometry"].get("elements", []):
                        objects["element"].add(element["type"])

                    for section in model["properties"].get("SectionForceDeformation", []):
                        objects["section"].add(section["type"])

                except Exception as e:
                    print(f"Error serializing model: {e}")
                    pass


            interp.eval("OpenSees::verify " + " ".join(f'"{arg}"' for arg in args))

        interp._tcl.createcommand("verify", verify)

        interp.eval(script)
        return {"status": "success"}

def run(args):

    if len(args) == 1:
        files = chain(
                  Path(".").rglob("test*.tcl"),
                  Path(".").rglob("o-*.tcl"),
                  Path(".").rglob("x-*.tcl")
                )
    else:
        files = (Path(arg) for arg in args[1:])

    Evaluation(files).run()
    return



if __name__ == "__main__":

    run(sys.argv)
