import sys
import json
from pathlib import Path

import bottle
from sees.config import Canvas

import numpy as np

class FemGlCanvas(Canvas):
    def __init__(self, model, config=None):
        self.mesh = {}
        self.model = model
        self.config = config
        self.annotations = []

    def build(self):
        self.mesh = _model2mesh(self.model)

    def show(self):
        self.build()

        # directory
        idir = Path(__file__).parents[0]

        with open(idir/"index.html", "r") as f:
            page = f.read()


        app = bottle.Bottle()
        app.route("/")(lambda : page ) # bottle.template(page, mesh=self.mesh))
        app.route("/mesh.json")(lambda : self.mesh) #json.dumps(self.mesh))
        bottle.run(app, host="localhost", port=8080)


def _model2mesh(data):
    p8 = [
            list(map(str, e["nodes"]))
            for e in data["assembly"].values() if len(e["nodes"]) == 8
    ] + [
            list(map(str, e["nodes"])) + list(map(str, e["nodes"]))
            for e in data["assembly"].values() if len(e["nodes"]) == 4
    ]

    p6 = [
            list(map(str, e["nodes"]))
            for e in data["assembly"].values() if len(e["nodes"]) == 6
    ] + [
            list(map(str, e["nodes"])) + list(map(str, e["nodes"]))
            for e in data["assembly"].values() if len(e["nodes"]) == 3
    ]

    return  {
                "coordinates":   {str(name): list(map(float,node["crd"])) for name,node in data["nodes"].items()},
                "displacements": {str(name): [0.0]*3 for name in data["nodes"]},
                "elements": [
                    {
                        "type": "P8",
                        "cells": p8,
                        "stresses": [0.0]*len(p8)
                    },
                    {
                        "type": "P6",
                        "cells": p6,
                        "stresses": [0.0]*len(p6)
                    },
                ],
                "palette": [
                  [0.0, 0.0, 1.0],
                  [0.0, 0.3999999999999999, 1.0],
                  [0.0, 0.55, 1.0],
                  [0.0, 0.6833333333333329, 1.0],
                  [0.0, 0.8166666666666664, 1.0],
                  [0.0, 0.95, 1.0],
                  [0.0, 1.0, 0.8999999999999998],
                  [0.0, 1.0, 0.7666666666666669],
                  [0.0, 1.0, 0.6333333333333334],
                  [0.0, 1.0, 0.4833333333333333],
                  [0.0666666666666671, 1.0, 0.0],
                  [0.6166666666666665, 1.0, 0.0],
                  [0.7499999999999993, 1.0, 0.0],
                  [0.8833333333333335, 1.0, 0.0],
                  [1.0, 0.9666666666666666, 0.0],
                  [1.0, 0.833333333333333, 0.0],
                  [1.0, 0.7000000000000002, 0.0],
                  [1.0, 0.5666666666666667, 0.0],
                  [1.0, 0.4166666666666665, 0.0],
                  [1.0, 0.0, 0.0]]
            }

def _ops2mesh(data):

    return json.dumps(
            {
                "coordinates":   {str(n["name"]): n["crd"] for n in data["geometry"]["nodes"]},
                "displacements": {str(n["name"]): [0.0]*3 for n in data["geometry"]["nodes"]},
                "elements": [{
                    "type": "P8",
                    "cells": [list(map(str, e["nodes"])) for e in data["geometry"]["elements"]],
                    "stresses": [0.0]*len(data["geometry"]["elements"])
                }],
                "palette": [
                  [0.0, 0.0, 1.0],
                  [0.0, 0.3999999999999999, 1.0],
                  [0.0, 0.55, 1.0],
                  [0.0, 0.6833333333333329, 1.0],
                  [0.0, 0.8166666666666664, 1.0],
                  [0.0, 0.95, 1.0],
                  [0.0, 1.0, 0.8999999999999998],
                  [0.0, 1.0, 0.7666666666666669],
                  [0.0, 1.0, 0.6333333333333334],
                  [0.0, 1.0, 0.4833333333333333],
                  [0.0666666666666671, 1.0, 0.0],
                  [0.6166666666666665, 1.0, 0.0],
                  [0.7499999999999993, 1.0, 0.0],
                  [0.8833333333333335, 1.0, 0.0],
                  [1.0, 0.9666666666666666, 0.0],
                  [1.0, 0.833333333333333, 0.0],
                  [1.0, 0.7000000000000002, 0.0],
                  [1.0, 0.5666666666666667, 0.0],
                  [1.0, 0.4166666666666665, 0.0],
                  [1.0, 0.0, 0.0]]
            }
    )

if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        mesh = json.load(f)["StructuralAnalysisModel"]

    print(_ops2mesh(mesh))

