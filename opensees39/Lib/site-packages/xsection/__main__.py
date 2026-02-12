from shps.frame.solvers import TriangleModel
from xsection.warping import WarpingSection
from xsection.library import from_aisc
from veux.config import NodeStyle
import matplotlib.pyplot as plt
import sys
import json
import veux

def main():
    file = None
    file_format = "json"
    operation = "Summarize"
    section = None

    argi = iter(sys.argv[1:])
    for arg in argi:
        if arg == "-":
            file = sys.stdin
        elif arg == "-l":
            section = from_aisc(next(argi))
        elif arg[0] == "-":
            operation = {
                "e": "Export",
                "v": "View"
            }[arg[1].lower()]
        elif file is None:
            file = open(arg, "r")
            file_format = arg.split(".")[-1].lower()

    if section is None:
        data = json.load(file)
        if "StructuralAnalysisModel" in data:
            model = TriangleModel.from_xara(data)
            section = WarpingSection(model)
        else:
            from xsection._io import _from_dict
            section = _from_dict(data)



    if operation == "Export":
        export_format = "json"
        argi = iter(sys.argv[1:])
        for arg in argi:
            if arg == "-f":
                export_format = next(argi)

        from xsection._io import export_fedeas
        print(export_fedeas(section, "General"))
        sys.exit(0)

    elif "View" in operation:
        section = section.translate(-section.centroid)
        sc = section._analysis.shear_center()
        print(sc)
        # section = section.translate(-sc)

        print(section.summary(shear=True))

        artist = veux.create_artist(section.model, ndf=1)

        artist.draw_origin(extrude=True)
        artist.draw_surfaces()

        # Rc = artist._plot_rotation.T
        # for point in section.exterior():
        #     artist.canvas.plot_nodes([Rc@[*point, 0]], style=NodeStyle(color="blue", scale=50))

        # for point in section.interior()[0]:
        #     artist.canvas.plot_nodes([Rc@[*point, 0]], style=NodeStyle(color="red", scale=50))

        # w = section._analysis.shear_warping()[1]
        # artist.draw_surfaces(field=w, state=w/w.max()*section.depth/5)
        # artist.draw_outlines()
        veux.serve(artist)

if __name__ == "__main__":
    main()
