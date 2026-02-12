#===----------------------------------------------------------------------===#
#
#         STAIRLab -- STructural Artificial Intelligence Laboratory
#
#===----------------------------------------------------------------------===#
#
import sys
import veux
import numpy as np
import xcae
import json

def analyze_step(model, step):

    loads = []
    for load in step.find_all("Cload"):
        for line in load.data:
            try: 
                data = json.loads("[" + line + "]")
            except:
                continue 
            loads.append(data)
    
    model.pattern("Plain", 1, "Linear", loads={
        item[0]: [0 if item[1] != i+1 else item[2] for i in range(6)]
        for item in loads
    })

    for _ in step.find_all("Static"):
        model.analysis("Static")
    
    model.analyze(1)


if __name__ == "__main__":

    file_name = sys.argv[2]
    doc = xcae.parser.load(file_name, verbose=False)

    if sys.argv[1] == "-p":
        if len(sys.argv) > 3:
            doc = next(doc.find_all(sys.argv[3]))

        print(doc)
        sys.exit(0)


    model = xcae.create_model(doc, verbose=True, mode="visualize")


    if sys.argv[1] == "-C":
        # Convert
        model.print("-json")

    elif sys.argv[1] == "-E":
        # Eigen
        model.constraints("Transformation")
        W = model.eigen(2)
        for w in W:
            print(f"T = {2*np.pi/np.sqrt(w)}")
        veux.serve(veux.render_mode(model, 1, 200.0, vertical=3, canvas="gltf"))

    elif sys.argv[1] == "-X":
        # Execute 
        for step in doc.find_all("Step"):
            analyze_step(model, step)
            artist = veux.create_artist(model, vertical=3)
            artist.draw_outlines(state=model.nodeDisp)
            veux.serve(artist)
            
    elif sys.argv[1][:2] == "-A":
        # Apply loads and analyze
        from xara.helpers import find_nodes, node_average
        top = max(model.nodeCoord(tag)[2] for tag in model.getNodeTags())
        model.pattern("Plain", 1, "Linear")
        for node in find_nodes(model, z=top):
            model.load(node, (0, -1/2, -1, 0, 0, 0))

        # model.integrator("LoadControl", 10.0)
        
        model.algorithm("Linear")
        model.system("UmfPack")
        model.numberer("RCM")
        model.analysis("Static")

        print("Starting analysis")
        model.analyze(1)
        print("Analysis complete")
        if "s" in sys.argv[1]:
            field = {node: stress["sxx"] for node, stress in node_average(model, "stressAtNodes")}
        else:
            field = None
        # artist = veux.render(model, model.nodeDisp, scale=1.0, vertical=3, hide={"node.marker"})
        artist = veux.create_artist(model, vertical=3)
        artist.draw_outlines()
        artist.draw_surfaces(state=model.nodeDisp, field=field, scale=1.0)
        veux.serve(artist)

    elif sys.argv[1] == "-V":
        # Visualize
        import veux
        # veux.serve(veux.render(model, canvas="gltf", vertical=3, hide={"node.marker"}))
                # artist = veux.render(model, model.nodeDisp, scale=1.0, vertical=3, hide={"node.marker"})
        artist = veux.create_artist(model, vertical=3)
        artist.draw_outlines()
        artist.draw_surfaces()
        veux.serve(artist)

    elif sys.argv[1] == "-Vn":
        # Visualize
        from scipy.linalg import null_space
        model.constraints("Transformation")
        model.analysis("Static")
        K = model.getTangent().T
        v = null_space(K)[:,0] #, rcond=1e-8)
        print(v)


        u = {
            tag: [1000*v[dof-1] for dof in model.nodeDOFs(tag)]
            for tag in model.getNodeTags()
        }

        import veux
        veux.serve(veux.render(model, u, canvas="gltf", vertical=3))

    elif sys.argv[1] == "-Q":
        # Quiet conversion
        pass
    else:
        raise ValueError(f"Unknown operation {sys.argv[1]}")

