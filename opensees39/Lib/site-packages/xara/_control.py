
if __name__ == "__main__":
    import sys
    import xara

    model = xara.Model()
    loads = [xara.NodalLoad(node=1, dof=1, value=1.0)]
    trace = xara.Trace(model, loads, method=xara.ArcLengthControl())

    while model.nodeDisp(1,1) < 0.1:
        trace.step()
        print(model.nodeDisp(1,1))