import veux
import numpy as np
def show_outline(shape):
    canvas = veux._create_canvas(name="gltf")
    ext = shape.exterior()
    ext3d = np.zeros((len(ext)+1, 3))
    ext3d[:-1, :2] = ext
    ext3d[-1, :2] = ext[0]
    canvas.plot_lines(ext3d)
    for hole in shape.interior():
        canvas.plot_lines(hole)
    
    return canvas


if __name__ == "__main__":

    import sys 
    from xsection.library import from_aisc
    canvas = show_outline(from_aisc(sys.argv[1]))
    if len(sys.argv) > 2:
        canvas.write(sys.argv[2])
    else:
        veux.serve(canvas)
