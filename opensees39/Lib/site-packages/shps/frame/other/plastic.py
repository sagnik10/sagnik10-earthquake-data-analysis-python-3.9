


def plastic_surface(shape, axis, fy=1):
    points = {
        i: {"area": fiber.area, "location": fiber.location, "lever": axis.dot(fiber.location)}
        for i, fiber in enumerate(shape.fibers())
    }
    nIP = len(points)
    # Qm = np.array([[fiber.lever, fiber.area] for fiber in points.values()])
    Yp =  fy 
    Yn = -fy
    states = [
        [[[Yp, Yn][int(j>i)] for i in range(1,nIP+1)] for j in range(1,1+nIP)]
    ]