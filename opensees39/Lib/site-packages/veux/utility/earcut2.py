"""
"""
import numpy as np


def _is_clockwise(polygon):
    """
    Check if a polygon is defined in clockwise order.
    """
    return np.sum((polygon[1:, 0] - polygon[:-1, 0]) * (polygon[1:, 1] + polygon[:-1, 1])) > 0

def _is_convex(a, b, c):
    """
    Check if three points make a convex corner.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0

def _is_ear(polygon, i):
    """
    Check if a vertex at index `i` forms an ear in the polygon.
    """
    prev_idx = (i - 1) % len(polygon)
    next_idx = (i + 1) % len(polygon)
    p_prev = polygon[prev_idx]
    p_curr = polygon[i]
    p_next = polygon[next_idx]

    # Triangle formed by the current vertex and its neighbors
    ear_triangle = np.array([p_prev, p_curr, p_next])

    # Check if the triangle is convex
    if not _is_convex(p_prev, p_curr, p_next):
        return False

    # Check if any other vertex lies inside the triangle
    for j, point in enumerate(polygon):
        if j not in [prev_idx, i, next_idx]:
            if _point_in_triangle(point, ear_triangle):
                return False

    return True

def _point_in_triangle(p, triangle):
    """
    Check if a point `p` is inside the triangle defined by three vertices.
    """
    def sign(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    a, b, c = triangle
    b1 = sign(p, a, b) < 0
    b2 = sign(p, b, c) < 0
    b3 = sign(p, c, a) < 0

    return b1 == b2 == b3


def earcut2(vertices):
    """
    Triangulate a concave polygon using the ear clipping algorithm.

    Parameters:
        polygon_vertices (array-like): List or numpy array of (x, y) coordinates defining the polygon.

    Returns:
        list of tuple: Indices of triangles as (i, j, k) referencing the input vertices.
    """
    # Coerce input to numpy array
    vertices = np.asarray(vertices, dtype=float)

    # Remove duplicate last point if it's the same as the first
    if np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]

    if len(vertices) < 3:
        raise ValueError("A polygon must have at least 3 vertices to triangulate.")

    # Ensure the polygon is in counter-clockwise order
    if _is_clockwise(vertices):
        vertices = vertices[::-1]

    # Copy vertices and initialize the triangle list
    polygon = vertices.tolist()
    triangles = []

    # Loop until the polygon is reduced to three vertices
    while len(polygon) > 3:
        for i in range(len(polygon)):
            if _is_ear(vertices, i):
                # Add the ear triangle to the result
                prev_idx = (i - 1) % len(polygon)
                next_idx = (i + 1) % len(polygon)
                triangles.append((prev_idx, i, next_idx))

                # Remove the ear vertex from the polygon
                del polygon[i]
                break

    # Add the last remaining triangle
    triangles.append((0, 1, 2))

    return triangles


def earcut3(vertices):
    """
    Triangulate a simple polygon using the ear clipping algorithm.

    Parameters:
    polycoord (list or ndarray): A list or array of [x, y] coordinates of the polygon vertices in order.

    Returns:
    triangles (ndarray): An (n, 3) array of indices into polycoord, each row representing a triangle.
    """

    def is_convex(a, b, c):
        """Check if the angle formed by points a, b, c is convex."""
        return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) < 0

    def point_in_triangle(a, b, c, p):
        """Check if point p is inside triangle abc."""
        # Barycentric coordinate method
        detT = (b[1] - c[1])*(a[0] - c[0]) + (c[0] - b[0])*(a[1] - c[1])
        if detT == 0:
            return False
        l1 = ((b[1] - c[1])*(p[0] - c[0]) + (c[0] - b[0])*(p[1] - c[1])) / detT
        l2 = ((c[1] - a[1])*(p[0] - c[0]) + (a[0] - c[0])*(p[1] - c[1])) / detT
        l3 = 1 - l1 - l2
        return (0 < l1 < 1) and (0 < l2 < 1) and (0 < l3 < 1)

    # Convert to numpy array
    vertices = np.asarray(vertices)

    # Remove duplicate last point if it's the same as the first
    if np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]

    n = len(vertices)
    if n < 3:
        raise ValueError("A polygon must have at least 3 vertices.")

    # Ensure the polygon is counter-clockwise
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += (vertices[j][0] - vertices[i][0]) * (vertices[j][1] + vertices[i][1])
    if area > 0:
        vertices = vertices[::-1]

    indices = list(range(len(vertices)))
    triangles = []

    while len(indices) > 3:
        ear_found = False
        for i in range(len(indices)):
            prev_idx = indices[i - 1]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % len(indices)]

            a = vertices[prev_idx]
            b = vertices[curr_idx]
            c = vertices[next_idx]

            if is_convex(a, b, c):
                # Check if any other vertex is inside the triangle abc
                ear = True
                for idx in indices:
                    if idx in (prev_idx, curr_idx, next_idx):
                        continue
                    p = vertices[idx]
                    if point_in_triangle(a, b, c, p):
                        ear = False
                        break
                if ear:
                    triangles.append([prev_idx, curr_idx, next_idx])
                    del indices[i]
                    ear_found = True
                    break

        if not ear_found:
            raise ValueError("No ear found. The polygon might be not simple or is degenerate.")

    # Add the last remaining triangle
    triangles.append([indices[0], indices[1], indices[2]])

    return np.array(triangles)


def earcut4(polygon, tolerance=1e-6):
    """
    Improved Ear Clipping algorithm with pre-sorting and tolerance for degenerate cases
    """
    from shapely import Polygon
    if isinstance(polygon, Polygon):
        vertices = list(polygon.exterior.coords)[:-1]  # Ignore the last point because it's a repetition of the first
    else:
        vertices = polygon
        polygon = Polygon(vertices)

    if vertices[-1][0] == vertices[0][0] and vertices[-1][1] == vertices[0][1]:
        vertices = vertices[:-1]
    triangles = []

    # Sort vertices by x-coordinate
    vertices = sorted(vertices, key=lambda vertex: vertex[0])

    while len(vertices) > 3:
        for i in range(len(vertices)):
            a, b, c = vertices[i - 1], vertices[i], vertices[(i + 1) % len(vertices)]
            triangle = Polygon([a, b, c])

            # Adjusted checks for valid "ear"
            if triangle.is_valid and triangle.area > tolerance and triangle.contains(polygon):
                triangles.append(triangle)
                vertices.pop(i)
                break
        else:
            raise Exception("No valid ear found")

    if len(vertices) == 3:
        triangles.append(Polygon(vertices))

    return triangles

