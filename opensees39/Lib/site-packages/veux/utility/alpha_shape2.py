"""
Adapted from https://github.com/bellockk/alphashape/
"""

import sys
import itertools
import shapely
from shapely.geometry import MultiPoint
from typing import Union, Tuple, List
import warnings
from scipy.spatial import Delaunay
import numpy as np

def _testalpha(points: Union[List[Tuple[float]], np.ndarray], alpha: float):
    """
    Evaluates an alpha parameter.

    This helper function creates an alpha shape with the given points and alpha
    parameter.  It then checks that the produced shape is a Polygon and that it
    intersects all the input points.

    Args:
        points: data points
        alpha: alpha value

    Returns:
        bool: True if the resulting alpha shape is a single polygon that
            intersects all the input data points.
    """
    polygon = _alpha_shape(points, alpha)

    if isinstance(polygon, shapely.geometry.polygon.Polygon):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points)).geoms
        return all((polygon.intersects(point) for point in points))

    else:
        return False


def optimizealpha(points: Union[List[Tuple[float]], np.ndarray],
                  max_iterations: int = 500, 
                  lower: float = 0.,
                  upper: float = 1000.0 , #sys.float_info.max, 
                  silent: bool = False):
    """
    Solve for the alpha parameter.

    Attempt to determine the alpha parameter that best wraps the given set of
    points in one polygon without dropping any points.

    Note:  If the solver fails to find a solution, a value of zero will be
    returned, which when used with the alphashape function will safely return a
    convex hull around the points.

    Args:

        points: an iterable container of points
        max_iterations (int): maximum number of iterations while finding the
            solution
        lower: lower limit for optimization
        upper: upper limit for optimization
        silent: silence warnings

    Returns:

        float: The optimized alpha parameter

    """

    # Set the bounds
    assert lower >= 0, "The lower bounds must be at least 0"
    # Ensure the upper limit bounds the solution
    assert upper <= sys.float_info.max, (
        f'The upper bounds must be less than or equal to {sys.float_info.max} '
        'on your system')

    if _testalpha(points, upper):
        if not silent:
            warnings.warn('the max float value does not bound the alpha '
                          'parameter solution')
        return 0.

    # Begin the bisection loop
    counter = 0
    while (upper - lower) > np.finfo(float).eps * 2:
        # Bisect the current bounds
        test_alpha = (upper + lower) * .5
        print(test_alpha)
        # Update the bounds to include the solution space
        if _testalpha(points, test_alpha):
            lower = test_alpha
        else:
            upper = test_alpha

        # Handle exceeding maximum allowed number of iterations
        counter += 1
        if counter > max_iterations:
            if not silent:
                warnings.warn('maximum allowed iterations reached while '
                              'optimizing the alpha parameter')
            # lower = 0.
            break
    return test_alpha

def circumcenter(points: Union[List[Tuple[float]], np.ndarray]) -> np.ndarray:
    """
    Calculate the circumcenter of a set of points in barycentric coordinates.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumcenter of a set of points in barycentric coordinates.
    """
    points = np.asarray(points)
    num_rows, num_columns = points.shape
    A = np.bmat([[2 * np.dot(points, points.T),
                  np.ones((num_rows, 1))],
                 [np.ones((1, num_rows)), np.zeros((1, 1))]])
    b = np.hstack((np.sum(points * points, axis=1),
                   np.ones((1))))
    return np.linalg.solve(A, b)[:-1]


def circumradius(points: Union[List[Tuple[float]], np.ndarray]) -> float:
    """
    Calculte the circumradius of a given set of points.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumradius of a given set of points.
    """
    points = np.asarray(points)
    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))


def alphasimplices(points: Union[List[Tuple[float]], np.ndarray]) -> \
        Union[List[Tuple[float]], np.ndarray]:
    """
    Returns an iterator of simplices and their circumradii of the given set of
    points.

    Args:
      points: An `N`x`M` array of points.

    Yields:
      A simplex, and its circumradius as a tuple.
    """
    coords = np.asarray(points)
    tri = Delaunay(coords)

    for simplex in tri.simplices:
        simplex_points = coords[simplex]
        try:
            yield simplex, circumradius(simplex_points)
        except np.linalg.LinAlgError:
            warnings.warn('Singular matrix. Likely caused by all points '
                          'lying in an N-1 space.')


def _alpha_shape(points: Union[List[Tuple[float]], np.ndarray],
               alpha: Union[None, float] = None):
    """
    Compute the alpha shape (concave hull) of a set of points.  If the number
    of points in the input is three or less, the convex hull is returned to the
    user.  For two points, the convex hull collapses to a `LineString`; for one
    point, a `Point`.

    Args:

      points (list or ``shapely.geometry.MultiPoint`` or \
          ``geopandas.GeoDataFrame``): an iterable container of points
      alpha (float): alpha value

    Returns:

      ``shapely.geometry.Polygon`` or ``shapely.geometry.LineString`` or
      ``shapely.geometry.Point`` or ``geopandas.GeoDataFrame``: \
          the resulting geometry
    """

    # If given a triangle for input, or an alpha value of zero or less,
    # return the convex hull.
    if len(points) < 4 or (alpha is not None and not callable(
            alpha) and alpha <= 0):
        #TODO
        return None

    # Determine alpha parameter if one is not given
    if alpha is None:
        alpha = optimizealpha(points)

    coords = np.array(points)

    # Create a set to hold unique edges of simplices that pass the radius
    # filtering
    edges = set()

    # Create a set to hold unique edges of perimeter simplices.
    # Whenever a simplex is found that passes the radius filter, its edges
    # will be inspected to see if they already exist in the `edges` set.  If an
    # edge does not already exist there, it will be added to both the `edges`
    # set and the `permimeter_edges` set.  If it does already exist there, it
    # will be removed from the `perimeter_edges` set if found there.  This is
    # taking advantage of the property of perimeter edges that each edge can
    # only exist once.
    perimeter_edges = set()

    for point_indices, circumradius in alphasimplices(coords):
        if callable(alpha):
            resolved_alpha = alpha(point_indices, circumradius)
        else:
            resolved_alpha = alpha

        # Radius filter
        if circumradius < 1.0 / resolved_alpha:
            for edge in itertools.combinations(
                    point_indices, r=coords.shape[-1]):
                if all([e not in edges for e in itertools.combinations(
                        edge, r=len(edge))]):
                    edges.add(edge)
                    perimeter_edges.add(edge)
                else:
                    perimeter_edges -= set(itertools.combinations(
                        edge, r=len(edge)))

    if coords.shape[-1] > 2: # > 2D
        return perimeter_edges


    from shapely.ops import unary_union, polygonize
    from shapely.geometry import MultiLineString
    # Create the resulting polygon from the edge points
    m = MultiLineString([coords[np.array(edge)] for edge in perimeter_edges])
    triangles = list(polygonize(m))
    result = unary_union(triangles)

    return result


def alpha_shape(points, alpha=None, radius=None):

    if radius is not None:
        alpha = 1/radius
    return _alpha_shape(points, alpha).exterior.coords[:-1]