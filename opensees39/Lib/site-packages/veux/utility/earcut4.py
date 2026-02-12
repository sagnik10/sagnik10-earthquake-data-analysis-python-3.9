"""
https://github.com/pawlowiczf/Polygon-Triangulation/tree/main
"""
class Node:
    def __init__(self, color, next = None):
        self.next  = None 

        
def _create_links(colors):
    array = [Node(color) for color in colors]

    for idx in range( len(array) ):
        pointer = array[idx]
        pointer.next = array[ ( idx + 1 ) % len(array) ]
    #
        
    return array[0]



def Orientation(a, b, c):
    return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])


def Position(a, b, c):
    #
    ax, ay = a 
    bx, by = b 
    cx, cy = c 

    answer = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])

    if answer > 0:
        return 1 
    if answer < 0:
        return -1
    return float('inf')

def _point_equal(a, b):
    return a[0] == b[0] and a[1] == b[1]

def onTriangle(a, b, c, p):
    return _point_equal(a,p) or _point_equal(b,p) or _point_equal(c,p)

def inTriangle(a, b, c, p):
    #
    number = 0
    number += Position(a, b, p)
    number += Position(b, c, p)
    number += Position(c, a, p)

    if abs( number ) == 3:
        return True
    #
    return False

def _find_ear(polygon, vis):
    #
    n = len(polygon)
    if n < 3:
        return []
    
    for i in range( n  ):
        #
        a = polygon[ (i - 1) % n ][0]
        b = polygon[ i % n ][0]
        c = polygon[ (i + 1) % n ][0]

        if Orientation(a, b, c) > 0:
            ear = True 

            for p, index in polygon:
                if not onTriangle(a,b,c,p) and inTriangle(a, b, c, p):
                    ear = False 
                    break

            if ear:    
                vis.append([a, b, c])
                return ( polygon[ (i - 1) % n ][1], i % n , polygon[ (i + 1) % n ][1] )

    return []

def earcut(vertices, vis=None):
    #
    if len(vertices) == 0:
        return []

    if vis is None: 
        vis = []

    polygon = [(vertices[i], i) for i in range(len(vertices))]
    
    while len(polygon) >= 3:
        #
        index = _find_ear(polygon, vis)
        if index == []: 
            break
        polygon.pop( index[1] )

    return vis