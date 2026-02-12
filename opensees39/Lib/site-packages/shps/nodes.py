from . import plane
from itertools import product, count

# elem = plane.Lagrange(3)
# nodes = elem.nodes
# #from block import plot
# from plotting import Plotter
# ax = None
# Plotter(ax=ax).nodes(nodes).show()

def feap_numbers(order):
    for i in range(order**2):
        if (0 < i < 4*(order-1)):
            if not i%(order-1):
                i //= order-1
            else:
                i += 3 - i//(order-1)
        yield i+1


