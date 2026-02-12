from shps.bases import stringify
from shps.plane import Lagrange, Legendre, Serendipity

"""
-I
<family> <order> -l       [lang]
                 -c        s t
                 -p/plot  <mode>
                 -b/block  n m
                 -g/gauss  [n]
"""

if __name__ == "__main__":
    import sys
    Elem = {
        "lagrange":    Lagrange,
        "legendre":    Legendre,
        "serendipity": Serendipity
    }[sys.argv[1].lower()]
    elem = Elem(int(sys.argv[2]))
#   print(elem.print_shape())
    if "-P" in sys.argv:
        import matplotlib.pyplot as plt
#       elem.plot(0)
        [elem.plot(i) for i in range(len(elem.shapes()))]
        plt.show()

    elif "-C" in sys.argv:
        print(stringify(elem, latex=False))

    elif "-s" in sys.argv:
        """
        evaluate shapes at coordinates
            python shp.py Lagrange 2 -s '[0.2, 0.2]'
        """
        import json
        coord = json.loads(sys.argv[4])
        for i,node in enumerate(elem.shapef):
            print(f"{i}: ", node(*coord))

    else:
        print(stringify(elem, latex=True))

