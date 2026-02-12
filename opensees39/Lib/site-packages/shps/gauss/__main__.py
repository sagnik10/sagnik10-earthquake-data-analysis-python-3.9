
from .forge import *

if __name__ == "__main__":

    rng = (1,100)
    source = False
    write_json = False
    fmt = "py"
    ndm = 1
    families = []
    argi = iter(sys.argv[1:])
    for arg in argi:
        if arg == "-n":
            rng = tuple(map(int,next(argi).split(",")))
        elif arg == "-s":
            source = True
        elif arg == "-json":
            fmt = "json"
        elif arg == "-c":
            fmt = "c"
        elif arg == "--ndm":
            ndm = int(next(argi))
        elif arg == "-cxx":
            fmt = "cxx"
        elif arg == "-h":
            print(HELP)
            sys.exit()
        else:
            families.append(arg.lower())

    if not families:
        families = "lobatto legendre radau kronrod".split(" ") \
                 + "newton_cotes_closed newton_cotes_open".split(" ")

    {
         "c": C,
         "cxx": PlaneCXX,
         "py": Python,
         "json": JSON
    }[fmt](ndm).print(families, rng)

