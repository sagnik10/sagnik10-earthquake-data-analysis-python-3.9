
def check(a, b):
    
    print(f"{a-b}")


def test(u):

    # check(u.oz, u.lbm/16)

    check(u.lbf/u.slug,   u.ft / u.sec**2)
    check(u.pdl,   0.138254954376*u.N)
    check(u.slug, 32.17405*u.lbm)
    check(u.slug,  14.59390*u.kg)
    check(u.lbm, 0.45359237*u.kg)  # International avoirdupois pound




if __name__ == "__main__":
    from xara.units import si, iks, ips, fks, fps

    test(si)
    test(iks)
    test(ips)
    test(fks)
    test(fps)