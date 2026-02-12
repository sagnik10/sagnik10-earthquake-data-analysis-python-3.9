#from functools import lru_cache

#@lru_cache
def trinomial(i, j, k):
    if i < 0 or j < 0 or k < 0:
        return 0

    head = (j==0  and  k==0)
    left = (j==i) and (k==0)
    rght = (j==i) and (k==i)

    if head or left or rght:
        return 1

    return   trinomial(i-1, j-1, k-1) \
           + trinomial(i-1, j-1, k) \
           + trinomial(i-1, j,   k)

def pyramid(order):
    return [
            [
                [
                    trinomial(i, j, k) for k in range(j+1)
            ] for j in range(i + 1)
        ] for i in range(order+1)
    ]

def terms(order):
    terms = []
    for i in range(order+1):
        for j in range(i):
            terms.append(f"x**{i}*y**{j}")

        terms.append(f"x**{i}*y**{i}")

        for j in reversed(range(i)):
            terms.append(f"x**{j}*y**{i}")
    return terms

import sys
#print(pyramid(int(sys.argv[1])))
print(terms(int(sys.argv[1])))

