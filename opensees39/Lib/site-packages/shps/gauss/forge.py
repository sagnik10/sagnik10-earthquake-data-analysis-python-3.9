import os
import re
import sys
import json
import errno
import signal
import inspect
import textwrap
import functools
from pathlib import Path
from datetime import date
from collections import namedtuple

import quadpy
import numpy as np

class Quadrature(dict):
    @property
    def points(self):  return self["points"]
    @property
    def weights(self): return self["weights"]

class ProductQuadrature:
    def __init__(self, interval, ndm=2):
        import itertools
        self.rule = interval
        self.points = []
        self.weights = []

        values = [(x,w) for x,w in zip(interval.points, interval.weights)]
        for crd in itertools.product(*[values]*ndm):
           #self.points.append((x[0], y[0]))
           #self.weights.append(x[1]*y[1])
            self.points.append(tuple(map(lambda i: i[0], crd)))
            self.weights.append(np.product(list(map(lambda i: i[1], crd))))

    def __iter__(self):
        return zip(self.points, self.weights)


def collect_family(family, rng):
    rules = []
    for n in range(*rng):
        try:
            rules.append(get_rule(family,n))
        except TimeoutError:
            break

    return rules

def write(families, rng, fmt):
    data = {f: collect_family(f, rng) for f in families}

    print(fmt.form_head())
    print(fmt.form_fams(data))
    print(fmt.form_foot())


class Writer:
    def __init__(self, ndm):
        self.ndm = ndm

    def print(self, families, rng):
        data = {f: collect_family(f, rng) for f in families}

        print(self.form_head(families))
        print(self.form_fams(data))
        print(self.form_foot())

    def form_fams(self, data):
        fams = {}
        return "".join(self.form_rule(fam, rules)
                       for fam, rules in data.items())

    def form_foot(self):
        return ""


class JSON(Writer):
    def print(self, families, rng):
        data = {f: collect_family(f, rng) for f in families}
        json.dump({
            fam: [
                {k: v.tolist() if hasattr(v,"tolist") else v
                    for k,v in rule[1].items()} for rule in data[fam] if rule[0]
            ] for fam in data
        }, sys.stdout, indent=4)

class Python(Writer):
    def form_data(self, n, q, g):
        return f"""
          "degree": {q['degree']},
          "points": {repr(q.points.tolist())},
          "weights": {repr(q.weights.tolist())},
          "generator": "{g.__name__}"
        """
    def form_head(self, *args):
        print(textwrap.dedent("""

        class IntervalQuadrature:
            def __init__(self, n):
                data = self.__class__.data[n]
                self.points = data["points"]
                self.weights = data["weights"]
                self.degree = data["degree"]
        """))

    def form_rule(self, family, data):
        NL = "    },\n        "
        return textwrap.dedent(f"""\
        class {family.replace('_',' ').title().replace(' ','')}(IntervalQuadrature):
            data = {{
                {NL.join(f"{n}: {{{self.form_data(n,q,g)}" for n,q,g in data if q)}
                }}
            }}
        """)

class PlaneCXX(Writer):
    def form_head(self, families, *args):
        return "\n".join(f"""
#pragma once
template <int ndm, int nip> struct Gauss{family.title()};
""" for family in families)

    def form_rule(self, family, data):
        NL = "\n"
        return (f"""
    {NL.join(self.form_data(n,q,family) for n,q,g in data if q)}

    """)

    def form_data(self, n, q, family):
        Q = ProductQuadrature(q, self.ndm)
       #points = ",\n    ".join(f"{{{x:20}, {y:20}}}" for x in X)  for X in Q.points)
        points = ",\n    ".join(
                ("{" if self.ndm > 1 else " ") \
                + ", ".join(f"{x:20.17}" for x in X) \
                + ("}"  if self.ndm > 1 else "") \
                for X in Q.points)
        weights = ",\n    ".join(f" {w:20.17}" for w in Q.weights)
        deg = n
        n = len(Q.points)
        ndm = self.ndm
        dim = f"[{ndm}]" if ndm > 1 else ""
        return textwrap.dedent(f"""
template<> struct Gauss{family.title()}<{ndm},{n}> {{
  constexpr static int nip = {n};
  constexpr static int deg = {deg};

  constexpr static double pts[{n}]{dim} = {{
    {points}
  }};
  constexpr static double wts[{n}] = {{
    {weights}
  }};
}};
        """)

class C(Writer):
    def form_head(self, *args):
        return ("""
#include <stdio.h>  // printf
#include <stdlib.h> // atoi
#include <string.h> // strcmp

const static struct IntervalQuadrature {int n, deg; double points[][2];}""")

    def form_rule(self, family, data):
        NL = "},\n"
        return (f"""
    {NL.join(f"{family}{n:02} = {{{self.form_data(n,q,g)}" for n,q,g in data if q)}}}

    """)

    def form_data(self, n, q, g):
        points = textwrap.indent(
            ",\n".join(f"{{{x:20}, {w:20}}}" for x,w in zip(q.points,q.weights)),
            " "*4
        )
        return (f"{n}, {q['degree']}, {{\n{points}}}")

    def form_foot(self):
        return ";"


class TimeoutError(Exception): pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    "https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish"
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try: result = func(*args, **kwargs)
            finally: signal.alarm(0)
            return result
        return wrapper
    return decorator

@timeout(10)
def get_rule(family, n):
    q = None
    g = None
    print(f"{family}({n})", file=sys.stderr)

    if Path(f"./data/{family}.json").is_file():
        with open(f"./data/{family}.json", "r") as f:
            data = json.load(f)

        for i in range(len(data[family])):
            if len(data[family][i]["points"]) == n:
#           if data[family][i]["degree"] == n:
                q = Quadrature(**data[family][i])
                print(f" Json", file=sys.stderr)
                g = data
                return n,q,g

    for s in ["{family}","gauss_{family}","{family}_gauss_"+str(n)]:
        for lib in [quadpy.c1, quadpy.e1r, quadpy.e1r2]:
            if hasattr(lib, s.format(family=family)):
                try:
                    g = getattr(lib, s.format(family=family))
                    q = Quadrature(**vars(g(n)))
                except AssertionError as e:
                    print(f"  Assert: {e}", file=sys.stderr)
                    return [None]*3
                except Exception as e:
                    print(f"  {e}", file=sys.stderr)
                    q = g
                break
    if q is None:
        return [None]*3

    return n,q,g

