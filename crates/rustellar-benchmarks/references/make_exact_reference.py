"""Write 60-digit reference values of the blocked-light integrals and the flux.

For a planet of radius ratio p at separation z (both exact doubles), with
quadratic limb darkening I = (1 - c2) + c2 mu + u2 r^2, c2 = u1 + 2 u2:

    b0  covered area / pi                         (integral of 1)
    b1  covered part integrated against mu, / pi  (integral of mu)
    b2  covered part integrated against r^2, / pi (integral of r^2)
    flux = 1 - ((1 - c2) b0 + c2 b1 + u2 b2) / (1 - u1/3 - u2/6)

computed with mpmath from the closed forms of Mandel &
Agol (2002) in Legendre's elliptic integrals: a different formulation from the
one the crate uses (Bulirsch's cel, as in Agol, Luger & Foreman-Mackey 2020).
Each point is worked at enough digits that every sum and product of its inputs
is exact (up to about 700 for a separation near the smallest double), plus 60
more. On the lines where the formulas change case the inputs are compared
exactly, so a point one rounding step off a line uses the general formula,
which keeps far more precision than a double has. Every value is rounded to
the nearest double once, at the end.

As a check on the closed forms themselves, b0, b1 and b2 are also computed by
adaptive quadrature at the same precision, at points away from the lines, and the two must
agree to 1e-40, or the script stops.

Writes, under crates/rustellar-model/tests/reference/:

    exact_lines.csv   on and around every case line, and around the points
                      where lines cross, from 1 rounding step to 0.1 away
    exact_random.csv  random points over the whole range (Sobol sequence)

Run from the repository root:

    python crates/rustellar-benchmarks/references/make_exact_reference.py
"""

import math
import multiprocessing
import platform
import sys
from pathlib import Path

import mpmath
from mpmath import mp, mpf

DPS = 60


def dps_for(*values):
    """Enough digits that sums and products of these doubles and 1 are exact,
    plus DPS more: a double spans at most 53 bits, so its digits run from its
    leading power of ten down to about 16 places below it."""
    exponents = [0] + [math.floor(math.log10(abs(v))) for v in values if v != 0]
    return DPS + 2 * (max(exponents) - min(exponents) + 17)
OUT = Path(__file__).resolve().parents[3] / "crates" / "rustellar-model" / "tests" / "reference"

# Limb darkening, cycled through the points: Kipping's (2013) prior covers the
# physical range; (1, 0) is the darkest linear law, (2, -1) is I = mu^2, and
# (-0.2, 0.1) is brighter towards the edge.
U_SETS = [(0.4, 0.25), (1.0, 0.0), (2.0, -1.0), (-0.2, 0.1), (0.1, 0.6), (0.7, -0.2)]


def b0(z, p):
    if z >= 1 + p:
        return mpf(0)
    if p > 1 and z <= p - 1:
        return mpf(1)
    if z <= 1 - p:
        return p * p
    k0 = mpmath.acos((p * p + z * z - 1) / (2 * p * z))
    k1 = mpmath.acos((1 - p * p + z * z) / (2 * z))
    s = mpmath.sqrt(4 * z * z - (1 + z * z - p * p) ** 2)
    return (k1 + p * p * k0 - s / 2) / mp.pi


def b2(z, p):
    if z >= 1 + p:
        return mpf(0)
    if p > 1 and z <= p - 1:
        return mpf(1) / 2
    eta2 = p * p * (p * p + 2 * z * z) / 2
    if z <= 1 - p:
        return eta2
    s = mpmath.sqrt((1 + p - z) * (1 + p + z) * (z - 1 + p) * (z + 1 - p))
    k1 = mpmath.atan2(s, 1 - p * p + z * z)
    k0 = mpmath.atan2(s, p * p + z * z - 1)
    return (k1 + 2 * eta2 * k0 - (1 + 5 * p * p + z * z) * s / 4) / (2 * mp.pi)


def b1(z, p):
    if z >= 1 + p:
        return mpf(0)
    if p > 1 and z <= p - 1:
        return mpf(2) / 3
    if z == 0:
        return mpf(2) / 3 * (1 - (1 - p * p) ** mpf(1.5))
    if z == p:
        if p == mpf(1) / 2:
            return mpf(1) / 3 - 4 / (9 * mp.pi)
        if p < mpf(1) / 2:
            m = 4 * p * p
            return mpf(1) / 3 + 2 / (9 * mp.pi) * (
                4 * (2 * p * p - 1) * mpmath.ellipe(m) + (1 - 4 * p * p) * mpmath.ellipk(m))
        m = 1 / (4 * p * p)
        return (mpf(1) / 3 + 16 * p / (9 * mp.pi) * (2 * p * p - 1) * mpmath.ellipe(m)
                - (1 - 4 * p * p) * (3 - 8 * p * p) / (9 * mp.pi * p) * mpmath.ellipk(m))
    if p < 1 and z == 1 - p:
        return (2 / (3 * mp.pi) * mpmath.acos(1 - 2 * p)
                - 4 / (9 * mp.pi) * (3 + 2 * p - 8 * p * p) * mpmath.sqrt(p * (1 - p)))
    a = (z - p) ** 2
    b = (z + p) ** 2
    q = p * p - z * z
    covers_centre = mpf(2) / 3 if z < p else mpf(0)
    if z < 1 - p:
        m = 4 * z * p / (1 - a)
        n = 1 - b / a
        lam = 2 / (9 * mp.pi * mpmath.sqrt(1 - a)) * (
            (1 - 5 * z * z + p * p + q * q) * mpmath.ellipk(m)
            + (1 - a) * (z * z + 7 * p * p - 4) * mpmath.ellipe(m)
            - 3 * (q / a) * mpmath.ellippi(n, m))
    else:
        m = (1 - a) / (4 * z * p)
        n = 1 - 1 / a
        lam = 1 / (9 * mp.pi * mpmath.sqrt(p * z)) * (
            ((1 - b) * (2 * b + a - 3) - 3 * q * (b - 2)) * mpmath.ellipk(m)
            + 4 * p * z * (z * z + 7 * p * p - 4) * mpmath.ellipe(m)
            - 3 * (q / a) * mpmath.ellippi(n, m))
    return lam + covers_centre


def by_quadrature(z, p):
    """b0, b1 and b2 by integrating over rings of the star."""

    def theta(r):
        # Half the angle of the ring of radius r that the planet covers.
        if r <= p - z:
            return mp.pi
        if r <= z - p or r >= z + p:
            return mpf(0)
        return mpmath.acos((r * r + z * z - p * p) / (2 * r * z))

    cuts = sorted({mpf(0), mpf(1)} | {c for c in (abs(z - p), z + p) if 0 < c < 1})
    ring = lambda w: mpmath.quad(lambda r: w(r) * 2 * theta(r) * r, cuts) / mp.pi
    return (ring(lambda r: 1), ring(lambda r: mpmath.sqrt(1 - r * r)), ring(lambda r: r * r))


def row(args):
    k, z, u1, u2 = args
    mp.dps = dps_for(k, z)
    p, zz = mpf(k), mpf(z)
    v0, v1, v2 = b0(zz, p), b1(zz, p), b2(zz, p)
    U1, U2 = mpf(u1), mpf(u2)
    c2 = U1 + 2 * U2
    flux = 1 - ((1 - c2) * v0 + c2 * v1 + U2 * v2) / (1 - U1 / 3 - U2 / 6)
    return (k, z, u1, u2, float(v0), float(v1), float(v2), float(flux))


def check(args):
    k, z = args
    mp.dps = dps_for(k, z)
    p, zz = mpf(k), mpf(z)
    closed = (b0(zz, p), b1(zz, p), b2(zz, p))
    quad = by_quadrature(zz, p)
    return max(abs(c - q) for c, q in zip(closed, quad))


K_LINES = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.49, 0.499999, 0.5,
           0.500001, 0.51, 0.7, 0.9, 0.99, 0.999999, 1.0, 1.000001, 1.01, 1.5, 2.0, 3.0, 5.0,
           10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 1e4]
OFFSETS = [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 0.1]
STEPS = [1, 2, 3, 10]


def near(x):
    """x, a few rounding steps either side, and fixed offsets either side."""
    out = {x}
    for n in STEPS:
        up = down = x
        for _ in range(n):
            up = math.nextafter(up, math.inf)
            down = math.nextafter(down, -math.inf)
        out |= {up, down}
    for d in OFFSETS:
        out |= {x + d, x - d}
    return out


def line_points():
    points = set()
    for k in K_LINES:
        lines = [0.0, k, abs(1 - k), 1 + k] + ([k - 1] if k > 1 else [])
        for line in lines:
            for z in near(line):
                if z >= 0:
                    points.add((k, z))
    # Where lines cross: z = k = 1/2 (z = k meets z = 1 - k) and z = 0, k = 1
    # (z = 0 meets z = |1 - k|).
    for k0, z0 in [(0.5, 0.5), (1.0, 0.0)]:
        for k in near(k0):
            for z in near(z0):
                if z >= 0 and k > 0:
                    points.add((k, z))
    return sorted(points)


def random_points(n):
    from scipy.stats import qmc

    sample = qmc.Sobol(d=5, scramble=True, seed=20261116).random(n)
    points = []
    for i, (s_k, s_z, s_q1, s_q2, s_where) in enumerate(sample):
        k = 10 ** (-6 + 10 * s_k)  # 1e-6 to 1e4
        if s_where < 0.5:
            # Across the whole transit and a little beyond.
            lo = max(0.0, k - 1.0) - 0.05 * min(k, 1.0)
            hi = 1.0 + k + 0.05 * min(k, 1.0)
        else:
            # Across the limb only, |1 - k| < z < 1 + k, which for a small
            # planet is too narrow for the first half to reach.
            lo, hi = abs(1.0 - k), 1.0 + k
        z = max(0.0, lo + (hi - lo) * s_z)
        if i % 4 == 3:
            # A quarter from anywhere in the allowed limb darkening: u1 + u2
            # <= 1, intensity never negative.
            u1, u2 = -1.0 + 3.0 * s_q1, -1.0 + 2.0 * s_q2
            if u1 + u2 > 1.0 or (u2 < 0 and 0 < u1 < -2 * u2 and u1 * u1 + 4 * u2 > 0):
                u1, u2 = 2 * math.sqrt(s_q1) * s_q2, math.sqrt(s_q1) * (1 - 2 * s_q2)
        else:
            # Kipping (2013): uniform q1, q2 cover every physical law.
            u1, u2 = 2 * math.sqrt(s_q1) * s_q2, math.sqrt(s_q1) * (1 - 2 * s_q2)
        points.append((float(k), float(z), float(u1), float(u2)))
    return points


def write(path, header, rows):
    with path.open("w", newline="\n", encoding="ascii") as f:
        f.write("\n".join(header) + "\n")
        for r in rows:
            f.write(",".join(repr(x) for x in r) + "\n")
    print(f"wrote {len(rows)} rows to {path}", file=sys.stderr)


def main():
    mp.dps = DPS
    with multiprocessing.Pool() as pool:
        # The closed forms against quadrature, away from the lines.
        generic = [(k, z) for k, z, _, _ in random_points(4096)[::8]
                   if min(abs(z - line) for line in (0, k, abs(1 - k), 1 + k)) > 1e-3
                   and z < 1 + k and not (k > 1 and z <= k - 1)]
        worst = max(pool.map(check, generic))
        print(f"closed forms vs quadrature at {len(generic)} points: {mpmath.nstr(worst, 3)}",
              file=sys.stderr)
        assert worst < mpf("1e-40"), worst

        lines = line_points()
        line_rows = pool.map(row, [(k, z) + U_SETS[i % len(U_SETS)] for i, (k, z) in enumerate(lines)])
        random_rows = pool.map(row, random_points(32768))

    common = [
        f"# b0, b1, b2: covered area, and the covered part integrated against mu and r^2, each over pi.",
        f"# flux = 1 - ((1 - c2) b0 + c2 b1 + u2 b2) / (1 - u1/3 - u2/6), c2 = u1 + 2 u2.",
        f"# Mandel & Agol (2002) closed forms with every input sum exact plus {DPS} digits (mpmath {mpmath.__version__}, Python "
        f"{platform.python_version()}), checked against quadrature to 1e-40 at "
        f"{len(generic)} points; each value rounded to the nearest double.",
    ]
    write(OUT / "exact_lines.csv",
          [f"# {len(line_rows)} points on and near every case line (z = 0, k, |1 - k|, 1 + k, k - 1) "
           f"for {len(K_LINES)} values of k from 1e-6 to 1e4: 1, 2, 3 and 10 rounding steps away, "
           f"and 1e-15 to 0.1 away; and around z = k = 1/2 and z = 0, k = 1."]
          + common + ["k,z,u1,u2,b0,b1,b2,flux"], line_rows)
    write(OUT / "exact_random.csv",
          [f"# {len(random_rows)} random points: k log-uniform from 1e-6 to 1e4, z across the transit "
           f"and a little beyond for half and across the limb for the other half, limb darkening mostly from Kipping's (2013) prior (Sobol, seed 20261116)."]
          + common + ["k,z,u1,u2,b0,b1,b2,flux"], random_rows)


if __name__ == "__main__":
    main()
