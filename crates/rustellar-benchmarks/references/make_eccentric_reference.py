"""Eccentric-orbit reference values for rustellar-model.

For six orbits, by mpmath 1.4.1 at 40 digits:

- the sky-projected separation z and whether the planet is in front, at
  times spread over the whole orbit and close about the transit: Kepler's
  equation by mpmath's findroot, z = r sqrt(cos^2(w + f) + sin^2(w + f)
  cos^2 i), with t0 the time of inferior conjunction (f = pi/2 - w) and b
  the impact parameter there (Winn 2010): cos i = (b / a)(1 + e sin w) /
  (1 - e^2);
- the contact times from mid-transit (z = 1 + k and 1 - k), by bisection
  to 1e-30 days.

And batman 2.5.1's flux at the times about the transit (quadratic limb
darkening u = 0.4, 0.25), for an end-to-end check against its model.

Writes crates/rustellar-model/tests/reference/eccentric_points.csv and
eccentric_contacts.csv.

Run: python crates/rustellar-benchmarks/references/make_eccentric_reference.py
"""

import math
import os

import batman
import mpmath as mp
import numpy as np

mp.mp.dps = 40

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "crates",
                   "rustellar-model", "tests", "reference")

# name, e, w (deg), a/R*, b, k, P (days)
ORBITS = [
    ("mild", 0.1, 30.0, 10.0, 0.3, 0.1, 3.0),
    ("moderate", 0.5, 120.0, 20.0, 0.5, 0.08, 10.0),
    ("extreme", 0.9, 270.0, 60.0, 0.2, 0.05, 100.0),
    ("negative_w", 0.3, -45.0, 8.0, 0.9, 0.12, 2.0),
    ("grazing", 0.6, 90.0, 15.0, 1.0, 0.1, 5.0),
    ("central", 0.05, 0.0, 5.0, 0.0, 0.1, 1.0),
]


class Orbit:
    def __init__(self, e, w, a, b, period):
        self.e, self.w, self.a, self.period = mp.mpf(e), mp.radians(w), mp.mpf(a), mp.mpf(period)
        self.cos_i = mp.mpf(b) / self.a * (1 + self.e * mp.sin(self.w)) / (1 - self.e ** 2)
        f = mp.pi / 2 - self.w
        big_e = 2 * mp.atan2(mp.sqrt(1 - self.e) * mp.sin(f / 2), mp.sqrt(1 + self.e) * mp.cos(f / 2))
        self.m0 = big_e - self.e * mp.sin(big_e)

    def sky(self, dt):
        m = self.m0 + 2 * mp.pi * mp.mpf(dt) / self.period
        m = (m + mp.pi) % (2 * mp.pi) - mp.pi
        big_e = mp.findroot(lambda x: x - self.e * mp.sin(x) - m, m + 0.85 * self.e * mp.sign(m or 1))
        r = self.a * (1 - self.e * mp.cos(big_e))
        cos_f = (mp.cos(big_e) - self.e) / (1 - self.e * mp.cos(big_e))
        sin_f = mp.sqrt(1 - self.e ** 2) * mp.sin(big_e) / (1 - self.e * mp.cos(big_e))
        s = mp.sin(self.w) * cos_f + mp.cos(self.w) * sin_f
        c = mp.cos(self.w) * cos_f - mp.sin(self.w) * sin_f
        return r * mp.sqrt(c ** 2 + (s * self.cos_i) ** 2), s > 0

    def front_z(self, dt):
        z, front = self.sky(dt)
        return z if front else mp.inf

    def crossings(self, edge):
        # The minimum from a fine scan, refined by golden sections.
        n = 4096
        ts = [-self.period / 2 + self.period * i / n for i in range(n + 1)]
        best = min(range(n + 1), key=lambda i: self.front_z(ts[i]))
        lo, hi = ts[best] - self.period / n, ts[best] + self.period / n
        g = (mp.sqrt(5) - 1) / 2
        for _ in range(200):
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            if self.front_z(x1) <= self.front_z(x2):
                hi = x2
            else:
                lo = x1
        mid = (lo + hi) / 2
        if self.front_z(mid) >= edge:
            return None

        def side(direction):
            inside, step = mid, self.period / n
            outside = mid + direction * step
            while self.front_z(outside) < edge:
                inside, step = outside, step * 2
                outside = mid + direction * step
            for _ in range(200):
                m = (inside + outside) / 2
                if self.front_z(m) < edge:
                    inside = m
                else:
                    outside = m
            return (inside + outside) / 2

        return side(-1), side(1)


def batman_flux(e, w, a, b, k, period, times):
    orbit = Orbit(e, w, a, b, period)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, period, k, a
    p.inc = math.degrees(math.acos(float(orbit.cos_i)))
    p.ecc, p.w = e, w
    p.limb_dark, p.u = "quadratic", [0.4, 0.25]
    m = batman.TransitModel(p, np.array(times))
    return m.light_curve(p)


def main():
    with open(os.path.join(OUT, "eccentric_points.csv"), "w", newline="\n") as points, \
            open(os.path.join(OUT, "eccentric_contacts.csv"), "w", newline="\n") as contacts:
        points.write("orbit,e,w_deg,a_rs,b,k,period,dt,z,front,batman_flux\n")
        contacts.write("orbit,edge,before,after\n")
        for name, e, w, a, b, k, period in ORBITS:
            orbit = Orbit(e, w, a, b, period)
            near = orbit.crossings(1 + k)
            # The whole orbit, then close about the transit.
            times = [period * (i / 97 - 0.5) for i in range(97)]
            if near is not None:
                lo, hi = float(near[0]), float(near[1])
                span = hi - lo
                times += [lo - 0.2 * span + 1.4 * span * i / 99 for i in range(100)]
            times.append(0.0)
            flux = batman_flux(e, w, a, b, k, period, times)
            for dt, f in zip(times, flux):
                z, front = orbit.sky(dt)
                points.write(f"{name},{e!r},{w!r},{a!r},{b!r},{k!r},{period!r},{dt!r},"
                             f"{mp.nstr(z, 20)},{int(bool(front))},{float(f)!r}\n")
            for edge in (1 + k, 1 - k):
                c = orbit.crossings(mp.mpf(edge))
                if c is None:
                    contacts.write(f"{name},{edge!r},,\n")
                else:
                    contacts.write(f"{name},{edge!r},{mp.nstr(c[0], 20)},{mp.nstr(c[1], 20)}\n")
            print(name, "done", flush=True)


if __name__ == "__main__":
    main()
