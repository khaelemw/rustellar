//! Eccentric orbits: Kepler's equation and where the planet is on the sky.
//!
//! The transit is at inferior conjunction, true anomaly `f = π/2 - ω`, and
//! `b` is the impact parameter there (Winn 2010, eq. 7):
//! `b = a cos(i) (1 - e^2) / (1 + e sin ω)`. With `r = a (1 - e cos E)` the
//! star-planet distance, the sky-projected separation is
//! `z = r sqrt(cos^2(ω + f) + sin^2(ω + f) cos^2 i)`, and the planet is in
//! front of the star while `sin(ω + f) > 0`.

use std::f64::consts::{PI, TAU};

/// An eccentric orbit's shape and orientation, worked out once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Eccentric {
    pub(crate) e: f64,
    /// The argument of periastron, in radians.
    pub(crate) omega: f64,
    sin_w: f64,
    cos_w: f64,
    /// `sqrt(1 - e^2)`.
    root: f64,
    /// The mean anomaly at mid-transit.
    m0: f64,
    pub(crate) cos_i: f64,
    /// The planet is clear of the star outside these times from
    /// mid-transit, in days (`None`: it never crosses the star).
    pub(crate) window: Option<(f64, f64)>,
}

impl Eccentric {
    /// The orbit with eccentricity `e` (in `(0, 1)`) and argument of
    /// periastron `omega_deg` (degrees) whose impact parameter at
    /// mid-transit is `b`, or `None` if no inclination gives it
    /// (`cos i >= 1`).
    pub(crate) fn new(e: f64, omega_deg: f64, a: f64, b: f64) -> Option<Self> {
        let (sin_w, cos_w) = sin_cos_deg(omega_deg);
        let root = ((1.0 - e) * (1.0 + e)).sqrt();
        let cos_i = b / a * e.mul_add(sin_w, 1.0) / (root * root);
        if cos_i >= 1.0 {
            return None;
        }
        // The eccentric anomaly at f = π/2 - ω, by the half-angle form.
        let (sin_half, cos_half) = sin_cos_deg(0.5f64.mul_add(-omega_deg, 45.0));
        let big_e = 2.0 * ((1.0 - e).sqrt() * sin_half).atan2((1.0 + e).sqrt() * cos_half);
        let m0 = (-e).mul_add(big_e.sin(), big_e);
        Some(Self {
            e,
            omega: omega_deg.to_radians(),
            sin_w,
            cos_w,
            root,
            m0,
            cos_i,
            window: None,
        })
    }

    /// The sky-projected separation, in stellar radii, at `dm` radians of
    /// mean anomaly after mid-transit, and whether the planet is in front.
    pub(crate) fn sky(&self, a: f64, dm: f64) -> (f64, bool) {
        let m = (self.m0 + dm + PI).rem_euclid(TAU) - PI;
        let (sin_e, cos_e) = kepler(m, self.e).sin_cos();
        let den = (-self.e).mul_add(cos_e, 1.0);
        let (cos_f, sin_f) = ((cos_e - self.e) / den, self.root * sin_e / den);
        // sin(ω + f) and cos(ω + f).
        let s = self.sin_w.mul_add(cos_f, self.cos_w * sin_f);
        let c = self.cos_w.mul_add(cos_f, -(self.sin_w * sin_f));
        (a * den * c.hypot(s * self.cos_i), s > 0.0)
    }
}

/// The sine and cosine of `deg` degrees, reduced exactly to within 45° of
/// a multiple of 90° first, so those multiples give exact zeros and ones
/// (`270f64.to_radians().cos()` is -1.8e-16, not 0).
pub(crate) fn sin_cos_deg(deg: f64) -> (f64, f64) {
    let quarter = (deg / 90.0).round();
    // Exact: `deg` and `90 quarter` are within a factor of two of each
    // other or the remainder is small (Sterbenz).
    let rest = 90.0f64.mul_add(-quarter, deg);
    let (s, c) = rest.to_radians().sin_cos();
    #[allow(clippy::cast_possible_truncation)]
    match (quarter.rem_euclid(4.0)) as u8 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// The eccentric anomaly `E` with `E - e sin E = m`, for `m` in `[-π, π]`
/// and `e` in `[0, 1)`: Halley's method from Danby's (1987) start
/// `m + 0.85 e sign(m)`, which converges for every such `m` and `e`, to
/// within a few units in the last place.
pub(crate) fn kepler(m: f64, e: f64) -> f64 {
    if m == 0.0 || e == 0.0 {
        return m;
    }
    let mut big_e = (0.85 * e).copysign(m) + m;
    for _ in 0..64 {
        let (s, c) = big_e.sin_cos();
        let f = (-e).mul_add(s, big_e) - m;
        let fp = (-e).mul_add(c, 1.0);
        let step = f / (0.5 * f * e * s).mul_add(-1.0 / fp, fp);
        big_e -= step;
        if step.abs() <= 2.0 * f64::EPSILON * big_e.abs().max(1.0) {
            break;
        }
    }
    big_e
}

/// The times from mid-transit, in days, where the separation in front of
/// the star first falls to `edge` and last rises past it (`period` in
/// days), or `None` if it never falls that far: the minimum of the
/// separation from a scan of the orbit refined by golden sections, then
/// each crossing by bisection, to the times' rounding.
pub(crate) fn crossings(orbit: &Eccentric, a: f64, period: f64, edge: f64) -> Option<(f64, f64)> {
    let z = |dt: f64| {
        let (z, front) = orbit.sky(a, TAU * dt / period);
        if front { z } else { f64::INFINITY }
    };
    const SCAN: usize = 2048;
    let spacing = period / SCAN as f64;
    let at = |i: usize| (i as f64).mul_add(spacing, -0.5 * period);
    let best = (0..=SCAN)
        .min_by(|&i, &j| z(at(i)).total_cmp(&z(at(j))))
        .unwrap_or(0);
    // Golden sections within a spacing either side.
    let (mut lo, mut hi) = (at(best) - spacing, at(best) + spacing);
    let ratio = 0.5 * (5.0f64.sqrt() - 1.0);
    for _ in 0..200 {
        let (x1, x2) = ((-ratio).mul_add(hi - lo, hi), ratio.mul_add(hi - lo, lo));
        if z(x1) <= z(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
        if hi - lo <= f64::EPSILON * period {
            break;
        }
    }
    let middle = 0.5 * (lo + hi);
    if z(middle) >= edge {
        return None;
    }
    let side = |direction: f64| {
        // Out until the separation is past the edge, then bisect.
        let mut inside = middle;
        let mut step = spacing;
        let mut outside = direction.mul_add(step, middle);
        while z(outside) < edge {
            inside = outside;
            step *= 2.0;
            outside = direction.mul_add(step, middle);
        }
        for _ in 0..200 {
            let mid = 0.5 * (inside + outside);
            if mid == inside || mid == outside {
                break;
            }
            if z(mid) < edge {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        0.5 * (inside + outside)
    };
    Some((side(-1.0), side(1.0)))
}
