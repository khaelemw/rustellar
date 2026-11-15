//! Complete elliptic integrals, from Bulirsch's general complete integral
//!
//! ```text
//! cel(kc, p, a, b) = integral from 0 to pi/2 of
//!     (a cos^2 t + b sin^2 t) / ((cos^2 t + p sin^2 t) sqrt(cos^2 t + kc^2 sin^2 t)) dt
//! ```
//!
//! (Bulirsch 1969, Numerische Mathematik 13, 305), as used by Agol, Luger &
//! Foreman-Mackey (2020, AJ 159, 123). The algorithm is an arithmetic-geometric
//! mean (AGM) iteration, which converges quadratically, so a few steps reach
//! full precision for any `kc`. Every Legendre integral is a special case: with
//! `k^2 = 1 - kc^2`, `cel(kc, 1, 1, 1) = K(k)`, `cel(kc, 1, 1, kc^2) = E(k)` and
//! `cel(kc, 1, 1, 0) = (E(k) - kc^2 K(k)) / k^2`.
//!
//! The transit formulas need three integrals with the same `kc`, so
//! [`cel_e_d`] computes them together, sharing the AGM steps.

use std::f64::consts::FRAC_PI_2;

// The AGM stops once its two means agree to this relative difference. The
// error left is about its square, below the rounding error of a double.
const AGM_TOLERANCE: f64 = 1e-9;

// More steps than the AGM ever needs: from kc = 1e-300 it converges in about
// a dozen.
const MAX_STEPS: usize = 64;

/// The three integrals for one `kc` in `(0, 1]` or above:
///
/// - `cel(kc, p, a, b)` for `p > 0`, taking `sqrt(p)` and `b / sqrt(p)`;
/// - `E(k) = cel(kc, 1, 1, kc^2)`;
/// - `D(k) = cel(kc, 1, 1, 0) = (E(k) - kc^2 K(k)) / k^2`.
///
/// Passing `sqrt(p)` and `b / sqrt(p)` rather than `p` and `b` lets callers
/// work them out without forming `p`, which in the transit formulas is the
/// square of a distance that can be far smaller than `sqrt(f64::MIN_POSITIVE)`.
///
/// `kc` must be positive: at `kc = 0` (`k = 1`) `K` is infinite.
pub(crate) fn cel_e_d(kc: f64, sqrt_p: f64, a: f64, b_over_sqrt_p: f64) -> (f64, f64, f64) {
    // The general integral, with its own p, and two with p = 1 (Bulirsch's
    // `cel` after its set-up for p > 0, which divides b by sqrt(p)).
    let mut general = Agm {
        a,
        b: b_over_sqrt_p,
        p: sqrt_p,
    };
    let mut e = Agm {
        a: 1.0,
        b: kc * kc,
        p: 1.0,
    };
    let mut d = Agm {
        a: 1.0,
        b: 0.0,
        p: 1.0,
    };

    let mut kc = kc;
    let mut ee = kc;
    let mut m = 1.0;

    for _ in 0..MAX_STEPS {
        general.step(ee);
        e.step(ee);
        d.step(ee);

        let previous_m = m;
        m += kc;
        if (previous_m - kc).abs() <= previous_m * AGM_TOLERANCE {
            break;
        }
        kc = 2.0 * ee.sqrt();
        ee = kc * m;
    }

    (general.finish(m), e.finish(m), d.finish(m))
}

// One integral's share of the AGM iteration.
struct Agm {
    a: f64,
    b: f64,
    p: f64,
}

impl Agm {
    fn step(&mut self, ee: f64) {
        let f = self.a;
        self.a += self.b / self.p;
        let g = ee / self.p;
        self.b = 2.0 * (self.b + f * g);
        self.p += g;
    }

    fn finish(&self, m: f64) -> f64 {
        FRAC_PI_2 * (self.a * m + self.b) / (m * (m + self.p))
    }
}
