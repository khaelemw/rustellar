//! Analytic flux for quadratic limb darkening (Mandel & Agol 2002; Agol,
//! Luger & Foreman-Mackey 2020).
//!
//! With `I(mu) = c0 + c2 mu + c4 mu^2`, the blocked light splits into three
//! integrals over the part of the star the planet covers, each divided by `pi`:
//!
//! - `B0`: of `1`, the covered area ([`blocked_area`]),
//! - `B1`: of `mu = sqrt(1 - r^2)`, Mandel & Agol's `lambda_d` plus `2/3` when
//!   the planet covers the star's centre,
//! - `B2`: of `r^2`, Mandel & Agol's `eta_d`.
//!
//! Over the whole star they are 1, 2/3 and 1/2.
//!
//! `B0` and `B2` need no elliptic integrals. `B1` uses the formulation of Agol,
//! Luger & Foreman-Mackey (2020, section 3 and appendix A), written with
//! Bulirsch's general complete integral `cel` ([`crate::elliptic`]). Their
//! `Lambda_1` is `3 pi` times Mandel & Agol's `lambda_d`, so
//! `B1 = Lambda_1 / (3 pi)`, plus `2/3` when the planet covers the centre.
//!
//! Compared with Mandel & Agol's own formulas, this needs one AGM iteration
//! instead of three separate integrals, and it does not cancel terms of size
//! `p^3` at large `p`. Near the line `z = p`, the third-kind integral carries a
//! factor `1 / (z - p)` that Mandel & Agol's formula removes by subtraction.
//! Here [`blocked_mu_inside`] and [`blocked_mu_partial`] instead pass `cel` the
//! square root of its parameter, and the ratio that contains the factor, both
//! already worked out, so the formula keeps full precision right up to the
//! line. Only the line itself, where the formula is `0 / 0`, needs its own
//! function, [`blocked_mu_centre_line`].
//!
//! In this module `p` is the radius ratio and `z` the separation, both in
//! stellar radii, with `z >= 0`, as in the papers.
use std::f64::consts::PI;

use crate::{
    LimbDarkening, RadiusRatio,
    elliptic::cel_e_d,
    lens::{Lens, arc_1, arc_2, sum3},
    uniform::blocked_area,
};

// Relative distance `|z - p| / (z + p)` below which `B1` is taken from the
// line `z = p`. The general formulas are exact up to the line, but their
// intermediate values grow like `(z + p) / |z - p|`, and would overflow for
// distances near the smallest doubles. `B1` is smooth across the line, so its
// value on the line differs by about this fraction of `B1`.
const CENTRE_LINE: f64 = 1e-100;

/// Flux of a limb-darkened star while a planet of radius ratio `k` sits a
/// distance `z` from the star's centre on the sky.
///
/// Exact for the quadratic limb-darkening law (Mandel & Agol 2002), and so
/// for the linear and uniform laws, which are quadratic laws with `u2 = 0`.
///
/// `z` is in stellar radii. Only its size matters, so a negative `z` gives
/// the same flux as `-z`. The flux is relative to the star on its own: exactly
/// 1 once `|z| >= 1 + k`, and exactly 0 when the planet covers the whole disk
/// (`k > 1` and `|z| <= k - 1`). A NaN `z` gives NaN.
///
/// This does not know about orbits: whether the planet is in front of the
/// star at all is for the caller to check.
///
/// # Accuracy
///
/// Within `1e-15` of the exact flux for `k` up to 1, and within about
/// `1.5e-16 k` above that, so `1.5e-12` at [`RadiusRatio::MAX`]. Checked at
/// 36,742 points against 60-digit values, on and next to every line where the
/// formulas change case and at separations down to the smallest double (see
/// the crate docs). The error is relative to the star's flux, so relative to
/// the transit depth `k^2` it is larger for small planets: about `1e-11` of
/// the depth at `k = 0.01`. Rounding is clamped so the result never leaves
/// `[0, 1]`.
///
/// # Examples
///
/// ```
/// use rustellar_model::{LimbDarkening, RadiusRatio, flux};
///
/// let k = RadiusRatio::new(0.1)?;
/// let ld = LimbDarkening::quadratic(0.3, 0.25)?;
///
/// assert!((flux(0.3, k, ld) - 0.988_527_81).abs() < 1e-8);
/// assert_eq!(flux(1.1, k, ld), 1.0); // just touching the limb
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn flux(z: f64, k: RadiusRatio, ld: LimbDarkening) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }

    let z = z.abs();
    let p = k.get();

    // Out of transit, which is most of any light curve: no elliptic integrals.
    if z >= 1.0 + p {
        return 1.0;
    }

    if p > 1.0 && z <= p - 1.0 {
        return 0.0;
    }

    let b0 = blocked_area(z, p);
    let b1 = blocked_mu(z, p);
    let b2 = blocked_r2(z, p);

    // I(mu) = c0 + c2 mu + c4 mu^2, and mu^2 = 1 - r^2, so the blocked light
    // over pi is c0 B0 + c2 B1 + c4 (B0 - B2), with c0 + c4 = 1 - c2.
    let (u1, u2) = ld.quadratic_coefficients();
    let c2 = u1 + 2.0 * u2;
    let c4 = -u2;

    let flux = 1.0 - ((1.0 - c2) * b0 + c2 * b1 - c4 * b2) / ld.disk_integral();

    // The exact flux is in [0, 1]; rounding can take it about 1e-12 outside,
    // near first contact and near full cover.
    flux.clamp(0.0, 1.0)
}

/// `B1`: the covered part of the star integrated against `mu`, divided by
/// `pi`, for a planet that overlaps the star without covering it:
/// `|1 - p| < z < 1 + p` or `z <= 1 - p`.
///
/// Picks the formula for the region `z` and `p` fall in. The lines between
/// regions cross, so the most specific comes first: `z = 0`, then `z = p`
/// (which includes `z = p = 1/2`), then inside the disk, across the limb, or
/// touching the limb from inside (`z = 1 - p`).
pub(crate) fn blocked_mu(z: f64, p: f64) -> f64 {
    if z == 0.0 {
        return blocked_mu_centred(p);
    }

    let sum = z + p;
    if (z - p).abs() <= CENTRE_LINE * sum {
        return blocked_mu_centre_line(p);
    }

    // 1 - (z + p)^2, as a product with an exact first factor, so its sign is
    // right however close z is to 1 - p.
    let one_minus_sum2 = sum3(1.0, -z, -p) * (1.0 + sum);
    if one_minus_sum2 > 0.0 {
        blocked_mu_inside(z, p)
    } else if one_minus_sum2 < 0.0 {
        blocked_mu_partial(z, p)
    } else {
        blocked_mu_inner_contact(p)
    }
}

/// `B2`: the covered part of the star integrated against `r^2`, divided by
/// `pi` (Mandel & Agol's `eta_d`). Valid for every `z` and `p`.
///
/// ```text
/// z >= 1 + p            0                                 (no overlap)
/// p > 1, z <= p - 1     1/2                               (star covered)
/// z <= 1 - p            eta_2 = (p^2 / 2)(p^2 + 2 z^2)    (planet inside)
/// otherwise             the lens, below
/// ```
///
/// For a lens ([`Lens`]), Green's theorem turns the integral of `r^2` into
/// `(1/4)` of the integral of `r^4` against the angle seen from the star's
/// centre, along the lens's edge. The star's arc gives `kappa1 / 2`. Along the
/// planet's arc `r^2 = d^2 + 4 z p h` and the angle element is
/// `p (2 z h - d) dphi`, with `h = sin^2(phi / 2)` and `d = z - p`, so it gives
///
/// ```text
/// (p / 4) [-2 d^3 kappa0 + 2 z d (d - 2 p) arc_1(kappa0) + 8 z^2 p arc_2(kappa0)]
/// ```
///
/// Mandel & Agol's `eta_1` is the same quantity, but it cancels terms of size
/// `p^4` when the planet is much larger than the star; these terms do not.
pub(crate) fn blocked_r2(z: f64, p: f64) -> f64 {
    if z >= 1.0 + p {
        return 0.0;
    }

    if p > 1.0 && z <= p - 1.0 {
        return 0.5;
    }

    if z <= 1.0 - p {
        // The whole planet disk: by the parallel-axis theorem, a disk of
        // radius p at distance z, weighted by r^2.
        return 0.5 * p * p * (p * p + 2.0 * z * z);
    }

    let lens = Lens::new(z, p);
    let k0 = lens.kappa0;
    let d = z - p;

    let planet_arc = 0.25
        * p
        * (-2.0 * d * d * d * k0
            + 2.0 * z * d * (d - 2.0 * p) * arc_1(k0)
            + 8.0 * z * z * p * arc_2(k0));

    (0.5 * lens.kappa1 + planet_arc) / PI
}

/// `B1` with the planet fully inside the stellar disk: `z > 0`,
/// `z + p < 1`, `z != p`.
///
/// Agol, Luger & Foreman-Mackey (2020), their cases 3 and 9 (`k^2 > 1`), with
/// `d = z - p` and `s = z + p`:
///
/// ```text
/// kc^2 = (1 - s^2) / (1 - d^2)       (the complementary modulus of 1 / k^2)
/// P    = (d / s)^2 kc^2              mu = 3 (d / s) / (1 - d^2)
///
/// Lambda_1 = (2/3) sqrt(1 - d^2) [(1 - s^2) cel(kc, P, 1 + mu, P + mu)
///                                 - (4 - 7 p^2 - z^2) E]
/// ```
///
/// `cel` gets `sqrt(P) = kc |d| / s` and
/// `(P + mu) / sqrt(P) = sqrt(P) + 3 sign(d) / (kc (1 - d^2))`, both free of
/// cancellation however close `z` is to `p`.
pub(crate) fn blocked_mu_inside(z: f64, p: f64) -> f64 {
    let d = z - p;
    let s = z + p;
    let one_minus_d2 = sum3(1.0, -z, p) * sum3(1.0, z, -p);
    let one_minus_s2 = sum3(1.0, -z, -p) * (1.0 + s);

    let kc = (one_minus_s2 / one_minus_d2).sqrt();
    let ratio = d / s;
    let mu = 3.0 * ratio / one_minus_d2;
    let sqrt_p = ratio.abs() * kc;
    let b_over_sqrt_p = sqrt_p + 3.0_f64.copysign(d) / (kc * one_minus_d2);

    let (pi_term, e, _) = cel_e_d(kc, sqrt_p, 1.0 + mu, b_over_sqrt_p);

    let lambda = 2.0 / 3.0
        * one_minus_d2.sqrt()
        * (one_minus_s2 * pi_term - (4.0 - 7.0 * p * p - z * z) * e);

    lambda / (3.0 * PI) + covers_centre(z, p)
}

/// `B1` with the planet across the stellar limb: `|1 - p| < z < 1 + p`,
/// `z != p`.
///
/// Agol, Luger & Foreman-Mackey (2020), their cases 2 and 8 (`k^2 < 1`), with
/// `d = z - p` and `s = z + p`:
///
/// ```text
/// kc^2 = (s^2 - 1) / (4 z p)       P = d^2 kc^2
///
/// Lambda_1 = (1 - d^2) [cel(kc, P, 0, 3 kc^2 d s)
///                       + (6 p^2 + 2 z p - 3) D - 4 z p E] / (3 sqrt(z p))
/// ```
///
/// `cel` gets `sqrt(P) = kc |d|` and `3 kc^2 d s / sqrt(P) = 3 kc s sign(d)`.
pub(crate) fn blocked_mu_partial(z: f64, p: f64) -> f64 {
    let d = z - p;
    let s = z + p;
    let one_minus_d2 = sum3(1.0, -z, p) * sum3(1.0, z, -p);
    let four_zp = 4.0 * z * p;

    let kc = (sum3(z, p, -1.0) * (s + 1.0) / four_zp).sqrt();
    let sqrt_p = d.abs() * kc;
    let b_over_sqrt_p = (3.0 * kc * s).copysign(d);

    let (pi_term, e, dk) = cel_e_d(kc, sqrt_p, 0.0, b_over_sqrt_p);

    let lambda = one_minus_d2 * (pi_term + (6.0 * p * p + 2.0 * z * p - 3.0) * dk - four_zp * e)
        / (3.0 * (z * p).sqrt());

    lambda / (3.0 * PI) + covers_centre(z, p)
}

/// `B1` with the planet centred on the star: `z = 0`, `p < 1`.
///
/// ```text
/// (2/3)(1 - (1 - p^2)^(3/2))
/// ```
///
/// This is Mandel & Agol's `lambda_6` plus `2/3`.
pub(crate) fn blocked_mu_centred(p: f64) -> f64 {
    // Whole rings of the star out to radius p:
    // 2 * integral from 0 to p of r sqrt(1 - r^2) dr.
    2.0 / 3.0 * (1.0 - (1.0 - p * p).powf(1.5))
}

/// `B1` with the planet's edge on the star's centre: `z = p`, any `p`.
///
/// Agol, Luger & Foreman-Mackey (2020), their cases 5, 6 and 7. No `2/3` is
/// added on this line: the formulas already include half of it.
///
/// ```text
/// p < 1/2:  1/3 + 2 / (9 pi) [(2 m - 3) E - m D]     kc^2 = 1 - m
/// p = 1/2:  1/3 - 4 / (9 pi)
/// p > 1/2:  1/3 + [(2 m - 3) D - m E] / (9 pi p)     kc^2 = 1 - 1 / m
/// ```
///
/// with `m = 4 p^2`. Neither formula needs `K`, so both stay finite at
/// `p = 1/2`, where they meet.
pub(crate) fn blocked_mu_centre_line(p: f64) -> f64 {
    if p == 0.5 {
        return 1.0 / 3.0 - 4.0 / (9.0 * PI);
    }

    let m = 4.0 * p * p;
    let two_p = 2.0 * p;

    if p < 0.5 {
        let kc = ((1.0 - two_p) * (1.0 + two_p)).sqrt();
        let (_, e, dk) = cel_e_d(kc, 1.0, 1.0, 1.0);
        return 1.0 / 3.0 + 2.0 / (9.0 * PI) * ((2.0 * m - 3.0) * e - m * dk);
    }

    let kc = ((two_p - 1.0) * (two_p + 1.0)).sqrt() / two_p;
    let (_, e, dk) = cel_e_d(kc, 1.0, 1.0, 1.0);
    1.0 / 3.0 + ((2.0 * m - 3.0) * dk - m * e) / (9.0 * PI * p)
}

/// `B1` at second or third contact, the planet touching the limb from inside:
/// `z = 1 - p`, `p < 1`.
///
/// Mandel & Agol's `lambda_5`, plus `2/3` when `p > 1/2` (then `z < p`). The
/// paper's own `-(2/3)` step term for `p > 1/2` cancels that, leaving:
///
/// ```text
/// 2 / (3 pi) acos(1 - 2 p) - 4 / (9 pi) (3 + 2 p - 8 p^2) sqrt(p (1 - p))
/// ```
///
/// On this line `kc = 0` in the general formulas, where `cel` is undefined.
/// They converge to this value as `z` approaches the line.
pub(crate) fn blocked_mu_inner_contact(p: f64) -> f64 {
    2.0 / (3.0 * PI) * (1.0 - 2.0 * p).acos()
        - 4.0 / (9.0 * PI) * (3.0 + 2.0 * p - 8.0 * p * p) * (p * (1.0 - p)).sqrt()
}

// The 2/3 that Mandel & Agol's lambda leaves out when the planet covers the
// star's centre (their Heaviside term), so the kernels return B1 itself.
fn covers_centre(z: f64, p: f64) -> f64 {
    if z < p { 2.0 / 3.0 } else { 0.0 }
}
