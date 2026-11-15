//! Analytic flux for quadratic limb darkening (Mandel & Agol 2002).
//!
//! With `I(mu) = c0 + c2 mu + c4 mu^2`, the blocked light splits into three
//! integrals over the part of the star the planet covers, each divided by `pi`:
//!
//! - `B0`: of `1`, the covered area (from `overlap_area`),
//! - `B1`: of `mu = sqrt(1 - r^2)`, Mandel & Agol's `lambda_d` plus `2/3` when
//!   the planet covers the star's centre,
//! - `B2`: of `r^2`, Mandel & Agol's `eta_d`.
//!
//! Over the whole star they are 1, 2/3 and 1/2.
//!
//! `B2` is one function for every position. `B1` has one function per region
//! of Mandel & Agol's Table 1, since the general formulas are `0/0` or
//! `0 * infinity` on the lines between regions, though `B1` itself is smooth
//! there; [`blocked_mu`] picks the function.
//!
//! In this module `p` is the radius ratio and `z` the separation, both in
//! stellar radii, with `z >= 0`, as in the paper. The shorthand is
//! `a = (z - p)^2`, `b = (z + p)^2` and `q = p^2 - z^2`. Elliptic integrals
//! use the parameter `m`, with `mc = 1 - m`, and characteristic `n`, as in
//! [`crate::elliptic`].
use std::f64::consts::PI;

use crate::{
    LimbDarkening, RadiusRatio,
    elliptic::{ellip_e, ellip_k, ellip_pi},
    uniform::overlap_area,
};

// Half-width of the band around `z = p` where `blocked_mu_centre_line`
// replaces the general formulas. There the general formulas lose precision like
// `1e-16 / |z - p|`, while the `z = p` value differs from the true one by about
// `0.4 |z - p|`; the two are equal near `1e-8`.
const CENTRE_LINE_BAND: f64 = 1e-8;

// Half-width of the band around `z = 1 - p` (second and third contact), and
// around `p = 1/2` on the line `z = p`. The formulas converge on these lines,
// so the band only keeps `mc` away from zero, where `K` is infinite.
const INNER_CONTACT_BAND: f64 = 1e-12;

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
/// Within about `1e-9` of [`flux_ring_sum`](crate::flux_ring_sum) at 20,000
/// rings, for `k` from 0.0001 to 1.5. Within `1e-8` of the planet's edge
/// crossing the star's centre (`|z| = k`), where the general formula loses
/// precision, a limiting formula is used instead; it is within about `1e-8`.
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

    let b0 = overlap_area(z, p, 1.0) / PI;
    let b1 = blocked_mu(z, p);
    let b2 = blocked_r2(z, p);

    // I(mu) = c0 + c2 mu + c4 mu^2, and mu^2 = 1 - r^2, so the blocked light
    // over pi is c0 B0 + c2 B1 + c4 (B0 - B2), with c0 + c4 = 1 - c2.
    let (u1, u2) = ld.quadratic_coefficients();
    let c2 = u1 + 2.0 * u2;
    let c4 = -u2;

    1.0 - ((1.0 - c2) * b0 + c2 * b1 - c4 * b2) / ld.disk_integral()
}

/// `B1`: the covered part of the star integrated against `mu`, divided by
/// `pi`, for a planet that overlaps the star without covering it:
/// `|1 - p| < z < 1 + p` or `z <= 1 - p`.
///
/// Picks the formula for the region of Mandel & Agol's Table 1 that `z` and
/// `p` fall in. The lines between regions cross, so the most specific comes
/// first: `z = 0`, then `z = 1 - p` (which includes `z = p = 1/2`), then
/// `z = p`.
pub(crate) fn blocked_mu(z: f64, p: f64) -> f64 {
    if z == 0.0 {
        return blocked_mu_centred(p);
    }

    if p < 1.0 && (z - (1.0 - p)).abs() < INNER_CONTACT_BAND {
        return blocked_mu_inner_contact(p);
    }

    if (z - p).abs() < CENTRE_LINE_BAND {
        return blocked_mu_centre_line(p);
    }

    if z < 1.0 - p {
        blocked_mu_inside(z, p)
    } else {
        blocked_mu_partial(z, p)
    }
}

/// `B2`: the covered part of the star integrated against `r^2`, divided by
/// `pi` (Mandel & Agol's `eta_d`). Valid for every `z` and `p`.
///
/// ```text
/// z >= 1 + p            0                                    (no overlap)
/// p > 1, z <= p - 1     1/2                                  (star covered)
/// z <= 1 - p            eta_2 = (p^2 / 2)(p^2 + 2 z^2)       (planet inside)
/// otherwise             eta_1 = [kappa_1 + 2 eta_2 kappa_0
///                                - (1/4)(1 + 5 p^2 + z^2) sqrt((1 - a)(b - 1))] / (2 pi)
/// ```
///
/// `kappa_0` and `kappa_1` are the half-angles of the arcs bounding the
/// overlap, seen from the planet's and the star's centre, as in
/// `overlap_area`.
pub(crate) fn blocked_r2(z: f64, p: f64) -> f64 {
    if z >= 1.0 + p {
        return 0.0;
    }

    if p > 1.0 && z <= p - 1.0 {
        return 0.5;
    }

    let p2 = p * p;
    let z2 = z * z;

    // The whole planet disk: by the parallel-axis theorem, a disk of radius p
    // at distance z, weighted by r^2.
    let eta2 = 0.5 * p2 * (p2 + 2.0 * z2);

    if z <= 1.0 - p {
        return eta2;
    }

    // Partial: |1 - p| < z < 1 + p. Same factored form as in `overlap_area`,
    // with R = 1; it is also sqrt((1 - a)(b - 1)).
    let s = ((1.0 + p - z) * (1.0 + p + z) * (z - 1.0 + p) * (z + 1.0 - p))
        .max(0.0)
        .sqrt();

    // kappa_1 = acos((1 - p^2 + z^2) / (2 z)) and
    // kappa_0 = acos((p^2 + z^2 - 1) / (2 p z)); atan2 keeps full precision
    // near 0 and pi, where acos does not.
    let kappa1 = s.atan2(1.0 - p2 + z2);
    let kappa0 = s.atan2(p2 + z2 - 1.0);

    (kappa1 + 2.0 * eta2 * kappa0 - 0.25 * (1.0 + 5.0 * p2 + z2) * s) / (2.0 * PI)
}

/// `B1` with the planet fully inside the stellar disk: `0 < z < 1 - p`, away
/// from `z = p`.
///
/// Mandel & Agol's `lambda_2`, plus `2/3` when `z < p`:
///
/// ```text
/// m = 4 z p / (1 - a)      mc = (1 - b) / (1 - a)      n = 1 - b / a
///
/// lambda_2 = 2 / (9 pi sqrt(1 - a)) * {
///       (1 - 5 z^2 + p^2 + q^2) K(m)
///     + (1 - a)(z^2 + 7 p^2 - 4) E(m)
///     - 3 (q / a) Pi(n | m) }
/// ```
///
/// `n` is negative throughout. Near `z = p`, `q / a` grows like
/// `1 / |z - p|` while `Pi` shrinks, and the result loses precision; see
/// [`blocked_mu_centre_line`]. Near `z = 0` it is accurate, but
/// [`blocked_mu_centred`] is exact and cheaper.
pub(crate) fn blocked_mu_inside(z: f64, p: f64) -> f64 {
    let p2 = p * p;
    let z2 = z * z;
    let a = (z - p) * (z - p);
    let b = (z + p) * (z + p);
    let q = p2 - z2;

    // mc from 1 - m = (1 - a - 4 z p) / (1 - a) with b - a = 4 z p, so it is
    // never a small difference from 1, even next to z = 1 - p where m -> 1.
    let m = 4.0 * z * p / (1.0 - a);
    let mc = (1.0 - b) / (1.0 - a);
    let n = 1.0 - b / a;

    let k = ellip_k(mc);
    let e = ellip_e(m, mc);
    let pi_n = ellip_pi(n, mc);

    let lambda2 = 2.0 / (9.0 * PI * (1.0 - a).sqrt())
        * ((1.0 - 5.0 * z2 + p2 + q * q) * k + (1.0 - a) * (z2 + 7.0 * p2 - 4.0) * e
            - 3.0 * (q / a) * pi_n);

    lambda2 + covers_centre(z, p)
}

/// `B1` with the planet across the stellar limb: `|1 - p| < z < 1 + p`, away
/// from `z = p`.
///
/// Mandel & Agol's `lambda_1`, plus `2/3` when `z < p`:
///
/// ```text
/// m = (1 - a) / (4 z p)    mc = (b - 1) / (4 z p)      n = 1 - 1 / a
///
/// lambda_1 = 1 / (9 pi sqrt(p z)) * {
///       [(1 - b)(2 b + a - 3) - 3 q (b - 2)] K(m)
///     + 4 p z (z^2 + 7 p^2 - 4) E(m)
///     - 3 (q / a) Pi(n | m) }
/// ```
///
/// `n` is negative throughout. Near `z = p` the result loses precision as for
/// [`blocked_mu_inside`].
pub(crate) fn blocked_mu_partial(z: f64, p: f64) -> f64 {
    let p2 = p * p;
    let z2 = z * z;
    let a = (z - p) * (z - p);
    let b = (z + p) * (z + p);
    let q = p2 - z2;

    // mc from 1 - m = (4 z p - 1 + a) / (4 z p) with 4 z p + a = b, so it is
    // never a small difference from 1, even next to z = 1 - p where m -> 1.
    let m = (1.0 - a) / (4.0 * z * p);
    let mc = (b - 1.0) / (4.0 * z * p);
    let n = 1.0 - 1.0 / a;

    let k = ellip_k(mc);
    let e = ellip_e(m, mc);
    let pi_n = ellip_pi(n, mc);

    let lambda1 = 1.0 / (9.0 * PI * (p * z).sqrt())
        * (((1.0 - b) * (2.0 * b + a - 3.0) - 3.0 * q * (b - 2.0)) * k
            + 4.0 * p * z * (z2 + 7.0 * p2 - 4.0) * e
            - 3.0 * (q / a) * pi_n);

    lambda1 + covers_centre(z, p)
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
/// Inside or across the limb, depending on `p`; no `2/3` is added on this line.
///
/// ```text
/// p < 1/2:  lambda_4 = 1/3 + 2 / (9 pi) [4 (2 p^2 - 1) E(m) + (1 - 4 p^2) K(m)]
///           m = 4 p^2,         mc = (1 - 2 p)(1 + 2 p)
///
/// p > 1/2:  lambda_3 = 1/3 + 16 p / (9 pi) (2 p^2 - 1) E(m)
///                          - (1 - 4 p^2)(3 - 8 p^2) / (9 pi p) K(m)
///           m = 1 / (4 p^2),   mc = (2 p - 1)(2 p + 1) / (4 p^2)
///
/// p = 1/2:  1/3 - 4 / (9 pi)
/// ```
///
/// Also the value to use near the line, where [`blocked_mu_inside`] and
/// [`blocked_mu_partial`] lose precision. It differs from the true `B1` at
/// distance `d` from the line by about `0.4 d`.
pub(crate) fn blocked_mu_centre_line(p: f64) -> f64 {
    // Both formulas have mc = 0 at p = 1/2, where K is infinite and its
    // coefficient is 0. They converge to this value, so use it close by too.
    if (p - 0.5).abs() < INNER_CONTACT_BAND {
        return 1.0 / 3.0 - 4.0 / (9.0 * PI);
    }

    let p2 = p * p;

    if p < 0.5 {
        let m = 4.0 * p2;
        let mc = (1.0 - 2.0 * p) * (1.0 + 2.0 * p);
        let lambda4 = 1.0 / 3.0
            + 2.0 / (9.0 * PI)
                * (4.0 * (2.0 * p2 - 1.0) * ellip_e(m, mc) + (1.0 - 4.0 * p2) * ellip_k(mc));
        return lambda4;
    }

    let m = 1.0 / (4.0 * p2);
    let mc = (2.0 * p - 1.0) * (2.0 * p + 1.0) / (4.0 * p2);
    1.0 / 3.0 + 16.0 * p / (9.0 * PI) * (2.0 * p2 - 1.0) * ellip_e(m, mc)
        - (1.0 - 4.0 * p2) * (3.0 - 8.0 * p2) / (9.0 * PI * p) * ellip_k(mc)
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
/// On this line `mc = 0` in the general formulas. They converge to this
/// value, with an error of about `3 d` at distance `d` from the line.
pub(crate) fn blocked_mu_inner_contact(p: f64) -> f64 {
    2.0 / (3.0 * PI) * (1.0 - 2.0 * p).acos()
        - 4.0 / (9.0 * PI) * (3.0 + 2.0 * p - 8.0 * p * p) * (p * (1.0 - p)).sqrt()
}

// The 2/3 that Mandel & Agol's lambda leaves out when the planet covers the
// star's centre (their Heaviside term), so the kernels return B1 itself.
fn covers_centre(z: f64, p: f64) -> f64 {
    if z < p { 2.0 / 3.0 } else { 0.0 }
}
