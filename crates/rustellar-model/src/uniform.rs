use std::f64::consts::PI;

use crate::{
    RadiusRatio,
    lens::{Lens, arc_1, sum3},
};

/// `B0`: the part of the star (radius 1) covered by a planet of radius `p` at
/// distance `z >= 0`, divided by `pi`, so 1 for the whole star.
///
/// For a lens ([`Lens`]), Green's theorem gives the area as
/// `kappa1 - p d kappa0 + p z arc_1(kappa0)` with `d = z - p`: the star's arc
/// contributes `kappa1`, and the planet's arc the rest. This is the usual
/// `kappa1 + p^2 kappa0 - s / 2` rearranged so that no two terms of size `p^2`
/// cancel when the planet is much larger than the star.
pub(crate) fn blocked_area(z: f64, p: f64) -> f64 {
    if z >= 1.0 + p {
        return 0.0;
    }

    if z <= 1.0 - p {
        return p * p;
    }

    if z <= p - 1.0 {
        return 1.0;
    }

    let lens = Lens::new(z, p);
    let d = z - p;
    (lens.kappa1 - p * d * lens.kappa0 + p * z * arc_1(lens.kappa0)) / PI
}

// Overlap area of a disk of radius `big_r` at the origin and a disk of radius
// `k` whose centre is `z` away (the lens formula), for the ring sum, which
// needs disks of any radius. Expects `z >= 0`, `k > 0` and `big_r >= 0`;
// `big_r = 0` gives 0 with no special case.
pub(crate) fn overlap_area(z: f64, k: f64, big_r: f64) -> f64 {
    if z >= big_r + k {
        return 0.0;
    }

    if z <= big_r - k {
        return PI * k * k;
    }

    if z <= k - big_r {
        return PI * big_r * big_r;
    }

    // Partial overlap, so |R - k| < z < R + k. In particular `z = 0` never gets
    // here: one of the two branches above always holds at `z = 0`.
    let r2 = big_r * big_r;
    let k2 = k * k;
    let z2 = z * z;

    // s = sqrt(4 z^2 R^2 - (R^2 + z^2 - k^2)^2), written as a product of exact
    // sums so that no factor loses precision near the boundaries, and as a
    // product of square roots so that it cannot underflow for a tiny `z`. Each
    // factor is >= 0 in this branch; `max` only guards against rounding.
    let s = (sum3(big_r, k, -z) * sum3(big_r, k, z)).max(0.0).sqrt()
        * sum3(z, -big_r, k).max(0.0).sqrt()
        * sum3(z, big_r, -k).max(0.0).sqrt();

    // Half-angles of the two circular segments. The usual form,
    // acos((R^2 - k^2 + z^2) / (2 R z)), loses about half the digits when the
    // angle is near 0 or pi (up to 1e-8 in the flux for k = 1.5); atan2 of the
    // sine and cosine parts does not, and needs no clamp.
    let kappa1 = s.atan2(r2 - k2 + z2);
    let kappa0 = s.atan2(k2 + z2 - r2);

    // s / 2 is the area of the kite with corners at the two centres and the
    // two points where the circles cross.
    r2 * kappa1 + k2 * kappa0 - 0.5 * s
}

/// Flux of a star with no limb darkening while a planet of radius ratio `k`
/// sits a distance `z` from the star's centre on the sky.
///
/// `z` is in stellar radii. Only its size matters, so a negative `z` gives
/// the same flux as `-z`.
///
/// The flux is relative to the star on its own: 1 when the planet is clear of
/// the disk (exactly 1 once `|z| >= 1 + k`), `1 - k^2` when it is fully
/// inside, and 0 when it covers the whole disk (`k >= 1` and
/// `|z| <= k - 1`). A NaN `z` gives NaN.
///
/// This does not know about orbits: whether the planet is in front of the
/// star at all is for the caller to check.
///
/// # Examples
///
/// ```
/// use rustellar_model::{RadiusRatio, flux_uniform};
///
/// let k = RadiusRatio::new(0.1)?;
/// assert!((flux_uniform(0.5, k) - 0.99).abs() < 1e-12); // fully inside
/// assert_eq!(flux_uniform(1.1, k), 1.0); // just touching the limb
/// # Ok::<(), rustellar_model::ParamError>(())
/// ```
#[must_use]
pub fn flux_uniform(z: f64, k: RadiusRatio) -> f64 {
    // `blocked_area` expects z >= 0; a negative z is the same distance.
    1.0 - blocked_area(z.abs(), k.get())
}
