use std::f64::consts::PI;

use crate::RadiusRatio;

// Overlap area of a disk of radius `big_r` at the origin and a disk of radius
// `k` whose centre is `z` away (the lens formula). Expects
// `z >= 0`, `k > 0` and `big_r >= 0`; `big_r = 0` gives 0 with no special
// case.
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

    // s = sqrt(4 z^2 R^2 - (R^2 + z^2 - k^2)^2), written as a product so that
    // no factor is a small difference of large numbers near the boundaries.
    // Each factor is >= 0 in this branch; `max` only guards against rounding.
    let s = ((big_r + k - z) * (big_r + k + z) * (z - big_r + k) * (z + big_r - k))
        .max(0.0)
        .sqrt();

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
pub fn flux_uniform(z: f64, k: RadiusRatio) -> f64 {
    // `overlap_area` expects z >= 0; a negative z is the same distance.
    1.0 - overlap_area(z.abs(), k.get(), 1.0) / PI
}
