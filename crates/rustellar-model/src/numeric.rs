use std::f64::consts::PI;
use std::num::NonZeroUsize;

use crate::uniform::overlap_area;
use crate::{LimbDarkening, RadiusRatio};

/// Flux of a limb-darkened star while a planet of radius ratio `k` sits a
/// distance `z` from the star's centre on the sky, found by slicing the star
/// into rings.
///
/// The part of the star the planet could cover, from radius `|z| - k` to
/// `|z| + k`, is cut into `n_rings` rings. Each ring adds the area of planet
/// inside it times the star's brightness at the ring's middle. The rings are
/// equal steps in `asin(r)`, so they get narrower towards the limb, where the
/// brightness changes fastest. This works for any limb-darkening law, but is
/// slow; it is the reference that faster methods are checked against.
///
/// `z` is in stellar radii, and only its size matters, as for
/// [`flux_uniform`](crate::flux_uniform). The flux is relative to the star on
/// its own: exactly 1 once `|z| >= 1 + k`, and exactly 0 when the planet
/// covers the whole disk. A NaN `z` gives NaN.
///
/// The error falls as `1 / n_rings^2`. With 1000 rings it is below `1e-9`
/// for `k = 0.1` and typical quadratic coefficients, and below `1e-6`
/// (1 ppm) for planets as large as `k = 1.5` with strong limb darkening. For
/// a uniform star the result matches [`flux_uniform`](crate::flux_uniform) to
/// rounding error for any number of rings.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use rustellar_model::{LimbDarkening, RadiusRatio, flux_ring_sum};
///
/// let k = RadiusRatio::new(0.1)?;
/// let ld = LimbDarkening::quadratic(0.3, 0.25)?;
/// let rings = NonZeroUsize::new(1000).unwrap();
///
/// let flux = flux_ring_sum(0.5, k, ld, rings);
/// assert!((flux - 0.988_887_01).abs() < 1e-8);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn flux_ring_sum(z: f64, k: RadiusRatio, ld: LimbDarkening, n_rings: NonZeroUsize) -> f64 {
    // `max` and `min` below would quietly turn a NaN into a number.
    if z.is_nan() {
        return f64::NAN;
    }

    let z = z.abs();
    let k = k.get();

    // The planet covers the whole disk. The sum would only come close to 0
    // here (it can even land slightly below), so return 0 exactly.
    if z <= k - 1.0 {
        return 0.0;
    }

    // Only rings between these radii can overlap the planet.
    let inner = (z - k).max(0.0);
    let outer = (z + k).min(1.0);

    if outer <= inner {
        return 1.0;
    }

    // Rings are equal steps in the angle `theta = asin(r)`, not in `r`. Then
    // `mu = cos(theta)`, which is smooth in `theta`, whereas
    // `mu = sqrt(1 - r^2)` is infinitely steep at the limb. The rings come out
    // narrower towards the limb, where the brightness changes fastest.
    let n = n_rings.get();
    let theta_inner = inner.asin();
    let theta_span = outer.asin() - theta_inner;
    let mut blocked = 0.0;
    let mut area_inside = overlap_area(z, k, inner);

    for i in 1..=n {
        // Each angle is worked out from `theta_inner` rather than by adding a
        // step again and again, which would drift. The last edge is set
        // exactly, since `sin(asin(outer))` can be off by a rounding error.
        let ring_outer = if i == n {
            outer
        } else {
            (theta_inner + theta_span * i as f64 / n as f64).sin()
        };
        let theta_middle = theta_inner + theta_span * (i as f64 - 0.5) / n as f64;

        let mu = theta_middle.cos();
        let area_outside = overlap_area(z, k, ring_outer);

        blocked += ld.intensity(mu) * (area_outside - area_inside);
        area_inside = area_outside;
    }

    1.0 - blocked / (PI * ld.disk_integral())
}
