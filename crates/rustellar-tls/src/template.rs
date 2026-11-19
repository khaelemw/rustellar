//! The transit template: the shape of a limb-darkened transit, 1 at
//! mid-transit and 0 at first and last contact, as a function of the time
//! from mid-transit in units of the transit's duration.
//!
//! The planet crosses the star in a straight line, so the shape is the same
//! at every period; one template serves every trial period and duration,
//! stretched to each duration. The flux is `rustellar-model`'s, evaluated
//! where it is needed, never resampled.

use std::f64::consts::FRAC_PI_2;

use rustellar_model::{LimbDarkening, RadiusRatio, flux};

/// A transit's shape `s(x)`, for `x` the time from mid-transit over the
/// duration from first to last contact.
///
/// With radius ratio `k` and impact parameter `b`, the planet's separation
/// from the star's centre is `z(x) = sqrt(b² + ((1 + k)² - b²) (2x)²)`, which
/// is `b` at mid-transit and `1 + k` at contact (`|x| = 1/2`). Then
///
/// ```text
/// s(x) = (1 - F(z(x))) / (1 - F(b))   for |x| < 1/2,   else 0,
/// ```
///
/// where `F` is the flux of [`rustellar_model::flux`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Template {
    k: RadiusRatio,
    b: f64,
    ld: LimbDarkening,
    /// `(1 + k)² - b²`, the squared half-chord in stellar radii.
    chord: f64,
    /// `1 - F(b)`, the depth at mid-transit.
    depth: f64,
}

/// TLS's default planet: `k = 0.03`, `a = 23.1` stellar radii, inclination
/// 89.21 degrees, so `b = 23.1 cos(89.21°)` (to the nearest double, from a
/// 40-digit evaluation).
const DEFAULT_B: f64 = 0.318_495_043_365_240_5;

/// TLS's grazing template: its default planet at `b = 0.99`.
const GRAZING_B: f64 = 0.99;

impl Template {
    /// The template of a planet with radius ratio `k` and impact parameter
    /// `b` in front of a star with limb darkening `ld`, or `None` if it does
    /// not transit (`b` not below `1 + k`) or `b` is negative or not finite.
    pub(crate) fn new(k: RadiusRatio, b: f64, ld: LimbDarkening) -> Option<Self> {
        let reach = 1.0 + k.get();
        if !(b >= 0.0 && b < reach) {
            return None;
        }
        let chord = (reach - b) * (reach + b);
        let depth = 1.0 - flux(b, k, ld);
        // A transit that dims nothing (a planet too small for the flux to
        // change in doubles) has no shape to fit.
        (depth > 0.0).then_some(Self {
            k,
            b,
            ld,
            chord,
            depth,
        })
    }

    /// TLS's default template: `k = 0.03`, `b = 0.318…`, quadratic limb
    /// darkening `(0.4804, 0.1867)` (a Sun-like star in the Kepler band).
    pub(crate) fn tls_default() -> Self {
        Self::tls_planet(DEFAULT_B)
    }

    /// TLS's grazing template: the default planet at `b = 0.99`.
    pub(crate) fn tls_grazing() -> Self {
        Self::tls_planet(GRAZING_B)
    }

    /// TLS's box template (a steep trapezoid): `k = 0.1`, `b = 0`, no limb
    /// darkening.
    pub(crate) fn tls_box() -> Self {
        let k = RadiusRatio::new(0.1).expect("0.1 is a valid radius ratio");
        Self::new(k, 0.0, LimbDarkening::uniform()).expect("a central transit")
    }

    fn tls_planet(b: f64) -> Self {
        let k = RadiusRatio::new(0.03).expect("0.03 is a valid radius ratio");
        let ld = LimbDarkening::quadratic(0.4804, 0.1867).expect("TLS's coefficients are valid");
        Self::new(k, b, ld).expect("TLS's planets transit")
    }

    /// The shape at `x`, the time from mid-transit over the duration: 1 at
    /// `x = 0`, 0 for `|x| >= 1/2`, in `[0, 1]` everywhere.
    pub(crate) fn shape(&self, x: f64) -> f64 {
        let two_x = 2.0 * x;
        if two_x.is_nan() || two_x.abs() >= 1.0 {
            return 0.0;
        }
        let z = self.b.mul_add(self.b, self.chord * two_x * two_x).sqrt();
        let dimming = 1.0 - flux(z, self.k, self.ld);
        (dimming / self.depth).clamp(0.0, 1.0)
    }

    /// The limb darkening of the star the template was made for.
    pub(crate) fn ld(&self) -> LimbDarkening {
        self.ld
    }

    /// The depth at mid-transit, `1 - F(b)`: the template's shape times this
    /// is the dimming of the planet it was made from.
    #[cfg(test)]
    pub(crate) fn depth(&self) -> f64 {
        self.depth
    }
}

/// A template's shape at `x`, the time from mid-transit over the duration.
pub(crate) trait Shape {
    fn shape(&self, x: f64) -> f64;
}

impl Shape for Template {
    fn shape(&self, x: f64) -> f64 {
        Template::shape(self, x)
    }
}

/// A template sampled densely, for the many fits of refinement.
///
/// Near first and second contact the shape goes as a distance to the
/// power 3/2, on both sides of second contact (the limb darkening's own
/// square root at the limb inside it), which linear interpolation in time
/// follows only to about `h^1.5`. So the shape is sampled in angles that
/// make it smooth there: inside the disc evenly in `φ`, with
/// `z = b + (1 - k - b) sin φ`, `φ` in `[0, π/2]`; across ingress evenly in
/// `θ`, with `z = 1 + k cos θ`. At each contact the distance goes as the
/// square of the angle's. With 2^16 intervals each it is within 2e-10 of the
/// template (measured in the tests), against the exact template's
/// microsecond a call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Sampled {
    b: f64,
    k: f64,
    chord: f64,
    /// `z` at second contact, `1 - k`, or `b` for a grazing transit (which
    /// has no inside).
    inner_z: f64,
    /// The shape at `φ = j (π / 2) / M`, and `M / (π / 2)`.
    inner: Vec<f64>,
    inner_scale: f64,
    /// The shape at `θ = j θ_max / M` (`θ_max` where `z = max(b, 1 - k)`),
    /// and `M / θ_max`.
    ingress: Vec<f64>,
    ingress_scale: f64,
}

/// Intervals in each part of a [`Sampled`].
pub(crate) const SAMPLED_INTERVALS: usize = 1 << 16;

/// Linear interpolation in `values` at `u` samples from the first, `u` in
/// `[0, len - 1]`.
fn interpolate(values: &[f64], u: f64) -> f64 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0 <= u <= len - 1"
    )]
    let j = (u as usize).min(values.len() - 2);
    #[expect(clippy::cast_precision_loss, reason = "a sample index")]
    let f = u - j as f64;
    f.mul_add(values[j + 1] - values[j], values[j])
}

impl Template {
    /// The shape where the planet's centre is `z` from the star's.
    fn shape_at(&self, z: f64) -> f64 {
        ((1.0 - flux(z, self.k, self.ld)) / self.depth).clamp(0.0, 1.0)
    }
}

impl Sampled {
    /// `template` at `intervals` intervals in each part.
    pub(crate) fn new(template: &Template, intervals: usize) -> Self {
        let (b, k, chord) = (template.b, template.k.get(), template.chord);
        let inner_z = b.max(1.0 - k);
        let theta_max = ((inner_z - 1.0) / k).clamp(-1.0, 1.0).acos();
        #[expect(clippy::cast_precision_loss, reason = "a sample count")]
        let m = intervals as f64;
        #[expect(clippy::cast_precision_loss, reason = "a sample index")]
        let inner = (0..=intervals)
            .map(|j| template.shape_at((inner_z - b).mul_add((FRAC_PI_2 * j as f64 / m).sin(), b)))
            .collect();
        #[expect(clippy::cast_precision_loss, reason = "a sample index")]
        let ingress = (0..=intervals)
            .map(|j| template.shape_at(k.mul_add((theta_max * j as f64 / m).cos(), 1.0)))
            .collect();
        Self {
            b,
            k,
            chord,
            inner_z,
            inner,
            inner_scale: m / FRAC_PI_2,
            ingress,
            ingress_scale: m / theta_max,
        }
    }
}

impl Shape for Sampled {
    /// 0 for `|x| >= 1/2` (and NaN).
    fn shape(&self, x: f64) -> f64 {
        let v = 2.0 * x.abs();
        if v.is_nan() || v >= 1.0 {
            return 0.0;
        }
        let z = self.b.mul_add(self.b, self.chord * v * v).sqrt();
        if z < self.inner_z {
            let phi = ((z - self.b) / (self.inner_z - self.b)).min(1.0).asin();
            return interpolate(&self.inner, phi * self.inner_scale);
        }
        let theta = ((z - 1.0) / self.k).clamp(-1.0, 1.0).acos();
        interpolate(&self.ingress, theta * self.ingress_scale)
    }
}
