use crate::error::LimbDarkeningError;

/// How the star's brightness falls from the centre of its disk to the edge.
///
/// The intensity is `I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2`, where `mu` is 1
/// at the disk centre and 0 at the edge. Uniform is `u1 = u2 = 0`, and linear
/// is `u2 = 0`.
///
/// Coefficients are rejected only if they make the intensity negative
/// somewhere on the disk. Negative `u1` (brighter towards the edge) is allowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimbDarkening(Law);

// Private, so a law can only be made through the checked constructors.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Law {
    Uniform,
    Linear { u1: f64 },
    Quadratic { u1: f64, u2: f64 },
}

impl LimbDarkening {
    /// A star with the same brightness everywhere on its disk.
    pub fn uniform() -> Self {
        Self(Law::Uniform)
    }

    /// The linear law, `I(mu) = 1 - u1 (1 - mu)`.
    ///
    /// # Errors
    ///
    /// - [`LimbDarkeningError::NonFinite`] if `u1` is NaN or infinite.
    /// - [`LimbDarkeningError::NegativeIntensity`] if `u1` is greater than 1.
    pub fn linear(u1: f64) -> Result<Self, LimbDarkeningError> {
        check_intensity(u1, 0.0)?;
        Ok(Self(Law::Linear { u1 }))
    }

    /// The quadratic law, `I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2`.
    ///
    /// # Errors
    ///
    /// - [`LimbDarkeningError::NonFinite`] if `u1` or `u2` is NaN or infinite.
    /// - [`LimbDarkeningError::NegativeIntensity`] if the intensity is
    ///   negative anywhere on the disk.
    pub fn quadratic(u1: f64, u2: f64) -> Result<Self, LimbDarkeningError> {
        check_intensity(u1, u2)?;
        Ok(Self(Law::Quadratic { u1, u2 }))
    }

    /// The law as quadratic coefficients `(u1, u2)`: `(0, 0)` for uniform and
    /// `(u1, 0)` for linear.
    pub fn quadratic_coefficients(self) -> (f64, f64) {
        match self.0 {
            Law::Uniform => (0.0, 0.0),
            Law::Linear { u1 } => (u1, 0.0),
            Law::Quadratic { u1, u2 } => (u1, u2),
        }
    }

    // Brightness at a point on the disk, relative to the centre.
    //
    // `mu = sqrt(1 - r^2)`, where `r` is the distance from the disk centre in
    // stellar radii, so `mu` is 1 at the centre and 0 at the edge. Crate-only,
    // so every caller passes a `mu` in `[0, 1]`; there the result is 1 at the
    // centre and never negative.
    pub(crate) fn intensity(self, mu: f64) -> f64 {
        let (u1, u2) = self.quadratic_coefficients();

        1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) * (1.0 - mu)
    }

    /// Total brightness of the disk, relative to a uniform disk with the same
    /// central brightness.
    ///
    /// This is the integral of the intensity over the disk, divided by `pi`:
    /// `1 - u1/3 - u2/6`. It is 1 for a uniform star and always greater than
    /// 0, because the intensity is never negative and is 1 at the centre.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::LimbDarkening;
    ///
    /// assert_eq!(LimbDarkening::uniform().disk_integral(), 1.0);
    ///
    /// let ld = LimbDarkening::linear(0.6)?;
    /// assert!((ld.disk_integral() - 0.8).abs() < 1e-12);
    /// # Ok::<(), rustellar_model::LimbDarkeningError>(())
    /// ```
    pub fn disk_integral(self) -> f64 {
        let (u1, u2) = self.quadratic_coefficients();

        1.0 - (u1 / 3.0) - (u2 / 6.0)
    }
}

// Checks that `I(x) = 1 - u1 x - u2 x^2` is not negative for `x = 1 - mu` in
// `[0, 1]`. `I(0) = 1`, so the minimum is at the edge (`x = 1`) unless the
// curve opens upwards (`u2 < 0`) with its lowest point inside the range.
fn check_intensity(u1: f64, u2: f64) -> Result<(), LimbDarkeningError> {
    if !u1.is_finite() || !u2.is_finite() {
        return Err(LimbDarkeningError::NonFinite);
    }

    let negative_at_edge = u1 + u2 > 1.0;

    // Lowest point at x = -u1 / (2 u2), with value 1 + u1^2 / (4 u2).
    let lowest_inside = u2 < 0.0 && u1 > 0.0 && u1 < -2.0 * u2;
    let negative_inside = lowest_inside && u1 * u1 + 4.0 * u2 > 0.0;

    if negative_at_edge || negative_inside {
        return Err(LimbDarkeningError::NegativeIntensity);
    }

    Ok(())
}
