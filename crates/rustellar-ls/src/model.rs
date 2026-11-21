//! The sinusoid fitted at one frequency, with its standard errors.

use crate::periodogram::{Buffer, full_rank, solve};
use crate::sums::{phase, split};
use crate::{LombScargle, LsError, Mean};

/// The sinusoid fitted by weighted least squares at one frequency:
/// `offset + cosine cos(2 pi f (t - t0)) + sine sin(2 pi f (t - t0))`, or
/// `offset + amplitude cos(2 pi f (t - t0) - phase)`, with `t0` the
/// [`reference`](Self::reference) time; and the standard errors from the
/// flux's errors (not scaled by the fit's chi-squared).
///
/// # Examples
///
/// ```
/// use rustellar_ls::{LightCurve, LombScargle, Mean};
///
/// let time: Vec<f64> = (0..200).map(|i| 0.1 * f64::from(i) + 0.01 * f64::from(i % 3)).collect();
/// let flux = time.iter().map(|t| 3.0 + 0.5 * (std::f64::consts::TAU * 0.8 * t).cos()).collect();
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 200])?;
/// let model = LombScargle::new(&lc, Mean::Floating)?.model(0.8)?;
/// assert!((model.offset() - 3.0).abs() < 1e-12);
/// assert!((model.amplitude() - 0.5).abs() < 1e-12);
/// assert!((model.value(1.25) - (3.0 + 0.5 * (std::f64::consts::TAU * 1.0).cos())).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sinusoid {
    frequency: f64,
    reference: f64,
    offset: f64,
    cosine: f64,
    sine: f64,
    offset_error: f64,
    /// The covariance of `(cosine, sine)`, `[aa, ab, bb]`; none where the
    /// two terms are the same at the points.
    covariance: Option<[f64; 3]>,
}

impl Sinusoid {
    /// The frequency, in cycles per day.
    #[must_use]
    pub const fn frequency(&self) -> f64 {
        self.frequency
    }

    /// The time the phase is measured from, in days: the middle of the
    /// light curve's span.
    #[must_use]
    pub const fn reference(&self) -> f64 {
        self.reference
    }

    /// The constant: fitted with a floating mean, the weighted mean with
    /// the mean subtracted, zero with none.
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    /// The cosine's coefficient.
    #[must_use]
    pub const fn cosine(&self) -> f64 {
        self.cosine
    }

    /// The sine's coefficient.
    #[must_use]
    pub const fn sine(&self) -> f64 {
        self.sine
    }

    /// The semi-amplitude, `hypot(cosine, sine)`.
    #[must_use]
    pub fn amplitude(&self) -> f64 {
        self.cosine.hypot(self.sine)
    }

    /// The phase in radians, from `-pi` to `pi`: `atan2(sine, cosine)`.
    #[must_use]
    pub fn phase(&self) -> f64 {
        self.sine.atan2(self.cosine)
    }

    /// The offset's standard error: of the fit with a floating mean, of
    /// the weighted mean with it subtracted, zero with none.
    #[must_use]
    pub const fn offset_error(&self) -> f64 {
        self.offset_error
    }

    /// The amplitude's standard error, to first order; infinite where the
    /// sinusoid's two terms are the same at the points (a degenerate
    /// frequency).
    #[must_use]
    pub fn amplitude_error(&self) -> f64 {
        let (a, b) = (self.cosine, self.sine);
        self.covariance.map_or(f64::INFINITY, |[aa, ab, bb]| {
            ((a * a * aa + 2.0 * a * b * ab + b * b * bb) / (a * a + b * b)).sqrt()
        })
    }

    /// The phase's standard error in radians, to first order; infinite
    /// where the amplitude's is.
    #[must_use]
    pub fn phase_error(&self) -> f64 {
        let (a, b) = (self.cosine, self.sine);
        let r2 = a * a + b * b;
        self.covariance.map_or(f64::INFINITY, |[aa, ab, bb]| {
            ((b * b * aa - 2.0 * a * b * ab + a * a * bb) / (r2 * r2)).sqrt()
        })
    }

    /// The sinusoid at `time`, in days, its phase exact.
    #[must_use]
    pub fn value(&self, time: f64) -> f64 {
        let t = time - self.reference;
        let ((cb, cx), (sb, sx)) = phase(self.frequency, split(self.frequency), t, split(t));
        self.offset + self.cosine * (cb + cx) + self.sine * (sb + sx)
    }
}

impl LombScargle {
    /// The sinusoid fitted at `frequency`, in cycles per day, with the
    /// [`Mean`] set up: the fit behind the power there, with its standard
    /// errors (astropy's `model_parameters`, with errors).
    ///
    /// # Errors
    ///
    /// [`LsError::NonFinite`] or [`LsError::NotPositive`] unless
    /// `frequency` is finite and above zero.
    pub fn model(&self, frequency: f64) -> Result<Sinusoid, LsError> {
        if !frequency.is_finite() {
            return Err(LsError::NonFinite { name: "frequency" });
        }
        if frequency <= 0.0 {
            return Err(LsError::NotPositive { name: "frequency" });
        }
        let mut buffer = Buffer::new(self.time.len());
        let (w, c_mean, s_mean) = self.centred(frequency, &mut buffer);
        let [cc, cs, ss, yc, ys] = self.sums(&buffer);
        #[expect(clippy::cast_precision_loss, reason = "a point count")]
        let n = self.time.len() as f64;
        let (cosine, sine) = solve(cc, cs, ss, yc, ys, n);
        // The covariance in the flux's own units: the inverse of the
        // normal equations with weights 1 / e^2, the scaled ones times
        // `scale`.
        let det = cc.mul_add(ss, -(cs * cs));
        let covariance = full_rank(cc, ss, det, n).then(|| {
            let k = 1.0 / (det * self.scale);
            [ss * k, -cs * k, cc * k]
        });
        // The mean's variance, and the flux left as a constant once the
        // centred terms are fitted (zero to rounding: the flux is centred).
        let mean_variance = 1.0 / (w * self.scale);
        let (offset, offset_variance) = match self.mean {
            Mean::Floating => {
                let left = self
                    .weight
                    .iter()
                    .zip(&self.flux)
                    .map(|(w, y)| w * y)
                    .sum::<f64>()
                    / w;
                let terms = covariance.map_or(f64::INFINITY, |[aa, ab, bb]| {
                    c_mean * c_mean * aa + 2.0 * c_mean * s_mean * ab + s_mean * s_mean * bb
                });
                (
                    self.level + left - cosine * c_mean - sine * s_mean,
                    mean_variance + terms,
                )
            }
            Mean::Subtracted => (self.level, mean_variance),
            Mean::Zero => (0.0, 0.0),
        };
        Ok(Sinusoid {
            frequency,
            reference: self.centre,
            offset,
            cosine,
            sine,
            offset_error: offset_variance.sqrt(),
            covariance,
        })
    }
}
