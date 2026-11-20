use std::f64::consts::PI;

use rustellar_model::{ImpactParameter, LimbDarkening, RadiusRatio, flux};

use crate::VetError;

/// The shape of a transit, fitted for its depth: 1 at mid-transit, 0 outside
/// the transit.
///
/// - [`Shape::BOX`]: 1 within half a duration of mid-transit (strictly), as a
///   BLS search fits.
/// - [`Shape::transit`]: a limb-darkened transit of a planet of radius ratio
///   `k` crossing at impact parameter `b`, on a circular orbit sized so that
///   first to last contact takes the candidate's duration, divided by its
///   own depth at mid-transit. The flux is `rustellar-model`'s analytic
///   one (Mandel & Agol 2002).
///
/// The default is a limb-darkened transit of `k` 0.05 at `b` 0.5 in front
/// of a Sun-like star (quadratic limb darkening 0.4804, 0.1867, as
/// transitleastsquares' default): a small planet half-way to the limb.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{ImpactParameter, LimbDarkening, RadiusRatio, Shape};
///
/// let central = Shape::transit(
///     RadiusRatio::new(0.1)?,
///     ImpactParameter::new(0.0)?,
///     LimbDarkening::quadratic(0.4, 0.25)?,
/// )?;
/// assert_ne!(central, Shape::BOX);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape(Kind);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Box,
    Transit {
        k: RadiusRatio,
        b: ImpactParameter,
        limb_darkening: LimbDarkening,
    },
}

impl Default for Shape {
    fn default() -> Self {
        Self(Kind::Transit {
            k: RadiusRatio::new(0.05).expect("valid"),
            b: ImpactParameter::new(0.5).expect("valid"),
            limb_darkening: LimbDarkening::quadratic(0.4804, 0.1867).expect("valid"),
        })
    }
}

impl Shape {
    /// A box: 1 in transit, 0 out.
    pub const BOX: Self = Self(Kind::Box);

    /// A limb-darkened transit of radius ratio `k` at impact parameter `b`.
    ///
    /// # Errors
    ///
    /// [`VetError::NoTransit`] if `b` is at least `1 + k`: the planet would
    /// miss the star.
    pub fn transit(
        k: RadiusRatio,
        b: ImpactParameter,
        limb_darkening: LimbDarkening,
    ) -> Result<Self, VetError> {
        if b.get() >= 1.0 + k.get() {
            return Err(VetError::NoTransit);
        }
        Ok(Self(Kind::Transit {
            k,
            b,
            limb_darkening,
        }))
    }

    /// The shape for one candidate's period and duration (in days), ready to
    /// evaluate.
    pub(crate) fn profile(self, period: f64, duration: f64) -> Result<Profile, VetError> {
        let half = 0.5 * duration;
        match self.0 {
            Kind::Box => Ok(Profile::Box { half }),
            Kind::Transit {
                k,
                b,
                limb_darkening,
            } => {
                // Past half the period the separation would turn back
                // before the contacts: no circular orbit has that duration.
                if duration > 0.5 * period {
                    return Err(VetError::DurationTooLongForShape);
                }
                // The orbit's scaled radius a for which the separation
                // sqrt(a^2 sin^2 theta + b^2 cos^2 theta) reaches 1 + k at
                // half the duration.
                let edge = 1.0 + k.get();
                let theta = PI * duration / period;
                let (sin, cos) = theta.sin_cos();
                let bb = b.get() * b.get();
                let a2 = (edge * edge - bb * cos * cos) / (sin * sin);
                let centre = 1.0 - flux(b.get(), k, limb_darkening);
                Ok(Profile::Transit {
                    half,
                    omega: 2.0 * PI / period,
                    a2,
                    bb,
                    k,
                    limb_darkening,
                    centre,
                })
            }
        }
    }
}

/// A shape set for one period and duration.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Profile {
    Box {
        half: f64,
    },
    Transit {
        half: f64,
        omega: f64,
        a2: f64,
        bb: f64,
        k: RadiusRatio,
        limb_darkening: LimbDarkening,
        centre: f64,
    },
}

impl Profile {
    /// The shape at `phase` days from mid-transit (within half a period):
    /// 1 at mid-transit, 0 at and beyond half a duration.
    pub(crate) fn at(&self, phase: f64) -> f64 {
        match *self {
            Self::Box { half } => {
                if phase.abs() < half {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Transit {
                half,
                omega,
                a2,
                bb,
                k,
                limb_darkening,
                centre,
            } => {
                if phase.abs() >= half {
                    return 0.0;
                }
                let (sin, cos) = (omega * phase).sin_cos();
                let z = (a2 * sin * sin + bb * cos * cos).sqrt();
                (1.0 - flux(z, k, limb_darkening)) / centre
            }
        }
    }
}
