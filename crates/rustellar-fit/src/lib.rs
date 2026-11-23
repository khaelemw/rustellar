//! Transit fits: the transit model of `rustellar-model` fitted to a light
//! curve by weighted least squares, from a starting guess to the best
//! parameters, their standard errors and the quantities derived from them.
//!
//! The model is Mandel & Agol's (2002) with quadratic limb darkening, on a
//! circular orbit or, with [`Parameter::ECosW`] and [`Parameter::ESinW`],
//! an eccentric one, optionally averaged over each point's exposure (Kipping
//! 2010). Times are in days of the light curve's own scale; nothing is tied
//! to an instrument.
//!
//! In this crate:
//!
//! - [`TransitFit`]: the light curve, the starting parameters and
//!   baseline, which [`Parameter`]s are free, and each point's exposure.
//! - [`TransitFit::run`]: Levenberg-Marquardt to the minimum (More 1978;
//!   Nielsen 1999) with geodesic acceleration (Transtrum & Sethna 2012),
//!   by QR throughout and a Jacobian from the transit's geometry, giving
//!   a [`FitResult`].
//! - Standard errors and the [`Covariance`] from the fit's `R`, and the
//!   depth, durations, inclination and stellar density, each an
//!   [`Estimate`] with its error to first order.
//!
//! # Examples
//!
//! ```
//! use rustellar_fit::{Parameter, TransitFit};
//! use rustellar_model::{
//!     Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
//!     TimeDelta, TransitParams,
//! };
//!
//! let start = TransitParams::new(
//!     Epoch::from_days(1.0)?,
//!     TimeDelta::from_days(3.0)?,
//!     RadiusRatio::new(0.1)?,
//!     ScaledSemiMajorAxis::new(10.0)?,
//!     ImpactParameter::new(0.3)?,
//!     LimbDarkening::quadratic(0.4, 0.25)?,
//! )?;
//! let time: Vec<f64> = (0..1000).map(|i| 0.01 * f64::from(i)).collect();
//! let lc = LightCurve::from_days(time, vec![1.0; 1000], vec![1e-3; 1000])?;
//! let fit = TransitFit::new(&lc, start).with_fixed(Parameter::Period);
//! assert_eq!(fit.free().len(), 7);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod covariance;
mod error;
mod jacobian;
mod lm;
mod parameter;
mod problem;
mod result;
mod run;
mod setup;

#[cfg(test)]
mod tests;

pub use covariance::{Covariance, Estimate};
pub use error::FitError;
pub use lm::Termination;
pub use parameter::Parameter;
pub use result::FitResult;
pub use setup::{LimbDarkeningBounds, TransitFit};

// The types a fit takes and returns, so `rustellar-core` and
// `rustellar-model` need not be listed beside this crate.
pub use rustellar_core::{Epoch, LightCurve, LightCurveError, TimeDelta, TimeError};
pub use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, LimbDarkeningError, ParamError, RadiusRatio,
    ScaledSemiMajorAxis, TransitParams,
};
