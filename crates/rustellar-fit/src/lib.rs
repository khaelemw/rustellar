//! Transit fits: the transit model of `rustellar-model` fitted to a light
//! curve by weighted least squares, from a starting guess to the best
//! parameters, their standard errors and the quantities derived from them.
//!
//! The model is Mandel & Agol's (2002) with quadratic limb darkening, on a
//! circular orbit, optionally averaged over each point's exposure (Kipping
//! 2010). Times are in days of the light curve's own scale; nothing is tied
//! to an instrument.
//!
//! **Status:** in development. So far:
//!
//! - [`TransitFit`]: the light curve, the starting parameters and
//!   baseline, which [`Parameter`]s are free, and each point's exposure.
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

mod error;
mod parameter;
mod setup;

#[cfg(test)]
mod tests;

pub use error::FitError;
pub use parameter::Parameter;
pub use setup::TransitFit;

// The types a fit takes and returns, so `rustellar-core` and
// `rustellar-model` need not be listed beside this crate.
pub use rustellar_core::{Epoch, LightCurve, LightCurveError, TimeDelta, TimeError};
pub use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, LimbDarkeningError, ParamError, RadiusRatio,
    ScaledSemiMajorAxis, TransitParams,
};
