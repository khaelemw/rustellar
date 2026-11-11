//! Light curve type and shared parts for the rustellar crates.

mod error;
mod lightcurve;

pub use error::LightCurveError;
pub use lightcurve::{FoldedLightCurve, LightCurve};
