//! Light curve type and shared parts for the rustellar crates.

mod error;
mod lightcurve;
mod time;

pub use error::{LightCurveError, TimeError};
pub use lightcurve::{FoldedLightCurve, LightCurve};
pub use time::{Duration, Epoch};
