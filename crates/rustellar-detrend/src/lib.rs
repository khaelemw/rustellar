//! Detrending light curves: the slow variation of a star (spots, rotation)
//! and of the instrument (drift, scattered light) estimated and divided out,
//! so that a transit search sees a flat light curve.
//!
//! The method is a time-windowed sliding filter, as in wotan (Hippke et al.
//! 2019): for each point, a location of the fluxes within half a window
//! either side is the trend there, and the flux is divided by it. The
//! crate provides:
//!
//! - the settings, [`Detrend`]: the [`Method`] (Tukey's biweight by
//!   default, the median or the mean), the window, where the light curve is
//!   split ([`BreakTolerance`]), the edge cutoff, the window's [`Edges`] and
//!   the threads;
//! - the trend at each point, [`Detrend::trend`], with an optional mask
//!   (transits left out of every window);
//! - the light curve divided by it, [`Detrend::flatten`] ([`Flattened`]).
//!
//! The biweight is solved to full double precision (correctly rounded
//! wherever its defining sum's sign can be told from zero), the median and
//! the mean are correctly rounded, and every window edge is decided exactly.
//! The results are the same, to the bit, on any number of threads. Accuracy
//! against wotan and exact arithmetic, and speed, are in the README.
//!
//! # Examples
//!
//! ```
//! use rustellar_core::{LightCurve, TimeDelta};
//! use rustellar_detrend::Detrend;
//!
//! let time: Vec<f64> = (0..720).map(|i| i as f64 / 720.0).collect();
//! let flux: Vec<f64> = time.iter().map(|&t| 100.0 + 2.0 * t).collect();
//! let lc = LightCurve::from_btjd(time, flux, vec![0.1; 720])?;
//!
//! let flat = Detrend::default()
//!     .with_window(TimeDelta::from_days(0.25)?)
//!     .flatten(&lc, None)?;
//! assert!(flat.light_curve().flux()[360..365].iter().all(|f| (f - 1.0).abs() < 1e-12));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod biweight;
mod detrend;
mod error;
mod exact_sum;
mod location;
mod trend;
mod window;

#[cfg(test)]
mod tests;

pub use detrend::{BreakTolerance, Detrend, Flattened, Method};
pub use error::DetrendError;
pub use window::Edges;

// The README's example, compiled and run as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
