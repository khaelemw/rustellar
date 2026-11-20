//! Detrending light curves: the slow variation of a star (spots, rotation)
//! and of the instrument (drift, scattered light) estimated and divided out,
//! so that a transit search sees a flat light curve.
//!
//! The method is a time-windowed sliding filter, as in wotan (Hippke et al.
//! 2019): for each point, a location of the fluxes within half a window
//! either side is the trend there, and the flux is divided by it; or a
//! least-squares cubic spline per segment, knots a window apart
//! ([`Method::Spline`]). The crate provides:
//!
//! - the settings, [`Detrend`]: the [`Method`] (Tukey's biweight by
//!   default, the median or the mean), the window, where the light curve is
//!   split ([`BreakTolerance`]), the edge cutoff, the window's [`Edges`] and
//!   the threads; the window can be set from the longest transit searched
//!   for ([`Detrend::with_window_for_transit`]);
//! - the trend at each point, [`Detrend::trend`], with an optional mask
//!   (transits left out of every window);
//! - the light curve divided by it, [`Detrend::flatten`] ([`Flattened`]);
//! - the mask of a candidate's transits, [`transit_mask`], to detrend again
//!   with them left out;
//! - both steps around a search's candidate, [`Detrend::flatten_around`]:
//!   flatten, search (any search, as a closure), mask the
//!   [`Candidate`]'s transits, flatten again ([`Masking`], [`Masked`]).
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
//! use rustellar_detrend::{Detrend, LightCurve, TimeDelta};
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

mod around;
mod detrend;
mod double;
mod error;
mod exact_sum;
mod huber_joint;
mod location;
mod m_estimate;
mod mask;
mod order;
mod psi;
mod spline;
mod trend;
mod window;

#[cfg(test)]
mod tests;

pub use around::{Candidate, MASK_PER_DURATION, MASKED_WINDOW_PER_DURATION, Masked, Masking};
pub use detrend::{BreakTolerance, Detrend, Flattened, Method, WINDOW_PER_DURATION};
pub use error::DetrendError;
pub use mask::transit_mask;
pub use window::Edges;

// Core's types, which this crate's API takes and returns: with them the
// crate works on its own, without listing rustellar-core, and always with
// the core it was built on. The same types, not copies.
#[doc(no_inline)]
pub use rustellar_core::{
    Epoch, FoldedLightCurve, LightCurve, LightCurveError, TimeDelta, TimeError,
};

// The README's example, compiled and run as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
