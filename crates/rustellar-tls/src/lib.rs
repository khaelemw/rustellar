//! Transit least squares (TLS): a search for a repeating transit with a
//! limb-darkened template instead of a box (Hippke & Heller 2019).
//!
//! At each trial period a transit template of each trial duration is fitted
//! at every mid-transit time, its depth by weighted least squares. The best
//! fit's chi-squared at each period gives the spectrum, and its highest
//! peaks, refined, the best candidate.
//!
//! Beyond TLS, the candidate comes with standard errors (from the fit's
//! curvature, and by bootstrap, which hold where the signal is weak), each
//! transit's depth, odd and even transits compared, a signal to noise for
//! correlated noise, and its shape with the planet's size and impact
//! parameter free.
//! The search runs on every core, or on an NVIDIA GPU with the `cuda`
//! feature, with the same results to the bit.
//!
//! # Examples
//!
//! ```
//! use rustellar_tls::{LightCurve, OfirPeriod, SearchOptions, Tls};
//!
//! // 12 days at 30 minutes with a 0.5% transit every 3 days.
//! let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 48.0).collect();
//! let flux: Vec<f64> = time
//!     .iter()
//!     .map(|&t| if (t - 0.7).rem_euclid(3.0) < 0.1 { 0.995 } else { 1.0 })
//!     .collect();
//! let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 576])?;
//!
//! let periods = OfirPeriod::default().periods(&lc)?;
//! let result = Tls::new(&lc)?.search(&periods, &SearchOptions::default())?;
//! let best = result.best().expect("a transit");
//! assert!((best.period.as_days() - 3.0).abs() < 0.01);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod bootstrap;
mod cells;
#[cfg(feature = "cuda")]
mod cuda;
mod durations;
mod error;
mod errors;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "the oracle the tests check against")
)]
mod exact;
mod fast;
mod fit;
mod noise;
mod options;
mod refine;
mod result;
mod search;
mod shape;
mod spectra;
mod template;
mod tls;

#[cfg(test)]
mod tests;

pub use error::TlsError;
pub use options::{Method, SearchOptions, Stars, Template};
pub use result::{
    Candidate, OddEven, PeriodFit, ShapeUncertainties, SignalToNoise, Spectra, TlsResult,
    TransitDepth, TransitShape, Uncertainties,
};
pub use tls::Tls;

// Core's types, which this crate's API takes and returns: with them the
// crate works on its own, without listing rustellar-core, and always with
// the core it was built on. The same types, not copies.
#[doc(no_inline)]
pub use rustellar_core::{
    Device, Epoch, FoldedLightCurve, GridError, LightCurve, LightCurveError, OfirPeriod, Periods,
    TimeDelta, TimeError,
};

// The README's example, compiled and run as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
