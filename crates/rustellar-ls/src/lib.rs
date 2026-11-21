//! The Lomb-Scargle periodogram: how well a sinusoid of each trial
//! frequency fits an unevenly sampled series.
//!
//! For each frequency a sinusoid (and, by default, a floating mean) is
//! fitted to the data by weighted least squares, and the power is how much
//! it lowers chi-squared (Lomb 1976; Scargle 1982; Zechmeister & Kurster
//! 2009). It finds rotation periods, pulsations and any other periodic
//! signal in data of any cadence.
//!
//! Frequencies are in cycles per day of the light curve's own time scale.
//! Nothing is tied to an instrument.
//!
//! **Status:** in development. So far:
//!
//! - [`Frequencies`]: the trial frequencies, an even grid or any list,
//!   checked once; [`AutoFrequency`], astropy's automatic grid.
//! - [`LombScargle`]: the light curve set up with a [`Mean`] (floating by
//!   default), and its power at any frequencies in any [`Normalization`],
//!   exactly ([`Periodogram`]).
//!
//! # Examples
//!
//! ```
//! use rustellar_ls::{AutoFrequency, LightCurve};
//!
//! let time: Vec<f64> = (0..200).map(|i| 0.05 * f64::from(i)).collect();
//! let lc = LightCurve::from_days(time, vec![1.0; 200], vec![1e-3; 200])?;
//! let grid = AutoFrequency::default().frequencies(&lc)?;
//! assert!(grid.max() > 9.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod frequencies;
mod periodogram;
mod sums;

#[cfg(test)]
mod tests;

pub use error::LsError;
pub use frequencies::{AutoFrequency, Frequencies};
pub use periodogram::{LombScargle, Mean, Normalization, Periodogram};

// Core's types, which this crate's API takes and returns: with them the
// crate works on its own, without listing rustellar-core, and always with
// the core it was built on. The same types, not copies.
#[doc(no_inline)]
pub use rustellar_core::{Epoch, LightCurve, LightCurveError, TimeDelta, TimeError};

// The README's example, compiled and run as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
