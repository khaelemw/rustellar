//! Vetting transit candidates: the evidence that a search's period is the
//! real one, shown rather than decided.
//!
//! A transit search reports one best period, and sometimes the true period
//! is a simple ratio of it: half (every other predicted transit is empty),
//! double (every other real transit was skipped), or a third, three times,
//! two thirds or three halves. Its chi-squared alone often cannot tell
//! these apart; the evidence transit by transit can. This crate will give,
//! for a candidate and each of its aliases, the fit and that evidence: odd
//! and even depths, how many predicted transits have data, each transit's
//! depth, and the signal to noise.
//!
//! **Status:** in development. So far:
//!
//! - [`Candidate`]: a period, mid-transit time, duration and optional
//!   depth, checked; from `rustellar-bls`'s best box with the `bls`
//!   feature and `rustellar-tls`'s candidate with the `tls` feature.
//! - [`fit_depth`]: the depth at a fixed period, mid-transit time and
//!   duration, for a [`Shape`] (limb-darkened by default, or a box), with
//!   its standard error ([`Fit`]).
//! - [`refine`]: the mid-transit time and duration refined at the
//!   candidate's period, exactly for a box ([`Refined`]).
//! - [`evidence`]: each predicted transit's points, coverage and depth,
//!   and the odd and even transits' depths compared ([`Evidence`]).
//! - [`signal_to_noise`]: the signal to noise, white and with red noise,
//!   transit by transit, and whether the transits share one depth
//!   ([`SignalToNoise`]).
//!
//! # Examples
//!
//! ```
//! use rustellar_vet::{Candidate, Epoch, TimeDelta};
//!
//! let candidate = Candidate::new(
//!     TimeDelta::from_days(3.0)?,
//!     Epoch::from_btjd(1000.0)?,
//!     TimeDelta::from_hours(2.4)?,
//! )?;
//! assert_eq!(candidate.depth(), None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

mod candidate;
mod error;
mod evidence;
mod fit;
mod grouped;
mod noise;
mod refine;
mod shape;
mod sum;

#[cfg(test)]
mod tests;

pub use candidate::Candidate;
pub use error::VetError;
pub use evidence::{Evidence, OddEven, Transit, evidence};
pub use fit::{Fit, fit_depth};
pub use noise::{Noise, SignalToNoise, TransitSnr, signal_to_noise};
pub use refine::{Refined, refine};
pub use shape::Shape;

// Core's types, which this crate's API takes and returns: with them the
// crate works on its own, without listing rustellar-core, and always with
// the core it was built on. The same types, not copies.
#[doc(no_inline)]
pub use rustellar_core::{
    Epoch, FoldedLightCurve, LightCurve, LightCurveError, TimeDelta, TimeError,
};

// The model's types a limb-darkened [`Shape`] is built from, re-exported
// for the same reason.
#[doc(no_inline)]
pub use rustellar_model::{
    ImpactParameter, LimbDarkening, LimbDarkeningError, ParamError, RadiusRatio,
};

// The README's example, compiled and run as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
