//! Box least squares (BLS): a search for a repeating, box-shaped dip in a
//! light curve (Kovacs, Zucker & Mazeh 2002).
//!
//! At each trial period the light curve is folded, and a box (a flat dip of
//! some depth, duration and mid-transit time) is fitted at every position.
//! The best box's improvement in chi-squared over a flat line is the power at
//! that period; the period with the most power is the best candidate.
//!
//! The crate provides:
//!
//! - the checked inputs, [`Durations`] and [`Periods`];
//! - two period grids: astropy's automatic one, [`AutoPeriod`], and Ofir's
//!   (2014) optimal one, [`OfirPeriod`];
//! - one box, [`TransitBox`], and its exact fit to a light curve,
//!   [`BoxFit`];
//! - the exact search at one period and duration,
//!   [`TransitBox::best`], over every mid-transit time ([`BestBox`]);
//! - the search over many periods, [`Bls::search`]: binned at every period,
//!   on all cores, with the highest peaks searched again exactly over every
//!   mid-transit time, a continuous range of durations and periods between
//!   the grid's ([`BlsResult`], [`Peak`]), set up by [`SearchOptions`] and
//!   maximising an [`Objective`];
//! - with the `cuda` feature, the same search with its binned step on an
//!   NVIDIA GPU, `CudaBls`: the same peaks, refined boxes and best box, to
//!   the bit.
//!
//! Its accuracy against astropy, transitleastsquares and exact arithmetic,
//! and its speed, are in the README.

mod autoperiod;
mod binned;
mod box_fit;
#[cfg(feature = "cuda")]
mod cuda;
mod error;
mod exact;
mod grid;
mod ofir;
mod parallel;
mod refine;
mod search;
mod sums;
#[cfg(any(feature = "cuda", test))]
mod top;
mod totals;

#[cfg(test)]
mod tests;

pub use autoperiod::AutoPeriod;
pub use box_fit::{BoxFit, TransitBox};
#[cfg(feature = "cuda")]
pub use cuda::{CudaBls, CudaError};
pub use error::BlsError;
pub use exact::BestBox;
pub use grid::{Durations, Periods};
pub use ofir::OfirPeriod;
pub use search::{Bls, BlsResult, Objective, Peak, SearchOptions};

// The README's example, compiled and run as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
