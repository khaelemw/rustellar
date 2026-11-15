//! Box least squares (BLS): a search for a repeating, box-shaped dip in a
//! light curve (Kovacs, Zucker & Mazeh 2002).
//!
//! At each trial period the light curve is folded, and a box (a flat dip of
//! some depth, duration and mid-transit time) is fitted at every position.
//! The best box's improvement in chi-squared over a flat line is the power at
//! that period; the period with the most power is the best candidate.
//!
//! This crate is in development. So far: the checked inputs, [`Durations`]
//! and [`Periods`], astropy's automatic period grid, [`AutoPeriod`], and
//! Ofir's (2014) optimal period grid, [`OfirPeriod`]. The search's phase
//! resolution will be a [`NonZeroUsize`](std::num::NonZeroUsize)
//! (`oversample`), so it needs no type of its own.

mod autoperiod;
mod error;
mod grid;
mod ofir;

#[cfg(test)]
mod tests;

pub use autoperiod::AutoPeriod;
pub use error::BlsError;
pub use grid::{Durations, Periods};
pub use ofir::OfirPeriod;
