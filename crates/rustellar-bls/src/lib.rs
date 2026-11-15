//! Box least squares (BLS): a search for a repeating, box-shaped dip in a
//! light curve (Kovacs, Zucker & Mazeh 2002).
//!
//! At each trial period the light curve is folded, and a box (a flat dip of
//! some depth, duration and mid-transit time) is fitted at every position.
//! The best box's improvement in chi-squared over a flat line is the power at
//! that period; the period with the most power is the best candidate.
//!
//! This crate is in development. So far:
//!
//! - the checked inputs, [`Durations`] and [`Periods`];
//! - two period grids: astropy's automatic one, [`AutoPeriod`], and Ofir's
//!   (2014) optimal one, [`OfirPeriod`];
//! - one box, [`TransitBox`], and its exact fit to a light curve,
//!   [`BoxFit`].
//!
//! The search's phase resolution will be a
//! [`NonZeroUsize`](std::num::NonZeroUsize) (`oversample`), so it needs no
//! type of its own.

mod autoperiod;
mod box_fit;
mod error;
mod grid;
mod ofir;
mod sums;

#[cfg(test)]
mod tests;

pub use autoperiod::AutoPeriod;
pub use box_fit::{BoxFit, TransitBox};
pub use error::BlsError;
pub use grid::{Durations, Periods};
pub use ofir::OfirPeriod;
