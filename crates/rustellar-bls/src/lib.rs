//! Box least squares (BLS): a search for a repeating, box-shaped dip in a
//! light curve (Kovacs, Zucker & Mazeh 2002).
//!
//! At each trial period the light curve is folded, and a box (a flat dip of
//! some depth, duration and mid-transit time) is fitted at every position.
//! The best box's improvement in chi-squared over a flat line is the power at
//! that period; the period with the most power is the best candidate.
//!
//! This crate is in development: nothing is implemented yet.
