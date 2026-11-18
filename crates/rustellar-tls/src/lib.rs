//! Transit least squares (TLS): a search for a repeating transit with a
//! limb-darkened template instead of a box (Hippke & Heller 2019).
//!
//! At each trial period the light curve is folded, and a transit template
//! of each trial duration is fitted at every mid-transit time, its depth by
//! weighted least squares. The best fit's chi-squared at each period gives
//! the spectrum, and its peak the best candidate.
//!
//! **Status:** in development. Nothing is public yet.
