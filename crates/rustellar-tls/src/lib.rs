//! Transit least squares (TLS): a search for a repeating transit with a
//! limb-darkened template instead of a box (Hippke & Heller 2019).
//!
//! At each trial period the light curve is folded, and a transit template
//! of each trial duration is fitted at every mid-transit time, its depth by
//! weighted least squares. The best fit's chi-squared at each period gives
//! the spectrum, and its peak the best candidate.
//!
//! **Status:** in development. Nothing is public yet.

// Only the tests use these until the public API is built on them. `allow`,
// not `expect`: Rust 1.85 counts as live what later versions find dead.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod cells;
#[cfg(feature = "cuda")]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod cuda;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod durations;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod exact;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod fast;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod fit;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod refine;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod search;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod spectra;
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the tests until the public API")
)]
mod template;

#[cfg(test)]
mod tests;
