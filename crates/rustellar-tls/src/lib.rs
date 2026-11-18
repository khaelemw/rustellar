//! Transit least squares (TLS): a search for a repeating transit with a
//! limb-darkened template instead of a box (Hippke & Heller 2019).
//!
//! At each trial period the light curve is folded, and a transit template
//! of each trial duration is fitted at every mid-transit time, its depth by
//! weighted least squares. The best fit's chi-squared at each period gives
//! the spectrum, and its peak the best candidate.
//!
//! **Status:** in development. Nothing is public yet.

// Only the tests use these until the search is built on them.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the tests until the search")
)]
mod durations;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the tests until the search")
)]
mod exact;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the tests until the search")
)]
mod fast;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the tests until the search")
)]
mod fit;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the tests until the search")
)]
mod template;

#[cfg(test)]
mod tests;
