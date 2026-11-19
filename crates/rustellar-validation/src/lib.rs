//! Cross-package comparisons of the rustellar crates with their Python
//! references, on the same data: the numbers the crates' READMEs quote,
//! made again with one command.
//!
//! - [`tls`]: `rustellar-tls` against transitleastsquares 1.32 on injected
//!   transits ([`Kind`] names the sets), rebuilt here from the seeds the
//!   references were run on.
//!
//! Run `cargo run --release -p rustellar-validation --bin compare -- --help`.

mod curves;
mod random;
mod stats;
pub mod tls;

#[cfg(test)]
mod tests;

pub use curves::{Found, Kind, Row, light_curve, rows, transit};
pub use stats::{median, sign_test, timing};
