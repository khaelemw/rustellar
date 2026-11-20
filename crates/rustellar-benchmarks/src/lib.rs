//! Cross-package comparisons of the rustellar crates with their Python
//! references, on the same data: the numbers the crates' READMEs quote,
//! made again with one command.
//!
//! - [`tls`]: `rustellar-tls` against transitleastsquares 1.32 on injected
//!   transits ([`Kind`] names the sets), rebuilt here from the seeds the
//!   references were run on.
//! - [`limb`]: `rustellar-bls` against `rustellar-tls` on limb-darkened
//!   transits, each at its own 1% false-alarm rate, with
//!   transitleastsquares 1.32 on a subset.
//! - [`window`]: how long a detrending window should be.
//! - [`methods`]: the detrending methods (biweight and splines) on the
//!   window study's curves, with wotan 1.10's.
//! - [`masking`]: detrending again around a search's candidate, its
//!   transits masked, on the same curves.
//! - [`vet`]: `rustellar-vet`'s alias table on transitleastsquares
//!   1.32's candidates, at the true period and at its aliases.
//! - [`speed`]: each crate timed against its Python references.
//!
//! Run `cargo run --release -p rustellar-benchmarks --bin compare -- --help`.

mod curves;
pub mod limb;
pub mod masking;
pub mod methods;
mod random;
// The BLS crate's survey recipe (its items are `pub(super)`, so the whole
// crate sees them), loaded once.
#[expect(dead_code, reason = "only the survey recipe")]
#[path = "../../rustellar-bls/src/tests/recipe.rs"]
mod recipe;
pub mod speed;
mod stats;
pub mod tls;
pub mod vet;
pub mod window;

#[cfg(test)]
mod tests;

pub use curves::{Found, Kind, Row, light_curve, rows, transit};
pub use stats::{median, sign_test, timing};
