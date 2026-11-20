// Unit tests for crate-only items, one file per source module, and the
// light-curve recipe they share with the reference data.
mod binned;
mod box_fit;
#[cfg(feature = "cuda")]
mod cuda;
mod exact;
mod grid;
#[expect(
    dead_code,
    reason = "the integration tests check their light curves against the reference data's"
)]
mod recipe;
mod refine;
mod sums;
mod survey;
mod top;
mod totals;
