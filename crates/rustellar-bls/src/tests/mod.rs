// Unit tests for crate-only items, one file per source module, and the
// light-curve recipe they share with the reference generator.
mod binned;
mod box_fit;
#[cfg(feature = "cuda")]
mod cuda;
mod exact;
mod grid;
mod recipe;
mod refine;
mod sums;
mod survey;
mod top;
mod totals;
