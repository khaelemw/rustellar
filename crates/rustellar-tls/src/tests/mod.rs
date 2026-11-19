// Unit tests, one file per module.

use crate::fit::{Points, Position};
use crate::template::Template;

mod cells;
#[cfg(feature = "cuda")]
mod cuda;
mod durations;
mod errors;
mod exact;
mod fast;
mod fit;
mod noise;
mod refine;
mod search;
mod shape;
mod spectra;
mod template;

/// A small, fast pseudo-random generator (SplitMix64), so the tests need no
/// dependency and give the same cases every run.
pub(super) struct Random(u64);

impl Random {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub(super) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Uniform in `[low, high)`.
    pub(super) fn between(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }
}

/// Points every `cadence` days from `start` for `span` days, with a
/// transit of `depth` at `position` and Gaussian-ish noise of `noise`.
pub(super) fn light_curve(
    random: &mut Random,
    template: &Template,
    position: Position,
    depth: f64,
    noise: f64,
    (start, span, cadence): (f64, f64, f64),
) -> Points {
    let mut time = Vec::new();
    let mut t = start;
    while t < start + span {
        time.push(t);
        t += cadence;
    }
    let residual = time
        .iter()
        .map(|&t| {
            // The sum of four uniforms, scaled to unit variance.
            let n: f64 = (0..4).map(|_| random.uniform() - 0.5).sum::<f64>() * 3_f64.sqrt();
            depth * template.shape(position.x(t)) - noise * n
        })
        .collect();
    let weight = vec![1.0 / (noise * noise).max(1e-12); time.len()];
    Points {
        time,
        residual,
        weight,
    }
}
