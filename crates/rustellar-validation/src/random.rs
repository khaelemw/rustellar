//! SplitMix64, as the reference generators draw it.

/// SplitMix64: the noise of every reference light curve, drawn the same way
/// in the generators.
pub(crate) struct Random(u64);

impl Random {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A multiple of 2^-53 in `[0, 1)`, exactly.
    pub(crate) fn uniform(&mut self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "below 2^53, exact")]
        let u = (self.next_u64() >> 11) as f64;
        u / 9_007_199_254_740_992.0
    }

    /// Unit-variance noise: four uniforms less 2, times sqrt(3).
    pub(crate) fn noise(&mut self) -> f64 {
        let mut n = 0.0;
        for _ in 0..4 {
            n += self.uniform() - 0.5;
        }
        n * 3_f64.sqrt()
    }
}
