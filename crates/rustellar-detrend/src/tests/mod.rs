// Unit tests for crate-only items, one file per source module.
mod biweight;
mod estimators;
mod exact_sum;
mod location;
mod trend;
mod window;

// A small deterministic generator (xorshift64*), so the tests need no crate.
pub(super) struct Random(u64);

impl Random {
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    // Uniform in [0, 1).
    pub(super) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}
