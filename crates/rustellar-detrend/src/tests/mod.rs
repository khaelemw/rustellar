// Unit tests for crate-only items, one file per source module.
mod biweight;
mod estimators;
mod exact_sum;
mod huber_joint;
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

// Windows that test a solver's corners, with a tuning constant each: noise
// about 1, small integers (ties, MAD 0), mixed magnitudes, subnormals of
// either sign, a tight cluster (subnormal or near it) beside distant values
// (c MAD far below the largest magnitude), longer survey-like windows
// (normalised flux with flares and a dip, and counts), and noise about
// 1e250 and about 1e-300 (scaled down and up). The constant from 1e-30 to
// 1e30.
pub(super) fn awkward_windows(random: &mut Random, count: usize) -> Vec<(Vec<f64>, f64)> {
    (0..count)
        .map(|_| {
            let kind = random.below(9);
            let n = 1 + random.below(if kind == 5 || kind == 6 { 400 } else { 80 }) as usize;
            let values = (0..n)
                .map(|_| match kind {
                    0 => 1.0 + 1e-3 * (random.unit() - 0.5),
                    1 => random.below(5) as f64,
                    2 => (random.unit() - 0.5) * 10f64.powi(random.below(8) as i32 - 4),
                    3 => {
                        f64::from_bits(1 + random.below(1 << 20))
                            * if random.below(2) == 0 { 1.0 } else { -1.0 }
                    }
                    4 => {
                        if random.below(3) == 0 {
                            10f64.powi(random.below(600) as i32 - 300)
                        } else {
                            let bits = 1 << (1 + random.below(60));
                            f64::from_bits(random.below(bits))
                        }
                    }
                    5 => {
                        let noise = 1.0 + 1.5e-3 * (random.unit() - 0.5);
                        match random.below(20) {
                            0 => noise + 0.05 * random.unit(),
                            1 => noise - 1e-3,
                            _ => noise,
                        }
                    }
                    6 => (2.3e4 * (1.0 + 1e-3 * (random.unit() - 0.5))).round(),
                    7 => 1e250 * (1.0 + 1e-3 * (random.unit() - 0.5)),
                    _ => 1e-300 * (1.0 + 1e-3 * (random.unit() - 0.5)),
                })
                .collect();
            let c = match random.below(4) {
                0 => 1e-30,
                1 => 1e30,
                _ => 1.0 + 8.0 * random.unit(),
            };
            (values, c)
        })
        .collect()
}
