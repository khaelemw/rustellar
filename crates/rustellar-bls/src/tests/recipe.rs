// The shared light-curve recipe: the reference generator builds the same
// light curves from the same seeds, bit for bit, so only astropy's results
// are stored. Only + - * / and floor are used, which IEEE 754 rounds the same
// on every platform.

use rustellar_core::LightCurve;

// SplitMix64.
pub(super) struct Random(pub(super) u64);

impl Random {
    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    // A multiple of 2^-53 in [0, 1), exactly.
    pub(super) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

pub(super) struct Recipe {
    pub(super) seed: u64,
    pub(super) n: u32,
    pub(super) t_start: f64,
    pub(super) cadence: f64,
    pub(super) noise: f64,
    pub(super) period: f64,
    pub(super) tc: f64,
    pub(super) duration: f64,
    pub(super) depth: f64,
}

impl Recipe {
    // Times every `cadence`, jittered by up to half a cadence (the first
    // exactly `t_start`); uniform noise of half-width `noise`; a box of
    // `depth` within duration / 2 of tc + k period; errors from 0.75 to 1.25
    // noise. Written as the generator writes it, operation for operation.
    pub(super) fn light_curve(&self) -> LightCurve {
        let mut rng = Random(self.seed);
        let (mut t, mut y, mut e) = (vec![], vec![], vec![]);
        for i in 0..self.n {
            let (u_t, u_y, u_e) = (rng.unit(), rng.unit(), rng.unit());
            let ti = if i == 0 {
                self.t_start
            } else {
                self.t_start + self.cadence * f64::from(i) + 0.5 * self.cadence * u_t
            };
            let mut yi = self.noise * (2.0 * u_y - 1.0);
            if self.depth > 0.0 {
                let r = (ti - self.tc) / self.period;
                let f = r - r.floor();
                if f.min(1.0 - f) * self.period < 0.5 * self.duration {
                    yi -= self.depth;
                }
            }
            t.push(ti);
            y.push(yi);
            e.push(self.noise * (0.75 + 0.5 * u_e));
        }
        LightCurve::from_btjd(t, y, e).unwrap()
    }
}

// The generator's samples, "index:time:flux:flux_err;...", must be the
// recipe's values to the last bit.
pub(super) fn check_samples(lc: &LightCurve, samples: &str) {
    for sample in samples.split(';') {
        let v: Vec<&str> = sample.split(':').collect();
        let i: usize = v[0].parse().unwrap();
        let expected: [f64; 3] = [
            v[1].parse().unwrap(),
            v[2].parse().unwrap(),
            v[3].parse().unwrap(),
        ];
        let ours = [lc.time()[i].as_btjd(), lc.flux()[i], lc.flux_err()[i]];
        for (a, b) in ours.iter().zip(expected) {
            assert_eq!(a.to_bits(), b.to_bits(), "point {i}: {a} vs {b}");
        }
    }
}
