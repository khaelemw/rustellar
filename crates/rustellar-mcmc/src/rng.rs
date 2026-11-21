//! The sampler's random numbers: xoshiro256** (Blackman & Vigna 2021),
//! seeded through SplitMix64 as its authors recommend, so a seed gives the
//! same numbers on every platform.

/// SplitMix64 (Steele, Lea & Flood 2014): one step from `state`.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The xoshiro256** generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// The generator whose state is four steps of SplitMix64 from `seed`
    /// (never all zero, as SplitMix64 is a bijection of distinct inputs).
    pub(crate) fn new(seed: u64) -> Self {
        let mut state = seed;
        Self {
            s: std::array::from_fn(|_| splitmix64(&mut state)),
        }
    }

    /// The generator with this raw state, for checking against the
    /// reference implementation.
    #[cfg(test)]
    pub(crate) const fn from_state(s: [u64; 4]) -> Self {
        Self { s }
    }

    /// The next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform on `[0, 1)`, in steps of `2^-53`.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * f64::EPSILON / 2.0
    }

    /// Uniform on `(0, 1)`, never 0 or 1: the midpoints of the steps of
    /// `2^-52`, each exact.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn open_unit(&mut self) -> f64 {
        Self::midpoint(self.next_u64())
    }

    /// The open unit's value for 64 random bits.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn midpoint(bits: u64) -> f64 {
        ((bits >> 12) as f64 + 0.5) * f64::EPSILON
    }

    /// A standard normal, by Marsaglia's polar method (one of each pair).
    pub(crate) fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0f64.mul_add(self.unit(), -1.0);
            let v = 2.0f64.mul_add(self.unit(), -1.0);
            let s = u.mul_add(u, v * v);
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }
}
