//! A sum of doubles held exactly, to which values can be added and from
//! which they can be taken away, and its mean correctly rounded.
//!
//! Every double is a whole number of 2^-1074 (the smallest subnormal) below
//! 2^2098, so a sum of up to 2^63 of them is a whole number of 2^-1074 below
//! 2^2161: a two's-complement integer of 34 64-bit limbs holds it, with no
//! rounding and no overflow.

/// Limbs of the sum, least significant first: 2,176 bits.
const LIMBS: usize = 34;

/// An exact sum of doubles, in units of 2^-1074.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactSum {
    limbs: [u64; LIMBS],
}

impl Default for ExactSum {
    fn default() -> Self {
        Self { limbs: [0; LIMBS] }
    }
}

impl ExactSum {
    /// The sum back to zero.
    pub(crate) fn clear(&mut self) {
        self.limbs = [0; LIMBS];
    }

    /// `x` (finite) added.
    pub(crate) fn add(&mut self, x: f64) {
        self.add_signed(x, false);
    }

    /// `x` (finite) taken away.
    pub(crate) fn remove(&mut self, x: f64) {
        self.add_signed(x, true);
    }

    fn add_signed(&mut self, x: f64, negate: bool) {
        debug_assert!(x.is_finite());
        let bits = x.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1 << 52) - 1);
        // x = mantissa 2^(offset - 1074).
        let (mantissa, offset) = if biased == 0 {
            (fraction, 0)
        } else {
            (fraction | (1 << 52), biased - 1)
        };
        if mantissa == 0 {
            return;
        }
        let wide = u128::from(mantissa) << (offset % 64);
        let (low, high) = (wide as u64, (wide >> 64) as u64);
        let at = offset / 64;
        if (bits >> 63 == 1) != negate {
            self.subtract_at(at, low, high);
        } else {
            self.add_at(at, low, high);
        }
    }

    fn add_at(&mut self, at: usize, low: u64, high: u64) {
        let (sum, carry) = self.limbs[at].overflowing_add(low);
        self.limbs[at] = sum;
        let (sum, c1) = self.limbs[at + 1].overflowing_add(high);
        let (sum, c2) = sum.overflowing_add(u64::from(carry));
        self.limbs[at + 1] = sum;
        let mut carry = c1 || c2;
        let mut k = at + 2;
        while carry && k < LIMBS {
            let (sum, c) = self.limbs[k].overflowing_add(1);
            self.limbs[k] = sum;
            carry = c;
            k += 1;
        }
    }

    fn subtract_at(&mut self, at: usize, low: u64, high: u64) {
        let (difference, borrow) = self.limbs[at].overflowing_sub(low);
        self.limbs[at] = difference;
        let (difference, b1) = self.limbs[at + 1].overflowing_sub(high);
        let (difference, b2) = difference.overflowing_sub(u64::from(borrow));
        self.limbs[at + 1] = difference;
        let mut borrow = b1 || b2;
        let mut k = at + 2;
        while borrow && k < LIMBS {
            let (difference, b) = self.limbs[k].overflowing_sub(1);
            self.limbs[k] = difference;
            borrow = b;
            k += 1;
        }
    }

    /// The sum over `count` (at least 1), correctly rounded to the nearest
    /// double, ties to even.
    pub(crate) fn mean(&self, count: u64) -> f64 {
        debug_assert!(count > 0);
        let negative = self.limbs[LIMBS - 1] >> 63 == 1;
        let mut magnitude = self.limbs;
        if negative {
            // Two's complement: invert and add one.
            let mut carry = true;
            for limb in &mut magnitude {
                let (value, c) = (!*limb).overflowing_add(u64::from(carry));
                *limb = value;
                carry = c;
            }
        }
        // Quotient and remainder by `count`, from the top limb down.
        let mut remainder: u128 = 0;
        for limb in magnitude.iter_mut().rev() {
            let wide = (remainder << 64) | u128::from(*limb);
            *limb = (wide / u128::from(count)) as u64;
            remainder = wide % u128::from(count);
        }
        let quotient = magnitude;
        let (remainder, count) = (remainder as u64, count);
        // Twice the remainder against the count: below, at or above half.
        let half = (u128::from(remainder) * 2).cmp(&u128::from(count));
        let length = bit_length(&quotient);
        let value = if length <= 53 {
            // A whole number of 2^-1074 within 53 bits: representable as it
            // is, rounded by the remainder alone.
            let mut q = quotient[0];
            if half.is_gt() || (half.is_eq() && q & 1 == 1) {
                q += 1;
            }
            q as f64 * f64::from_bits(1)
        } else {
            // 53 bits from the top, rounded by the bits below and the
            // remainder.
            let shift = length - 53;
            let mut m = bits_at(&quotient, shift, 53);
            let half_bit = bits_at(&quotient, shift - 1, 1) == 1;
            let rest = remainder != 0 || any_below(&quotient, shift - 1);
            if half_bit && (rest || m & 1 == 1) {
                m += 1;
            }
            m as f64 * power_of_two(shift as i32 - 1074)
        };
        if negative { -value } else { value }
    }
}

/// The position of the highest set bit plus one; 0 for zero.
fn bit_length(limbs: &[u64; LIMBS]) -> usize {
    (0..LIMBS)
        .rev()
        .find(|&k| limbs[k] != 0)
        .map_or(0, |k| 64 * k + 64 - limbs[k].leading_zeros() as usize)
}

/// `count` (at most 64) bits from bit `from` up; `from` below the top limb
/// (a quotient is below 2^2161, so its 53 leading bits start below bit 2108).
fn bits_at(limbs: &[u64; LIMBS], from: usize, count: usize) -> u64 {
    let (at, shift) = (from / 64, from % 64);
    let wide = (u128::from(limbs[at]) | (u128::from(limbs[at + 1]) << 64)) >> shift;
    (wide as u64) & (u64::MAX >> (64 - count))
}

/// Whether any bit below bit `below` is set.
fn any_below(limbs: &[u64; LIMBS], below: usize) -> bool {
    let (at, shift) = (below / 64, below % 64);
    limbs[..at].iter().any(|&l| l != 0) || limbs[at] & ((1 << shift) - 1) != 0
}

/// 2^`power`, for `power` from -1074 to 1023.
fn power_of_two(power: i32) -> f64 {
    debug_assert!((-1074..=1023).contains(&power));
    if power >= -1022 {
        f64::from_bits(((power + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (power + 1074))
    }
}
