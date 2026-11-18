//! The M-estimators' psi functions, in the forms the solver needs, and the
//! exponential, sine and cosine they use: series written out here, so the
//! same bits on every platform (the standard library's come from the
//! platform's maths library).
//!
//! An M-estimator of location `T` solves `sum psi(u_i) = 0`, `u_i = (x_i -
//! T) / s` with `s = c MAD`. With the weight `w(u) = psi(u) / u` that is
//! `sum w(u_i) d_i = 0`, `d_i = x_i - T`: the residual the solver finds the
//! sign change of.
//!
//! The plain forms take [`LANES`] values at once and work stage by stage
//! across them, so that they run as vectors; each lane's result is the same
//! as it would be alone.

use std::f64::consts::PI;

use crate::double::{Double, add, multiply, negate, scale, split, two_product};

/// Values side by side, as many as one vector of the widest instructions
/// the solver is built for holds.
pub(crate) const LANES: usize = 4;

/// [`LANES`] values.
pub(crate) type Lanes = [f64; LANES];

/// The lanes `f(0)`, `f(1)`, ...
#[inline(always)]
fn each(f: impl FnMut(usize) -> f64) -> Lanes {
    std::array::from_fn(f)
}

/// How the plain series run: lane by lane ([`Plain`]) or as one vector
/// ([`Avx2`]). Each is the same sequence of correctly rounded additions and
/// multiplications, so the same bits either way.
pub(crate) trait Isa: Copy {
    /// `sum c_j z^j`, `j` from 0 to 7, by Estrin's scheme.
    fn estrin8(self, c: [f64; 8], z: Lanes) -> Lanes;
    /// `sum c_j z^j`, `j` from 0 to 15, by Estrin's scheme.
    fn estrin(self, c: [f64; 16], z: Lanes) -> Lanes;
    /// `e^x` for `x <= 0` with `series(r)` for `e^r`, `|r| <= ln 2 / 2`.
    fn exp_by(self, x: Lanes, series: impl Fn(Lanes) -> Lanes) -> Lanes;
    /// The double-double sum, as [`add`], in each lane.
    fn add_double(self, a: Doubles, b: Doubles) -> Doubles;
    /// The double-double product, as [`multiply`], in each lane.
    fn multiply_double(self, a: Doubles, b: Doubles) -> Doubles;
    /// `sum c_j z^j` by Horner's scheme: the `tail` (the highest powers)
    /// plainly on `z`'s high part, then the `head` in double-double, as
    /// [`exp_negative_double`] and [`sinc_double`] do; for `G` groups of
    /// lanes, step by step together (each step waits on the last, so one
    /// group alone would leave the processor waiting).
    fn series_double<const G: usize>(
        self,
        tail: &[Double],
        head: &[Double],
        z: [Doubles; G],
    ) -> [Doubles; G];
}

/// Lane by lane, with whatever vectors the compiler makes of it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Plain;

impl Isa for Plain {
    #[inline(always)]
    fn estrin8(self, c: [f64; 8], z: Lanes) -> Lanes {
        estrin8(c, z)
    }

    #[inline(always)]
    fn estrin(self, c: [f64; 16], z: Lanes) -> Lanes {
        estrin(c, z)
    }

    #[inline(always)]
    fn exp_by(self, x: Lanes, series: impl Fn(Lanes) -> Lanes) -> Lanes {
        exp_by(x, series)
    }

    #[inline(always)]
    fn add_double(self, a: Doubles, b: Doubles) -> Doubles {
        add_lanes(a, b)
    }

    #[inline(always)]
    fn multiply_double(self, a: Doubles, b: Doubles) -> Doubles {
        multiply_lanes(a, b)
    }

    #[inline(always)]
    fn series_double<const G: usize>(
        self,
        tail: &[Double],
        head: &[Double],
        z: [Doubles; G],
    ) -> [Doubles; G] {
        let mut plain = [[0.0; LANES]; G];
        for c in tail {
            for g in 0..G {
                plain[g] = each(|i| plain[g][i] * z[g].0[i] + c.0);
            }
        }
        let mut p: [Doubles; G] = std::array::from_fn(|g| (plain[g], [0.0; LANES]));
        for &c in head {
            for g in 0..G {
                p[g] = add_lanes(multiply_lanes(p[g], z[g]), splat(c));
            }
        }
        p
    }
}

/// AVX2's vectors of four doubles: made only where the processor has AVX2
/// (by [`Avx2::new`]), so that its use is safe.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Avx2(());

#[cfg(target_arch = "x86_64")]
impl Avx2 {
    /// AVX2's vectors, if the processor has them.
    pub(crate) fn new() -> Option<Self> {
        std::arch::is_x86_feature_detected!("avx2").then_some(Self(()))
    }
}

#[cfg(target_arch = "x86_64")]
impl Isa for Avx2 {
    #[inline(always)]
    fn estrin8(self, c: [f64; 8], z: Lanes) -> Lanes {
        // SAFETY: an `Avx2` exists only where the processor has AVX2.
        unsafe { wide::store(wide::estrin8(c, wide::load(z))) }
    }

    #[inline(always)]
    fn estrin(self, c: [f64; 16], z: Lanes) -> Lanes {
        // SAFETY: as above.
        unsafe { wide::store(wide::estrin(c, wide::load(z))) }
    }

    #[inline(always)]
    fn exp_by(self, x: Lanes, series: impl Fn(Lanes) -> Lanes) -> Lanes {
        // SAFETY: as above.
        unsafe {
            let (r, half, rest) = wide::reduce(wide::load(x));
            let p = wide::load(series(wide::store(r)));
            wide::store(wide::scale(p, half, rest))
        }
    }

    #[inline(always)]
    fn add_double(self, a: Doubles, b: Doubles) -> Doubles {
        // SAFETY: as above.
        unsafe { wide::store_double(wide::add_double(wide::load_double(a), wide::load_double(b))) }
    }

    #[inline(always)]
    fn multiply_double(self, a: Doubles, b: Doubles) -> Doubles {
        // SAFETY: as above.
        unsafe {
            wide::store_double(wide::multiply_double(
                wide::load_double(a),
                wide::load_double(b),
            ))
        }
    }

    #[inline(always)]
    fn series_double<const G: usize>(
        self,
        tail: &[Double],
        head: &[Double],
        z: [Doubles; G],
    ) -> [Doubles; G] {
        // SAFETY: as above.
        unsafe { wide::series_double(tail, head, z) }
    }
}

/// The plain series on AVX2's vectors: step for step as [`estrin8`],
/// [`estrin`] and [`exp_by`] lane by lane. Built with AVX2 and reached only
/// through an [`Avx2`]; unsafe for that (the processor must have AVX2).
/// (Calls between them need `unsafe` before Rust 1.86 only.)
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
mod wide {
    use std::arch::x86_64::{
        __m256d, _mm256_add_epi64, _mm256_add_pd, _mm256_castpd_si256, _mm256_castsi256_pd,
        _mm256_loadu_pd, _mm256_max_pd, _mm256_mul_pd, _mm256_set1_epi64x, _mm256_set1_pd,
        _mm256_slli_epi64, _mm256_storeu_pd, _mm256_sub_epi64, _mm256_sub_pd,
    };

    use super::{Double, Doubles, LN2_HI, LN2_LO, Lanes, SHIFT};

    /// A double-double in each lane: the high parts, then the low.
    type Wide = (__m256d, __m256d);

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn load_double(a: Doubles) -> Wide {
        unsafe { (load(a.0), load(a.1)) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn store_double(v: Wide) -> Doubles {
        unsafe { (store(v.0), store(v.1)) }
    }

    /// As [`crate::double::two_sum`].
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn two_sum(a: __m256d, b: __m256d) -> Wide {
        unsafe {
            let sum = add(a, b);
            let b_virtual = sub(sum, a);
            let a_virtual = sub(sum, b_virtual);
            (sum, add(sub(a, a_virtual), sub(b, b_virtual)))
        }
    }

    /// As [`crate::double::split`].
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn split(x: __m256d) -> Wide {
        unsafe {
            let t = mul(splat(134_217_729.0), x);
            let hi = sub(t, sub(t, x));
            (hi, sub(x, hi))
        }
    }

    /// As [`crate::double::two_product`].
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn two_product(a: __m256d, b: __m256d) -> Wide {
        unsafe {
            let product = mul(a, b);
            let (a_hi, a_lo) = split(a);
            let (b_hi, b_lo) = split(b);
            let error = add(
                add(
                    add(sub(mul(a_hi, b_hi), product), mul(a_hi, b_lo)),
                    mul(a_lo, b_hi),
                ),
                mul(a_lo, b_lo),
            );
            (product, error)
        }
    }

    /// As [`crate::double::add`].
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn add_double(a: Wide, b: Wide) -> Wide {
        unsafe {
            let (sum, error) = two_sum(a.0, b.0);
            two_sum(sum, add(add(error, a.1), b.1))
        }
    }

    /// As [`crate::double::multiply`].
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn multiply_double(a: Wide, b: Wide) -> Wide {
        unsafe {
            let (product, error) = two_product(a.0, b.0);
            two_sum(product, add(add(error, mul(a.0, b.1)), mul(a.1, b.0)))
        }
    }

    /// As [`super::Plain`]'s `series_double`. (Loops, not closures: a
    /// closure would be built without AVX2.)
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn series_double<const G: usize>(
        tail: &[Double],
        head: &[Double],
        z: [Doubles; G],
    ) -> [Doubles; G] {
        unsafe {
            let mut zs = [(splat(0.0), splat(0.0)); G];
            for g in 0..G {
                zs[g] = load_double(z[g]);
            }
            let mut p = [(splat(0.0), splat(0.0)); G];
            for c in tail {
                for g in 0..G {
                    p[g].0 = add(mul(p[g].0, zs[g].0), splat(c.0));
                }
            }
            for c in head {
                let c = (splat(c.0), splat(c.1));
                for g in 0..G {
                    p[g] = add_double(multiply_double(p[g], zs[g]), c);
                }
            }
            let mut out = [([0.0; 4], [0.0; 4]); G];
            for g in 0..G {
                out[g] = store_double(p[g]);
            }
            out
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn load(a: Lanes) -> __m256d {
        // SAFETY: four doubles read from a four-double array.
        unsafe { _mm256_loadu_pd(a.as_ptr()) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn store(v: __m256d) -> Lanes {
        let mut a = [0.0; 4];
        // SAFETY: four doubles written to a four-double array.
        unsafe { _mm256_storeu_pd(a.as_mut_ptr(), v) };
        a
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_add_pd(a, b) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_sub_pd(a, b) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mul(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_mul_pd(a, b) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn splat(x: f64) -> __m256d {
        unsafe { _mm256_set1_pd(x) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn estrin8(c: [f64; 8], z: __m256d) -> __m256d {
        unsafe {
            let z2 = mul(z, z);
            let z4 = mul(z2, z2);
            let a0 = add(splat(c[0]), mul(splat(c[1]), z));
            let a1 = add(splat(c[2]), mul(splat(c[3]), z));
            let a2 = add(splat(c[4]), mul(splat(c[5]), z));
            let a3 = add(splat(c[6]), mul(splat(c[7]), z));
            add(add(a0, mul(a1, z2)), mul(add(a2, mul(a3, z2)), z4))
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn estrin(c: [f64; 16], z: __m256d) -> __m256d {
        unsafe {
            let z2 = mul(z, z);
            let z4 = mul(z2, z2);
            let z8 = mul(z4, z4);
            let low = estrin8([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]], z);
            let high = estrin8([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]], z);
            add(low, mul(high, z8))
        }
    }

    /// `x` (at most 0) as `k ln 2 + r`: `r`, and the two halves of `k`
    /// plus [`SHIFT`].
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn reduce(x: __m256d) -> (__m256d, __m256d, __m256d) {
        unsafe {
            let x = _mm256_max_pd(x, splat(-746.0));
            let k = sub(
                add(mul(x, splat(std::f64::consts::LOG2_E)), splat(SHIFT)),
                splat(SHIFT),
            );
            let r = sub(sub(x, mul(k, splat(LN2_HI))), mul(k, splat(LN2_LO)));
            let half = add(mul(k, splat(0.5)), splat(SHIFT));
            let rest = add(sub(k, sub(half, splat(SHIFT))), splat(SHIFT));
            (r, half, rest)
        }
    }

    /// `p 2^k`, `k`'s halves as [`reduce`] gives them.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn scale(p: __m256d, half: __m256d, rest: __m256d) -> __m256d {
        unsafe { mul(mul(p, power(half)), power(rest)) }
    }

    /// 2^k for `shifted` = k + [`SHIFT`], as [`super::shifted_power`].
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn power(shifted: __m256d) -> __m256d {
        unsafe {
            let k = _mm256_sub_epi64(
                _mm256_castpd_si256(shifted),
                _mm256_set1_epi64x(SHIFT.to_bits() as i64),
            );
            _mm256_castsi256_pd(_mm256_slli_epi64::<52>(_mm256_add_epi64(
                k,
                _mm256_set1_epi64x(1023),
            )))
        }
    }
}

/// A psi function, `u = d / s`.
pub(crate) trait Psi {
    /// Points with `|u|` at or beyond this have no weight (infinite: none).
    /// `psi` is continuous there, so which side a point at the edge counts
    /// on changes the residual by no more than its rounding.
    const REACH: f64;
    /// A bound on a term's rounding error in the plain sums, in units of
    /// `eps |d| m(u)` (`m` from [`weigh`](Self::weigh)).
    const TERM_ERROR: f64;
    /// The weights `w(u)`, roughly (to steer wotan's iteration only).
    fn rough<V: Isa>(v: V, u: Lanes) -> Lanes;
    /// The weights, `m(u)` (the size a term's error is bounded by) and
    /// `psi'(u)` (for Newton's method; unused, it goes uncomputed).
    fn weigh<V: Isa>(v: V, u: Lanes) -> (Lanes, Lanes, Lanes);
    /// The term `w(u) d` in double-double, for `u`, `d` and `s` given so.
    fn term(u: Double, d: Double, s: Double) -> Double;
    /// [`term`](Self::term) for each lane of `G` groups, to the same bits:
    /// independent chains side by side, for the processor to overlap.
    #[inline(always)]
    fn terms<V: Isa, const G: usize>(
        _: V,
        u: [Doubles; G],
        d: [Doubles; G],
        s: Double,
    ) -> [Doubles; G] {
        std::array::from_fn(|g| {
            let (u, d) = (u[g], d[g]);
            let t: [Double; LANES] =
                std::array::from_fn(|l| Self::term((u.0[l], u.1[l]), (d.0[l], d.1[l]), s));
            (each(|l| t[l].0), each(|l| t[l].1))
        })
    }
}

/// Tukey's biweight: `w = (1 - u^2)^2` for `|u| < 1`.
pub(crate) struct Biweight;

impl Psi for Biweight {
    const REACH: f64 = 1.0;
    /// `u` within 4 eps (the reciprocal, the product, `d`'s low part and
    /// c MAD's), `1 - u^2` within 9 eps u^2 + eps (1 - u^2), its square and
    /// the product with `d`: 22 in units of `eps |d| (1 - u^2)`; doubled.
    const TERM_ERROR: f64 = 44.0;

    #[inline(always)]
    fn rough<V: Isa>(isa: V, u: Lanes) -> Lanes {
        Self::weigh(isa, u).0
    }

    #[inline(always)]
    fn weigh<V: Isa>(_: V, u: Lanes) -> (Lanes, Lanes, Lanes) {
        let v = each(|i| 1.0 - u[i] * u[i]);
        (
            each(|i| v[i] * v[i]),
            v,
            each(|i| v[i] * (1.0 - 5.0 * u[i] * u[i])),
        )
    }

    #[inline(always)]
    fn term(u: Double, d: Double, _: Double) -> Double {
        let v = add((1.0, 0.0), negate(multiply(u, u)));
        multiply(multiply(v, v), d)
    }

    #[inline(always)]
    fn terms<V: Isa, const G: usize>(
        isa: V,
        u: [Doubles; G],
        d: [Doubles; G],
        _: Double,
    ) -> [Doubles; G] {
        // Loops, not closures, round the vector steps: a closure would be
        // built without AVX2.
        let mut terms = [ZERO; G];
        for g in 0..G {
            let square = isa.multiply_double(u[g], u[g]);
            let v = isa.add_double(
                splat((1.0, 0.0)),
                (each(|i| -square.0[i]), each(|i| -square.1[i])),
            );
            terms[g] = isa.multiply_double(isa.multiply_double(v, v), d[g]);
        }
        terms
    }
}

/// Welsch's (Dennis and Welsch 1978): `w = exp(-u^2 / 2)`, never zero. (wotan
/// also sets it to zero at `|u| >= 1`, which makes the residual jump there,
/// so that it can change sign with no root.)
pub(crate) struct Welsch;

impl Psi for Welsch {
    const REACH: f64 = f64::INFINITY;
    /// The exponential within 4 eps relative, `-u^2 / 2` within eps (so
    /// `w` within `eps u^2 / 2`), `u` within 4 eps (so `w` within
    /// `4 eps u^2 w`), the product with `d` 1: under 6 in units of
    /// `eps |d| w (1 + u^2)`; doubled and rounded up.
    const TERM_ERROR: f64 = 16.0;

    /// The exponential's series to `r^7 / 7!`: within 6e-9 relative.
    #[inline(always)]
    fn rough<V: Isa>(v: V, u: Lanes) -> Lanes {
        v.exp_by(each(|i| -0.5 * (u[i] * u[i])), |r| {
            v.estrin8(first_eight(&EXP), r)
        })
    }

    #[inline(always)]
    fn weigh<V: Isa>(v: V, u: Lanes) -> (Lanes, Lanes, Lanes) {
        let w = exp_lanes(v, each(|i| -0.5 * (u[i] * u[i])));
        (
            w,
            // A subnormal w is wrong by up to 2^-1074 besides: the least
            // normal double covers that.
            each(|i| w[i] * (1.0 + u[i] * u[i]) + f64::MIN_POSITIVE),
            each(|i| w[i] * (1.0 - u[i] * u[i])),
        )
    }

    #[inline(always)]
    fn term(u: Double, d: Double, _: Double) -> Double {
        let x = multiply(u, u);
        multiply(exp_negative_double((-x.0 / 2.0, -x.1 / 2.0)), d)
    }

    #[inline(always)]
    fn terms<V: Isa, const G: usize>(
        v: V,
        u: [Doubles; G],
        d: [Doubles; G],
        _: Double,
    ) -> [Doubles; G] {
        let mut x = [ZERO; G];
        for g in 0..G {
            let square = v.multiply_double(u[g], u[g]);
            x[g] = (each(|i| -square.0[i] / 2.0), each(|i| -square.1[i] / 2.0));
        }
        let mut terms = exp_negative_lanes(v, x);
        for g in 0..G {
            terms[g] = v.multiply_double(terms[g], d[g]);
        }
        terms
    }
}

/// Andrews' sine wave: `psi = sin u`, `w = sin(u) / u`, for `|u| < pi`.
pub(crate) struct Andrews;

impl Psi for Andrews {
    const REACH: f64 = PI;
    /// The series within 12 eps absolute, `u` within 4 eps (`|w'| u` under
    /// 1.4: 6 eps), the product with `d` 1: 19 in units of `eps |d|`;
    /// doubled and rounded up.
    const TERM_ERROR: f64 = 40.0;

    /// `sinc` roughly: its series to `y^7 / 15!`, within 3e-7 absolute.
    #[inline(always)]
    fn rough<V: Isa>(v: V, u: Lanes) -> Lanes {
        v.estrin8(first_eight(&SINC), each(|i| u[i] * u[i]))
    }

    #[inline(always)]
    fn weigh<V: Isa>(v: V, u: Lanes) -> (Lanes, Lanes, Lanes) {
        let y = each(|i| u[i] * u[i]);
        // The slope only scales Newton's steps: cos to y^7 / 14! (within
        // 5e-6) does.
        (
            v.estrin(SINC, y),
            [1.0; LANES],
            v.estrin8(first_eight(&COS), y),
        )
    }

    #[inline(always)]
    fn term(u: Double, d: Double, _: Double) -> Double {
        multiply(sinc_double(multiply(u, u)), d)
    }

    #[inline(always)]
    fn terms<V: Isa, const G: usize>(
        v: V,
        u: [Doubles; G],
        d: [Doubles; G],
        _: Double,
    ) -> [Doubles; G] {
        let mut y = [ZERO; G];
        for g in 0..G {
            y[g] = v.multiply_double(u[g], u[g]);
        }
        let (tail, head) = SINC_DOUBLE.split_at(SINC_DOUBLE.len() - 14);
        let mut terms = v.series_double(tail, head, y);
        for g in 0..G {
            terms[g] = v.multiply_double(terms[g], d[g]);
        }
        terms
    }
}

/// Huber's: `psi = u` for `|u| < 1`, else its sign; `w = min(1, 1 / |u|)`.
pub(crate) struct Huber;

impl Psi for Huber {
    const REACH: f64 = f64::INFINITY;
    /// Inside, the term is `d` (exact with its low part). Outside, `d / |u|`
    /// (`= s`): `u` within 4 eps, its reciprocal and the product 2: 6 in
    /// units of `eps |d| w`; doubled and rounded up.
    const TERM_ERROR: f64 = 16.0;

    #[inline(always)]
    fn rough<V: Isa>(v: V, u: Lanes) -> Lanes {
        Self::weigh(v, u).0
    }

    #[inline(always)]
    fn weigh<V: Isa>(_: V, u: Lanes) -> (Lanes, Lanes, Lanes) {
        let w = each(|i| {
            if u[i].abs() < 1.0 {
                1.0
            } else {
                1.0 / u[i].abs()
            }
        });
        let slope = each(|i| if u[i].abs() < 1.0 { 1.0 } else { 0.0 });
        (w, w, slope)
    }

    #[inline(always)]
    fn term(u: Double, d: Double, s: Double) -> Double {
        // |u| < 1 exactly, from the double-double.
        let size = if u.0 < 0.0 { negate(u) } else { u };
        if size.0 < 1.0 || (size.0 == 1.0 && size.1 < 0.0) {
            d
        } else if d.0 < 0.0 {
            negate(s)
        } else {
            s
        }
    }
}

/// `ln 2` in two parts: the first with 21 trailing zero bits, so its product
/// with any integer up to 2^11 is exact (Cody and Waite).
const LN2_HI: f64 = 6.931_471_803_691_238e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;

/// `ln 2` as a double-double.
const LN2: Double = (std::f64::consts::LN_2, 2.319_046_813_846_299_6e-17);

/// `1 / j!`, `j` from 0 to 13, correctly rounded, and two zeros (exact
/// in the sums, so the series' sizes and roundings are those of 14 terms).
pub(crate) const EXP: [f64; 16] = [
    1.0,
    1.0,
    0.5,
    0.16666666666666666,
    0.041666666666666664,
    0.008333333333333333,
    0.001388888888888889,
    0.0001984126984126984,
    2.48015873015873e-05,
    2.7557319223985893e-06,
    2.755731922398589e-07,
    2.505210838544172e-08,
    2.08767569878681e-09,
    1.6059043836821613e-10,
    0.0,
    0.0,
];

/// `1 / j!`, `j` from 27 down to 0, as double-doubles.
pub(crate) const EXP_DOUBLE: [Double; 28] = [
    (9.183689863795546e-29, 1.4303150396787322e-45),
    (2.4795962632247976e-27, -1.2953730964765229e-43),
    (6.446950284384474e-26, -1.9330404233703465e-42),
    (1.6117375710961184e-24, -3.6846573564509766e-41),
    (3.868170170630684e-23, -8.843177655482344e-40),
    (8.896791392450574e-22, -7.911402614872376e-38),
    (1.9572941063391263e-20, -1.3643503830087908e-36),
    (4.110317623312165e-19, 1.4412973378659527e-36),
    (8.22063524662433e-18, 2.2141894119604265e-34),
    (1.5619206968586225e-16, 1.1910679660273754e-32),
    (2.8114572543455206e-15, 1.6508842730861433e-31),
    (4.779477332387385e-14, 4.399205485834081e-31),
    (7.647163731819816e-13, 7.03872877733453e-30),
    (1.1470745597729725e-11, 2.0655512752830745e-28),
    (1.6059043836821613e-10, 1.2585294588752098e-26),
    (2.08767569878681e-09, -1.20734505911326e-25),
    (2.505210838544172e-08, -1.448814070935912e-24),
    (2.755731922398589e-07, 2.3767714622250297e-23),
    (2.7557319223985893e-06, -1.858393274046472e-22),
    (2.48015873015873e-05, 2.1511947866775882e-23),
    (0.0001984126984126984, 1.7209558293420705e-22),
    (0.001388888888888889, -5.300543954373577e-20),
    (0.008333333333333333, 1.1564823173178714e-19),
    (0.041666666666666664, 2.3129646346357427e-18),
    (0.16666666666666666, 9.25185853854297e-18),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.0, 0.0),
];

/// `(-1)^j / (2j + 1)!`, `j` from 0 to 15, correctly rounded.
pub(crate) const SINC: [f64; 16] = [
    1.0,
    -0.16666666666666666,
    0.008333333333333333,
    -0.0001984126984126984,
    2.7557319223985893e-06,
    -2.505210838544172e-08,
    1.6059043836821613e-10,
    -7.647163731819816e-13,
    2.8114572543455206e-15,
    -8.22063524662433e-18,
    1.9572941063391263e-20,
    -3.868170170630684e-23,
    6.446950284384474e-26,
    -9.183689863795546e-29,
    1.1309962886447716e-31,
    -1.216125041553518e-34,
];

/// `(-1)^j / (2j)!`, `j` from 0 to 15, correctly rounded.
pub(crate) const COS: [f64; 16] = [
    1.0,
    -0.5,
    0.041666666666666664,
    -0.001388888888888889,
    2.48015873015873e-05,
    -2.755731922398589e-07,
    2.08767569878681e-09,
    -1.1470745597729725e-11,
    4.779477332387385e-14,
    -1.5619206968586225e-16,
    4.110317623312165e-19,
    -8.896791392450574e-22,
    1.6117375710961184e-24,
    -2.4795962632247976e-27,
    3.279889237069838e-30,
    -3.7699876288159054e-33,
];

/// `(-1)^j / (2j + 1)!`, `j` from 24 down to 0, as double-doubles.
pub(crate) const SINC_DOUBLE: [Double; 25] = [
    (1.643974708316579e-63, -4.080880981844294e-80),
    (-3.866628513960594e-60, 1.564355005786389e-76),
    (8.359650847182804e-57, -5.0402798850883064e-73),
    (-1.6552108677421951e-53, -4.147105190494824e-70),
    (2.9893108271424046e-50, -1.0407247703033156e-66),
    (-4.902469756513544e-47, 1.213019100517928e-63),
    (7.265460179153071e-44, -4.364097149354446e-61),
    (-9.67759295863189e-41, -3.202295548645562e-57),
    (1.151633562077195e-37, -6.09957445788454e-54),
    (-1.216125041553518e-34, -5.586290567888806e-51),
    (1.1309962886447716e-31, 1.0498015412959506e-47),
    (-9.183689863795546e-29, -1.4303150396787322e-45),
    (6.446950284384474e-26, -1.9330404233703465e-42),
    (-3.868170170630684e-23, 8.843177655482344e-40),
    (1.9572941063391263e-20, -1.3643503830087908e-36),
    (-8.22063524662433e-18, -2.2141894119604265e-34),
    (2.8114572543455206e-15, 1.6508842730861433e-31),
    (-7.647163731819816e-13, -7.03872877733453e-30),
    (1.6059043836821613e-10, 1.2585294588752098e-26),
    (-2.505210838544172e-08, 1.448814070935912e-24),
    (2.7557319223985893e-06, -1.858393274046472e-22),
    (-0.0001984126984126984, -1.7209558293420705e-22),
    (0.008333333333333333, 1.1564823173178714e-19),
    (-0.16666666666666666, -9.25185853854297e-18),
    (1.0, 0.0),
];

/// `e^x` for `x <= 0`, within 6 eps relative (0 below -746; where `e^x` is
/// subnormal, within 2^-1074 besides).
#[cfg(test)]
pub(crate) fn exp_negative(x: f64) -> f64 {
    exp_lanes(Plain, [x; LANES])[0]
}

/// [`exp_negative`] on lanes.
#[inline(always)]
fn exp_lanes<V: Isa>(v: V, x: Lanes) -> Lanes {
    // The series to r^13 / 13! (the rest under 1e-18 relative), by
    // Estrin's scheme: the coefficients' and the steps' roundings come to
    // under 5.4 eps relative, r's under 0.4.
    v.exp_by(x, |r| v.estrin(EXP, r))
}

/// `e^x` for `x <= 0` with `series(r)` for `e^r`, `|r| <= ln 2 / 2`.
#[inline(always)]
fn exp_by(x: Lanes, series: impl Fn(Lanes) -> Lanes) -> Lanes {
    // e^-746 rounds to 0, so do all below: no branch.
    let x = each(|i| x[i].max(-746.0));
    // x = k ln 2 + r.
    let k = each(|i| (x[i] * std::f64::consts::LOG2_E + SHIFT) - SHIFT);
    let r = each(|i| (x[i] - k[i] * LN2_HI) - k[i] * LN2_LO);
    let p = series(r);
    // 2^k (k from -1077 to 0) in two normal halves: the first product
    // exact, the second rounded only where the result is subnormal.
    let half = each(|i| k[i] * 0.5 + SHIFT);
    let rest = each(|i| (k[i] - (half[i] - SHIFT)) + SHIFT);
    each(|i| p[i] * shifted_power(half[i]) * shifted_power(rest[i]))
}

/// 1.5 2^52: adding and taking it away rounds a double below 2^51 in size
/// to the nearest integer (ties to even), which the sum's low bits hold.
/// No call to the maths library, as `round` is on most x86-64 builds.
const SHIFT: f64 = 6_755_399_441_055_744.0;

/// 2^k for `shifted` = k + [`SHIFT`] (k an integer from -1022 to 1023):
/// the sum's bits are SHIFT's plus k.
#[inline(always)]
fn shifted_power(shifted: f64) -> f64 {
    let k = shifted.to_bits().wrapping_sub(SHIFT.to_bits());
    f64::from_bits(k.wrapping_add(1023) << 52)
}

/// The integer nearest `x` (ties to even), for `|x|` below 2^51.
fn nearest_integer(x: f64) -> f64 {
    (x + SHIFT) - SHIFT
}

/// `e^x` for `x <= 0` in double-double, within a few units of 2^-104
/// relative (0 below -746).
pub(crate) fn exp_negative_double(x: Double) -> Double {
    if x.0 < -746.0 {
        return (0.0, 0.0);
    }
    let k = nearest_integer(x.0 * std::f64::consts::LOG2_E);
    let r = add(add(x, negate(two_product(k, LN2.0))), (-k * LN2.1, 0.0));
    // To r^27 / 27! (the rest under 1e-38): from r^14 on plainly (those
    // terms are under 5e-18, so their roundings under 1e-33), then the
    // first fourteen in double-double.
    let (tail, head) = EXP_DOUBLE.split_at(EXP_DOUBLE.len() - 14);
    let tail = tail.iter().fold(0.0, |p, &c| p * r.0 + c.0);
    let p = head
        .iter()
        .fold((tail, 0.0), |p, &c| add(multiply(p, r), c));
    let k = k as i32;
    (scale(p.0, k), scale(p.1, k))
}

/// `sin(u) / u` for `y = u^2 < pi^2`, by its series to `y^15 / 31!` (the
/// rest under 1e-19), by Estrin's scheme: within 12 eps absolute (the
/// coefficients' and the steps' roundings, bounded with the terms' sizes).
#[cfg(test)]
pub(crate) fn sinc(y: f64) -> f64 {
    estrin(SINC, [y; LANES])[0]
}

/// `cos u` for `y = u^2 < pi^2`, by its series to `y^15 / 30!`: within
/// 40 eps absolute (it only steers).
#[cfg(test)]
pub(crate) fn cos(y: f64) -> f64 {
    estrin(COS, [y; LANES])[0]
}

/// `sum c_j z^j` by Estrin's scheme: neighbours paired as `c_2i + c_2i+1 z`,
/// then those as `a_2i + a_2i+1 z^2`, and so on, `z`'s powers by squaring.
/// A tree of depth 4 rather than Horner's chain of 16 steps.
#[inline(always)]
fn estrin(c: [f64; 16], z: Lanes) -> Lanes {
    let z8 = each(|i| {
        let z2 = z[i] * z[i];
        let z4 = z2 * z2;
        z4 * z4
    });
    let (low, high) = c.split_at(8);
    let low = estrin8(first_eight(low), z);
    let high = estrin8(first_eight(high), z);
    each(|i| low[i] + high[i] * z8[i])
}

/// [`estrin`] for eight coefficients.
#[inline(always)]
fn estrin8(c: [f64; 8], z: Lanes) -> Lanes {
    each(|i| {
        let z = z[i];
        let z2 = z * z;
        let z4 = z2 * z2;
        let a0 = c[0] + c[1] * z;
        let a1 = c[2] + c[3] * z;
        let a2 = c[4] + c[5] * z;
        let a3 = c[6] + c[7] * z;
        (a0 + a1 * z2) + (a2 + a3 * z2) * z4
    })
}

/// The first eight of `c`.
#[inline(always)]
fn first_eight(c: &[f64]) -> [f64; 8] {
    std::array::from_fn(|i| c[i])
}

/// [`sinc`] in double-double, to `y^24 / 49!` (under 1e-38): from `y^14`
/// on plainly (those terms are under 1e-17, so their roundings under
/// 1e-32), then the first fourteen in double-double.
pub(crate) fn sinc_double(y: Double) -> Double {
    let (tail, head) = SINC_DOUBLE.split_at(SINC_DOUBLE.len() - 14);
    let tail = tail.iter().fold(0.0, |p, &c| p * y.0 + c.0);
    head.iter()
        .fold((tail, 0.0), |p, &c| add(multiply(p, y), c))
}

// ==========================
// Double-doubles four at a time
// ==========================

/// A double-double in each lane: the high parts, then the low.
pub(crate) type Doubles = (Lanes, Lanes);

/// Zero in each lane.
const ZERO: Doubles = ([0.0; LANES], [0.0; LANES]);

/// `x` in every lane.
#[inline(always)]
fn splat(x: Double) -> Doubles {
    ([x.0; LANES], [x.1; LANES])
}

/// [`crate::double::two_sum`] in each lane.
#[inline(always)]
pub(crate) fn two_sum_lanes(a: Lanes, b: Lanes) -> Doubles {
    let sum = each(|i| a[i] + b[i]);
    let b_virtual = each(|i| sum[i] - a[i]);
    let a_virtual = each(|i| sum[i] - b_virtual[i]);
    (sum, each(|i| (a[i] - a_virtual[i]) + (b[i] - b_virtual[i])))
}

/// [`crate::double::two_product`] in each lane (Dekker's, no FMA).
#[inline(always)]
pub(crate) fn two_product_lanes(a: Lanes, b: Lanes) -> Doubles {
    let product = each(|i| a[i] * b[i]);
    let a_parts: [Double; LANES] = std::array::from_fn(|i| split(a[i]));
    let b_parts: [Double; LANES] = std::array::from_fn(|i| split(b[i]));
    let error = each(|i| {
        let ((a_hi, a_lo), (b_hi, b_lo)) = (a_parts[i], b_parts[i]);
        ((a_hi * b_hi - product[i]) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    });
    (product, error)
}

/// [`add`] in each lane.
#[inline(always)]
pub(crate) fn add_lanes(a: Doubles, b: Doubles) -> Doubles {
    let (sum, error) = two_sum_lanes(a.0, b.0);
    two_sum_lanes(sum, each(|i| error[i] + a.1[i] + b.1[i]))
}

/// [`multiply`] in each lane.
#[inline(always)]
pub(crate) fn multiply_lanes(a: Doubles, b: Doubles) -> Doubles {
    let (product, error) = two_product_lanes(a.0, b.0);
    two_sum_lanes(
        product,
        each(|i| error[i] + a.0[i] * b.1[i] + a.1[i] * b.0[i]),
    )
}

/// [`exp_negative_double`] in each lane of `G` groups, to the same bits.
#[inline(always)]
pub(crate) fn exp_negative_lanes<V: Isa, const G: usize>(v: V, x: [Doubles; G]) -> [Doubles; G] {
    // Lanes below -746 are 0; worked through as 0 meanwhile.
    let under: [[bool; LANES]; G] =
        std::array::from_fn(|g| std::array::from_fn(|i| x[g].0[i] < -746.0));
    let k: [Lanes; G] = std::array::from_fn(|g| {
        each(|i| {
            let x = if under[g][i] { 0.0 } else { x[g].0[i] };
            (x * std::f64::consts::LOG2_E + SHIFT) - SHIFT
        })
    });
    let mut r = [ZERO; G];
    for g in 0..G {
        let x = (
            each(|i| if under[g][i] { 0.0 } else { x[g].0[i] }),
            each(|i| if under[g][i] { 0.0 } else { x[g].1[i] }),
        );
        let (hi, lo) = two_product_lanes(k[g], [LN2.0; LANES]);
        r[g] = v.add_double(
            v.add_double(x, (each(|i| -hi[i]), each(|i| -lo[i]))),
            (each(|i| -k[g][i] * LN2.1), [0.0; LANES]),
        );
    }
    let (tail, head) = EXP_DOUBLE.split_at(EXP_DOUBLE.len() - 14);
    let p = v.series_double(tail, head, r);
    std::array::from_fn(|g| {
        let part = |part: Lanes| {
            each(|i| {
                if under[g][i] {
                    0.0
                } else {
                    scale(part[i], k[g][i] as i32)
                }
            })
        };
        (part(p[g].0), part(p[g].1))
    })
}
