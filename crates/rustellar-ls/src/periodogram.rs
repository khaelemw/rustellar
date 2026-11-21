//! The periodogram, exactly: at each frequency the sinusoid fitted by
//! weighted least squares, from compensated, centred sums.

use std::num::NonZeroUsize;

use rustellar_core::LightCurve;

use crate::parallel::{run_all, threads_for};
use crate::sums::{Sum, phase, split};
use crate::{Frequencies, LsError};

/// The fewest point-frequencies worth a thread of their own: about
/// two milliseconds of work.
pub(crate) const GRAIN: usize = 1 << 18;

/// What the sinusoid is fitted with.
///
/// # Examples
///
/// ```
/// use rustellar_ls::Mean;
///
/// assert_eq!(Mean::default(), Mean::Floating);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Mean {
    /// A constant fitted with the sinusoid at each frequency (the
    /// generalised periodogram of Zechmeister & Kurster 2009; astropy's
    /// `fit_mean=True`). Right when the data do not cover every phase.
    #[default]
    Floating,
    /// The weighted mean subtracted once, then the sinusoid alone (the
    /// classic periodogram of Lomb 1976 and Scargle 1982 with weights;
    /// astropy's `fit_mean=False, center_data=True`).
    Subtracted,
    /// The sinusoid alone on the data as given (astropy's
    /// `fit_mean=False, center_data=False`).
    Zero,
}

/// How the power is scaled; with `chi2_ref` the chi-squared of the
/// reference fit (the mean, or zero with [`Mean::Zero`]) and `chi2` that
/// with the sinusoid, as astropy's normalisations (VanderPlas 2018, section
/// 6).
///
/// # Examples
///
/// ```
/// use rustellar_ls::Normalization;
///
/// assert_eq!(Normalization::default(), Normalization::Standard);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Normalization {
    /// `(chi2_ref - chi2) / chi2_ref`, from 0 to 1.
    #[default]
    Standard,
    /// `(chi2_ref - chi2) / chi2`, from 0 up.
    Model,
    /// `ln(chi2_ref / chi2)`, from 0 up.
    Log,
    /// `(chi2_ref - chi2) / 2`, in the flux's units squared over the
    /// errors' (a power spectral density for unit errors).
    Psd,
}

/// The light curve set up for periodograms: times about the middle of the
/// span, weights, and the flux the sinusoid is fitted to.
///
/// # Examples
///
/// ```
/// use rustellar_ls::{Frequencies, LightCurve, LombScargle, Mean, Normalization};
///
/// // A sinusoid of 0.25 cycles per day, sampled unevenly.
/// let time: Vec<f64> = (0..300).map(|i| 0.1 * f64::from(i) + 0.01 * f64::from(i % 7)).collect();
/// let flux = time.iter().map(|t| 1.0 + 0.01 * (std::f64::consts::TAU * 0.25 * t).sin()).collect();
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 300])?;
///
/// let ls = LombScargle::new(&lc, Mean::Floating)?;
/// let grid = Frequencies::uniform(0.01, 0.01, 100)?;
/// let periodogram = ls.power(&grid, Normalization::Standard);
/// let (frequency, power) = periodogram.best();
/// assert!((frequency - 0.25).abs() < 1e-9);
/// assert!(power > 0.999_999);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct LombScargle {
    /// Times less `centre`, in days.
    pub(crate) time: Vec<f64>,
    /// Each time's halves, for exact products.
    pub(crate) halves: Vec<(f64, f64)>,
    /// The middle of the span, in days.
    pub(crate) centre: f64,
    /// The flux the sinusoid is fitted to: less its weighted mean unless
    /// [`Mean::Zero`].
    pub(crate) flux: Vec<f64>,
    /// `(e_min / e)^2`: the weights, scaled so the largest is 1.
    pub(crate) weight: Vec<f64>,
    /// `1 / e_min^2`, the scale of the weights.
    pub(crate) scale: f64,
    /// The weighted mean taken from the flux (0 with [`Mean::Zero`]).
    pub(crate) level: f64,
    pub(crate) mean: Mean,
    /// The reference chi-squared, in scaled weights.
    pub(crate) chi2_ref: f64,
    pub(crate) threads: NonZeroUsize,
}

impl LombScargle {
    /// Sets up `lc` for periodograms with the given [`Mean`], computed on
    /// all cores.
    ///
    /// # Errors
    ///
    /// [`LsError::NoVariation`] if the reference chi-squared is zero: the
    /// flux is constant ([`Mean::Floating`], [`Mean::Subtracted`]) or zero
    /// ([`Mean::Zero`]), so no power is defined.
    pub fn new(lc: &LightCurve, mean: Mean) -> Result<Self, LsError> {
        let time = lc.time();
        let (first, last) = (time[0].as_days(), time[time.len() - 1].as_days());
        let centre = first + 0.5 * (last - first);
        let e_min = lc.flux_err().iter().copied().fold(f64::INFINITY, f64::min);
        let weight: Vec<f64> = lc
            .flux_err()
            .iter()
            .map(|&e| {
                let r = e_min / e;
                r * r
            })
            .collect();
        let mut flux = lc.flux().to_vec();
        let mut level = 0.0;
        if mean != Mean::Zero {
            let (mut wy, mut w) = (Sum::default(), Sum::default());
            for (&y, &wi) in flux.iter().zip(&weight) {
                wy.add(wi * y);
                w.add(wi);
            }
            // The mean to twice the digits, so the flux less it is exact to
            // one rounding of the difference: a mean rounded once leaves a
            // constant that the sinusoid alone (without a floating mean)
            // fits at low frequency.
            let (hi, lo) = wy.quotient(w);
            level = hi + lo;
            for y in &mut flux {
                *y = (*y - hi) - lo;
            }
        }
        let mut chi2_ref = Sum::default();
        for (&y, &wi) in flux.iter().zip(&weight) {
            chi2_ref.add(wi * y * y);
        }
        let chi2_ref = chi2_ref.value();
        if chi2_ref <= 0.0 {
            return Err(LsError::NoVariation);
        }
        let time: Vec<f64> = time.iter().map(|t| t.as_days() - centre).collect();
        Ok(Self {
            halves: time.iter().map(|&t| split(t)).collect(),
            time,
            centre,
            flux,
            weight,
            scale: 1.0 / (e_min * e_min),
            level,
            mean,
            chi2_ref,
            threads: std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
        })
    }

    /// The same, computed on `threads` threads.
    #[must_use]
    pub fn with_threads(self, threads: NonZeroUsize) -> Self {
        Self { threads, ..self }
    }

    /// The [`Mean`] fitted.
    #[must_use]
    pub const fn mean(&self) -> Mean {
        self.mean
    }

    /// The threads used.
    #[must_use]
    pub const fn threads(&self) -> NonZeroUsize {
        self.threads
    }

    /// The middle of the light curve's span, in days: the time the phases
    /// are measured from.
    #[must_use]
    pub const fn centre(&self) -> f64 {
        self.centre
    }

    /// The reference chi-squared: of the weighted mean, or of zero with
    /// [`Mean::Zero`].
    #[must_use]
    pub fn chi2_ref(&self) -> f64 {
        self.chi2_ref * self.scale
    }

    /// The power at each frequency, exactly: the sinusoid fitted by
    /// weighted least squares at each, from phases reduced exactly in
    /// cycles (so large times lose nothing) and compensated sums centred
    /// on their means (so low frequencies lose nothing). Where the
    /// sinusoid's two terms are the same at the points (as at the Nyquist
    /// frequency of even sampling), the one that is not is fitted.
    #[must_use]
    pub fn power(&self, frequencies: &Frequencies, normalization: Normalization) -> Periodogram {
        let n = frequencies.len();
        let mut power = vec![0.0; n];
        // A thread to each GRAIN point-frequencies at least.
        let threads = threads_for(self.threads.get(), n * self.time.len(), GRAIN);
        let chunk = n.div_ceil(threads);
        // Only these divide by the chi-squared left.
        let residual = matches!(normalization, Normalization::Model | Normalization::Log);
        let wide = has_avx2();
        run_all(power.chunks_mut(chunk).enumerate().map(|(k, out)| {
            move || {
                let mut buffer = Buffer::new(self.time.len());
                for (j, p) in out.iter_mut().enumerate() {
                    let fit = self.fit(frequencies.get(k * chunk + j), residual, wide, &mut buffer);
                    *p = self.normalized(fit, normalization);
                }
            }
        }));
        Periodogram {
            frequencies: frequencies.clone(),
            power,
            normalization,
        }
    }

    /// The fit at `f`: the reduction in chi-squared and, if `residual`, the
    /// chi-squared left (else 0), in scaled weights; with AVX2's vectors
    /// where `wide` (the same arithmetic, so the same bits, either way).
    pub(crate) fn fit(&self, f: f64, residual: bool, wide: bool, buffer: &mut Buffer) -> Fit {
        #[cfg(target_arch = "x86_64")]
        if wide {
            // SAFETY: `wide` only where the processor has AVX2.
            return unsafe { self.fit_avx2(f, residual, buffer) };
        }
        let _ = wide;
        self.fit_any(f, residual, buffer)
    }

    /// [`fit`](Self::fit) compiled for AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fit_avx2(&self, f: f64, residual: bool, buffer: &mut Buffer) -> Fit {
        self.fit_any(f, residual, buffer)
    }

    /// [`fit`](Self::fit) for any processor.
    #[inline(always)]
    fn fit_any(&self, f: f64, residual: bool, buffer: &mut Buffer) -> Fit {
        self.centred(f, buffer);
        let [cc, cs, ss, yc, ys] = self.sums(buffer);
        #[expect(clippy::cast_precision_loss, reason = "a point count")]
        let (beta, gamma) = solve(cc, cs, ss, yc, ys, self.time.len() as f64);
        let reduction = (beta * yc + gamma * ys).max(0.0);
        let mut chi2 = Sum::default();
        if residual {
            for (((c, s), &wi), &y) in buffer
                .c
                .iter()
                .zip(&buffer.s)
                .zip(&self.weight)
                .zip(&self.flux)
            {
                let r = y - (beta * c.0 + gamma * s.0);
                chi2.add(wi * r * r);
            }
        }
        Fit {
            reduction,
            chi2: chi2.value(),
        }
    }

    /// Each point's two terms at `f` into `buffer`: base and part in the
    /// pair, then the term, less its weighted mean with a
    /// floating mean, in the first. Returns the weights' sum and the terms'
    /// weighted means `(cos, sin)` (0 without a floating mean).
    #[inline(always)]
    pub(crate) fn centred(&self, f: f64, buffer: &mut Buffer) -> (f64, f64, f64) {
        // Each term as a whole base and a small part (exact near 0 and 1).
        // The means need no compensation: one off by a few roundings only
        // adds its square to the centred sums.
        let f_halves = split(f);
        for (((c, s), &t), &th) in buffer
            .c
            .iter_mut()
            .zip(buffer.s.iter_mut())
            .zip(&self.time)
            .zip(&self.halves)
        {
            (*c, *s) = phase(f, f_halves, t, th);
        }
        let mut means = [0.0; 5];
        for ((c, s), &wi) in buffer.c.iter().zip(&buffer.s).zip(&self.weight) {
            means[0] += wi;
            means[1] += wi * c.0;
            means[2] += wi * c.1;
            means[3] += wi * s.0;
            means[4] += wi * s.1;
        }
        // With a floating mean the terms less their weighted means, base
        // from base and part from part (so terms the same at every point
        // cancel exactly); without one the terms themselves.
        let [w, cb, cx, sb, sx] = means;
        let (cb, cx, sb, sx) = if self.mean == Mean::Floating {
            (cb / w, cx / w, sb / w, sx / w)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        for i in 0..self.time.len() {
            let (b, x) = buffer.c[i];
            buffer.c[i].0 = (b - cb) + (x - cx);
            let (b, x) = buffer.s[i];
            buffer.s[i].0 = (b - sb) + (x - sx);
        }
        (w, cb + cx, sb + sx)
    }

    /// The normal equations' sums from the centred terms in `buffer`:
    /// `[cc, cs, ss, yc, ys]`, weighted.
    #[inline(always)]
    pub(crate) fn sums(&self, buffer: &Buffer) -> [f64; 5] {
        let mut sums = [Sum::default(); 5];
        for i in 0..self.time.len() {
            let (c, s) = (buffer.c[i].0, buffer.s[i].0);
            let (wi, y) = (self.weight[i], self.flux[i]);
            let (wc, ws) = (wi * c, wi * s);
            sums[0].add(wc * c);
            sums[1].add(wc * s);
            sums[2].add(ws * s);
            sums[3].add(wc * y);
            sums[4].add(ws * y);
        }
        sums.map(Sum::value)
    }

    /// `fit`'s power in `normalization`.
    pub(crate) fn normalized(&self, fit: Fit, normalization: Normalization) -> f64 {
        match normalization {
            Normalization::Standard => fit.reduction / self.chi2_ref,
            Normalization::Model => fit.reduction / fit.chi2,
            Normalization::Log => (fit.reduction / fit.chi2).ln_1p(),
            Normalization::Psd => 0.5 * fit.reduction * self.scale,
        }
    }
}

/// The reduction in chi-squared at one frequency and the chi-squared left:
/// each the more accurate where the other is the difference of two close
/// numbers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fit {
    pub(crate) reduction: f64,
    pub(crate) chi2: f64,
}

/// Whether the processor has AVX2.
pub(crate) fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    return std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Each point's two terms at one frequency: base and part, then the term
/// (centred with a floating mean) in the base.
pub(crate) struct Buffer {
    pub(crate) c: Vec<(f64, f64)>,
    pub(crate) s: Vec<(f64, f64)>,
}

impl Buffer {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            c: vec![(0.0, 0.0); n],
            s: vec![(0.0, 0.0); n],
        }
    }
}

/// The least-squares coefficients for the normal equations
/// `[[a, b], [b, d]] x = [u, v]` of `n` points' sums: the full solution,
/// or along one direction where the two terms are the same at the points
/// to rounding (their determinant lost in the sums' rounding), or none.
///
/// The test is on the terms' correlation, not their sizes: a term far
/// smaller than the other but known to its own digits is fitted, as exact
/// least squares does.
pub(crate) fn solve(a: f64, b: f64, d: f64, u: f64, v: f64, n: f64) -> (f64, f64) {
    let trace = a + d;
    let det = a.mul_add(d, -(b * b));
    if full_rank(a, d, det, n) {
        ((d * u - b * v) / det, (a * v - b * u) / det)
    } else if trace > 0.0 {
        // Rank one: the matrix is about e e^T trace, e along its larger
        // column.
        let (x, y) = if a >= d { (a, b) } else { (b, d) };
        let norm = x.hypot(y);
        let (x, y) = (x / norm, y / norm);
        let along = x.mul_add(u, y * v) / trace;
        (along * x, along * y)
    } else {
        (0.0, 0.0)
    }
}

/// Whether the normal equations of `n` points with diagonal `a`, `d` and
/// determinant `det` are of full rank above their rounding.
pub(crate) fn full_rank(a: f64, d: f64, det: f64, n: f64) -> bool {
    det > 8.0 * n * f64::EPSILON * (a * d)
}

/// A periodogram: the frequencies and the power at each.
///
/// # Examples
///
/// ```
/// use rustellar_ls::{Frequencies, LightCurve, LombScargle, Mean, Normalization};
///
/// let time: Vec<f64> = (0..50).map(|i| 0.37 * f64::from(i)).collect();
/// let flux = time.iter().map(|t| (std::f64::consts::TAU * t / 3.0).cos()).collect();
/// let lc = LightCurve::from_days(time, flux, vec![0.1; 50])?;
/// let ls = LombScargle::new(&lc, Mean::Floating)?;
/// let p = ls.power(&Frequencies::from_per_day(&[0.2, 1.0 / 3.0])?, Normalization::Model);
/// assert_eq!(p.power().len(), 2);
/// assert_eq!(p.best().0, 1.0 / 3.0);
/// assert_eq!(p.normalization(), Normalization::Model);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Periodogram {
    pub(crate) frequencies: Frequencies,
    pub(crate) power: Vec<f64>,
    pub(crate) normalization: Normalization,
}

impl Periodogram {
    /// The frequencies, in cycles per day.
    #[must_use]
    pub const fn frequencies(&self) -> &Frequencies {
        &self.frequencies
    }

    /// The power at each frequency.
    #[must_use]
    pub fn power(&self) -> &[f64] {
        &self.power
    }

    /// The normalisation of the power.
    #[must_use]
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }

    /// The frequency with the most power and its power (the first of a
    /// tie), on the grid.
    #[must_use]
    pub fn best(&self) -> (f64, f64) {
        let mut best = 0;
        for (k, &p) in self.power.iter().enumerate() {
            if p > self.power[best] {
                best = k;
            }
        }
        (self.frequencies.get(best), self.power[best])
    }
}
