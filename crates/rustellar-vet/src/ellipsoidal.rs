use std::f64::consts::TAU;

use rustellar_core::LightCurve;

use crate::Candidate;
use crate::fit::phase;
use crate::sum::Sum;

/// A sinusoid's two terms, in fractions of the baseline, at `phase` days
/// from mid-transit: `cos * cos(k w phase) + sin * sin(k w phase)` with `w
/// = 2 pi / period`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Harmonic {
    /// The cosine term: below zero for a minimum at mid-transit.
    pub cos: f64,
    /// Its standard error.
    pub cos_err: f64,
    /// The sine term.
    pub sin: f64,
    /// Its standard error.
    pub sin_err: f64,
}

impl Harmonic {
    /// The amplitude, `sqrt(cos^2 + sin^2)`.
    #[must_use]
    pub fn amplitude(&self) -> f64 {
        self.cos.hypot(self.sin)
    }

    /// How far from no sinusoid at all, in standard errors: `sqrt((cos /
    /// cos_err)^2 + (sin / sin_err)^2)`, from chi-squared with two degrees
    /// of freedom when there is none.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        (self.cos / self.cos_err).hypot(self.sin / self.sin_err)
    }
}

/// Sinusoids at the orbital period and half of it, fitted to the light
/// curve out of both eclipses.
///
/// A close binary's stars are stretched toward each other, so its light
/// varies twice an orbit with minima at both eclipses (ellipsoidal
/// variation: [`twice`](Self::twice) with a negative cosine), often by
/// hundreds of parts per million; a planet's is far smaller. Once an orbit:
/// light reflected by the companion (a negative cosine, brightest at half
/// the period) and Doppler beaming (the sine).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Ellipsoidal {
    /// The sinusoid once an orbit.
    pub once: Harmonic,
    /// The sinusoid twice an orbit.
    pub twice: Harmonic,
    /// The points fitted: those at least a duration from mid-transit and
    /// from half the period.
    pub points: usize,
}

/// The sinusoids once and twice an orbit in `lc`, out of `candidate`'s
/// eclipses, by weighted least squares with a constant.
///
/// `None` with fewer than six points out of eclipse, or phases too few to
/// tell the sinusoids apart.
///
/// # Examples
///
/// ```
/// use std::f64::consts::TAU;
/// use rustellar_vet::{Candidate, Epoch, LightCurve, TimeDelta, ellipsoidal};
///
/// // Light varying by 500 ppm twice an orbit, minima at the eclipses.
/// let time: Vec<f64> = (0..1728).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .map(|&t| 1.0 - 5e-4 * (2.0 * TAU * (t - 0.75) / 3.0).cos())
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-4; 1728])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_btjd(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
/// let e = ellipsoidal(&lc, &candidate).unwrap();
/// assert!((e.twice.cos + 5e-4).abs() < 1e-12);
/// assert!(e.once.amplitude() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn ellipsoidal(lc: &LightCurve, candidate: &Candidate) -> Option<Ellipsoidal> {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let t0 = candidate.transit_time().as_btjd();
    let omega = TAU / period;
    // Out of both eclipses: phase, weight, flux.
    let points: Vec<(f64, f64, f64)> = lc
        .time()
        .iter()
        .zip(lc.flux())
        .zip(lc.flux_err())
        .filter_map(|((t, &y), &e)| {
            let ph = phase(t.as_btjd() - t0, period);
            let far = ph.abs() >= duration && (ph.abs() - 0.5 * period).abs() >= duration;
            far.then(|| (ph, 1.0 / (e * e), y))
        })
        .collect();
    if points.len() < 6 {
        return None;
    }
    // The flux about its weighted mean, for small sums.
    let (mut w, mut wy) = (Sum::default(), Sum::default());
    for &(_, wi, y) in &points {
        w.add(wi);
        wy.add(wi * y);
    }
    let mean = wy.get() / w.get();
    // The normal equations for 1, cos, sin, cos 2, sin 2.
    let mut m = [[Sum::default(); 5]; 5];
    let mut v = [Sum::default(); 5];
    for &(ph, wi, y) in &points {
        let (s1, c1) = (omega * ph).sin_cos();
        let (s2, c2) = (2.0 * omega * ph).sin_cos();
        let x = [1.0, c1, s1, c2, s2];
        for i in 0..5 {
            v[i].add(wi * x[i] * (y - mean));
            for j in 0..5 {
                m[i][j].add(wi * x[i] * x[j]);
            }
        }
    }
    let m: Vec<Vec<f64>> = m
        .iter()
        .map(|row| row.iter().map(|x| x.get()).collect())
        .collect();
    let cov = invert(m)?;
    let params: Vec<f64> = (0..5)
        .map(|i| (0..5).map(|j| cov[i][j] * v[j].get()).sum())
        .collect();
    let baseline = mean + params[0];
    let term = |i: usize| (params[i] / baseline, cov[i][i].sqrt() / baseline);
    let harmonic = |i: usize| {
        let ((cos, cos_err), (sin, sin_err)) = (term(i), term(i + 1));
        Harmonic {
            cos,
            cos_err,
            sin,
            sin_err,
        }
    };
    Some(Ellipsoidal {
        once: harmonic(1),
        twice: harmonic(3),
        points: points.len(),
    })
}

/// The inverse of `m`, by Gauss-Jordan elimination with partial pivoting;
/// `None` where `m` is singular to working precision.
pub(crate) fn invert(mut m: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let scale = m.iter().flatten().fold(0.0_f64, |a, &x| a.max(x.abs()));
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() <= 1e-12 * scale {
            return None;
        }
        m.swap(col, pivot);
        inv.swap(col, pivot);
        let p = m[col][col];
        for j in 0..n {
            m[col][j] /= p;
            inv[col][j] /= p;
        }
        for row in 0..n {
            if row != col {
                let f = m[row][col];
                for j in 0..n {
                    m[row][j] -= f * m[col][j];
                    inv[row][j] -= f * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}
