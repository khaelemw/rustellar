//! The periodogram's peaks refined off the grid: each to where the power's
//! derivative is zero, from the same centred sums as the power, by the ITP
//! method (Oliveira & Takahashi 2020, ACM Trans. Math. Softw. 47, 5).

use crate::periodogram::{Buffer, has_avx2, solve};
use crate::{LombScargle, Mean, Normalization, Periodogram};

/// A peak of a periodogram: its frequency and power.
///
/// # Examples
///
/// ```
/// use rustellar_ls::{Frequencies, LightCurve, LombScargle, Mean, Normalization};
///
/// let time: Vec<f64> = (0..300).map(|i| 0.1 * f64::from(i) + 0.01 * f64::from(i % 7)).collect();
/// let flux = time.iter().map(|t| (std::f64::consts::TAU * 0.4321 * t).sin()).collect();
/// let lc = LightCurve::from_days(time, flux, vec![0.1; 300])?;
/// let ls = LombScargle::new(&lc, Mean::Floating)?;
/// let p = ls.power(&Frequencies::uniform(0.05, 0.01, 100)?, Normalization::Standard);
///
/// let best = ls.peaks(&p, 1)[0];
/// assert!((best.frequency() - 0.4321).abs() < 1e-12);
/// assert!((best.period() - 1.0 / 0.4321).abs() < 1e-11);
/// assert!(best.power() > 1.0 - 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    frequency: f64,
    power: f64,
}

impl Peak {
    /// The frequency, in cycles per day.
    #[must_use]
    pub const fn frequency(&self) -> f64 {
        self.frequency
    }

    /// The period, in days.
    #[must_use]
    pub fn period(&self) -> f64 {
        1.0 / self.frequency
    }

    /// The power, in the periodogram's normalisation.
    #[must_use]
    pub const fn power(&self) -> f64 {
        self.power
    }
}

impl LombScargle {
    /// The `count` highest peaks of `periodogram` (computed from this
    /// set-up), highest first, each refined off the grid.
    ///
    /// A peak is a frequency with more power than the one before and at
    /// least as much as the one after (the grid's ends against their one
    /// neighbour). Each is moved to where the power's derivative is zero
    /// between it and the neighbour the derivative points to, found to
    /// the last bits of the frequency from the derivative's own sums, and
    /// its power computed exactly there. Where the derivative does not
    /// change sign before that neighbour (or there is none) the peak stays
    /// on the grid. Interpolating the power itself (a parabola through
    /// three powers, or searching it) cannot place a peak closer than
    /// about `1e-8` of its width, where the power's change falls below its
    /// rounding; the derivative's zero is found to rounding.
    #[must_use]
    pub fn peaks(&self, periodogram: &Periodogram, count: usize) -> Vec<Peak> {
        let p = periodogram.power();
        let last = p.len() - 1;
        let mut found: Vec<usize> = (0..=last)
            .filter(|&k| (k == 0 || p[k] > p[k - 1]) && (k == last || p[k] >= p[k + 1]))
            .collect();
        // Highest first, the first of a tie.
        found.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        found.truncate(count);
        let grid = periodogram.frequencies();
        let normalization = periodogram.normalization();
        let residual = matches!(normalization, Normalization::Model | Normalization::Log);
        let wide = has_avx2();
        let mut buffer = Buffer::new(self.time.len());
        let mut peaks: Vec<Peak> = found
            .into_iter()
            .map(|k| {
                let f = grid.get(k);
                let g = self.slope(f, &mut buffer);
                // The neighbour the derivative points to.
                let other = if g > 0.0 {
                    (k < last).then_some(k + 1)
                } else {
                    k.checked_sub(1)
                };
                let frequency = other
                    .map(|o| (grid.get(o), self.slope(grid.get(o), &mut buffer)))
                    .filter(|&(_, go)| g != 0.0 && (go > 0.0) != (g > 0.0))
                    .map_or(f, |(fo, go)| {
                        let ((a, ga), (b, gb)) = if f < fo {
                            ((f, g), (fo, go))
                        } else {
                            ((fo, go), (f, g))
                        };
                        itp(|x| self.slope(x, &mut buffer), a, b, ga, gb)
                    });
                let fit = self.fit(frequency, residual, wide, &mut buffer);
                Peak {
                    frequency,
                    power: self.normalized(fit, normalization),
                }
            })
            .collect();
        peaks.sort_by(|a, b| b.power.total_cmp(&a.power));
        peaks
    }

    /// The derivative of the reduction in chi-squared at `f`, over `2 pi`:
    /// `2 x . db - x . dA x` for the normal equations `A x = b` of the
    /// centred terms, from the terms' derivatives `-t s` and `t c`
    /// (centred too with a floating mean).
    fn slope(&self, f: f64, buffer: &mut Buffer) -> f64 {
        let (w, c_mean, s_mean) = self.centred(f, buffer);
        let [cc, cs, ss, yc, ys] = self.sums(buffer);
        #[expect(clippy::cast_precision_loss, reason = "a point count")]
        let (beta, gamma) = solve(cc, cs, ss, yc, ys, self.time.len() as f64);
        // The terms' derivatives' weighted means, with a floating mean.
        let (mut du, mut dv) = (0.0, 0.0);
        if self.mean == Mean::Floating {
            for (((c, s), &t), &wi) in buffer
                .c
                .iter()
                .zip(&buffer.s)
                .zip(&self.time)
                .zip(&self.weight)
            {
                du -= wi * t * (s.0 + s_mean);
                dv += wi * t * (c.0 + c_mean);
            }
            (du, dv) = (du / w, dv / w);
        }
        let mut d = [0.0; 5];
        for ((((c, s), &t), &wi), &y) in buffer
            .c
            .iter()
            .zip(&buffer.s)
            .zip(&self.time)
            .zip(&self.weight)
            .zip(&self.flux)
        {
            let (c, s) = (c.0, s.0);
            // The raw terms' derivatives, less their means.
            let u = -t * (s + s_mean) - du;
            let v = t * (c + c_mean) - dv;
            d[0] += wi * c * u;
            d[1] += wi * (u * s + c * v);
            d[2] += wi * s * v;
            d[3] += wi * u * y;
            d[4] += wi * v * y;
        }
        let [dcc, dcs, dss, dyc, dys] = d;
        2.0 * (beta * dyc + gamma * dys)
            - (2.0 * beta * beta * dcc + 2.0 * beta * gamma * dcs + 2.0 * gamma * gamma * dss)
    }
}

/// The zero of `g` between `a` and `b` (`a < b`, `g(a)` and `g(b)` of
/// opposite signs) to the spacing of the floats there, by the ITP method
/// (Oliveira & Takahashi 2020): never more steps than bisection's plus
/// one, superlinear where `g` is smooth.
pub(crate) fn itp(mut g: impl FnMut(f64) -> f64, a: f64, b: f64, ga: f64, gb: f64) -> f64 {
    // Signed so it is below zero at a and above at b.
    let sign = if ga < 0.0 { 1.0 } else { -1.0 };
    let (mut a, mut b, mut ya, mut yb) = (a, b, sign * ga, sign * gb);
    let epsilon = f64::EPSILON * a.abs().max(b.abs());
    let width = b - a;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a small positive count of halvings"
    )]
    let most = (width / (2.0 * epsilon)).log2().ceil().max(0.0) as i32 + 1;
    let kappa = 0.2 / width;
    let mut j = 0;
    while b - a > 2.0 * epsilon && j <= most {
        let half = 0.5 * (a + b);
        let r = epsilon * 2_f64.powi(most - j) - 0.5 * (b - a);
        let delta = kappa * (b - a) * (b - a);
        // Regula falsi, truncated towards the middle, then projected
        // within r of it.
        let falsi = (yb * a - ya * b) / (yb - ya);
        let toward = (half - falsi).signum();
        let truncated = if delta <= (half - falsi).abs() {
            falsi + toward * delta
        } else {
            half
        };
        let x = if (truncated - half).abs() <= r {
            truncated
        } else {
            half - toward * r
        };
        // A zero exactly at x is kept as the lower end.
        let y = sign * g(x);
        if y > 0.0 {
            (b, yb) = (x, y);
        } else {
            (a, ya) = (x, y);
        }
        j += 1;
    }
    0.5 * (a + b)
}
