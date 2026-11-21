//! The false-alarm probability of a periodogram's highest peak under the
//! null hypothesis of Gaussian noise about the reference fit, and its
//! inverse, after Baluev (2008, MNRAS 385, 1279), Table 1 and equations 5
//! and 6; each in logarithms so small probabilities keep their digits.

use std::f64::consts::PI;

use crate::peaks::itp;
use crate::sums::Sum;
use crate::{LombScargle, LsError, Mean, Normalization};

/// How the false-alarm probability of the highest peak is approximated
/// (Baluev 2008; astropy's `method`).
///
/// # Examples
///
/// ```
/// use rustellar_ls::FalseAlarm;
///
/// assert_eq!(FalseAlarm::default(), FalseAlarm::Baluev);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FalseAlarm {
    /// The alias-free approximation, `1 - (1 - p1) e^(-tau)` (equation 6):
    /// an upper bound, close where the probability is small.
    #[default]
    Baluev,
    /// The Davies bound `p1 + tau` (equation 5), at most 1.
    Davies,
    /// `1 - (1 - p1)^(f_max T)`: the frequencies up to `f_max` taken as
    /// `f_max T` independent ones over the span `T`.
    Naive,
    /// `p1`: the probability at one frequency fixed in advance.
    Single,
}

/// The null distribution for one set-up, normalisation and maximum
/// frequency.
struct Null {
    normalization: Normalization,
    method: FalseAlarm,
    /// The degrees of freedom without and with the sinusoid.
    nh: f64,
    nk: f64,
    /// `ln W`, `W = f_max sqrt(4 pi var(t))` the effective number of
    /// frequencies, and `ln gamma(N)`.
    ln_w: f64,
    ln_gamma: f64,
    /// `f_max T`.
    independent: f64,
}

impl Null {
    /// `ln p1(z)`, the single frequency's probability.
    fn ln_single(&self, z: f64) -> f64 {
        match self.normalization {
            Normalization::Standard => 0.5 * self.nk * (-z).ln_1p(),
            Normalization::Model => -0.5 * self.nk * z.ln_1p(),
            Normalization::Log => -0.5 * self.nk * z,
            Normalization::Psd => -z,
        }
    }

    /// `ln tau(z)` (Table 1), for `z` above 0.
    fn ln_tau(&self, z: f64) -> f64 {
        let (nh, nk) = (self.nh, self.nk);
        // The PSD's tau has no gamma (Table 1).
        let gamma = if self.normalization == Normalization::Psd {
            0.0
        } else {
            self.ln_gamma
        };
        self.ln_w
            + gamma
            + match self.normalization {
                Normalization::Standard => {
                    0.5 * (nk - 1.0) * (-z).ln_1p() + 0.5 * (0.5 * nh * z).ln()
                }
                Normalization::Model => -0.5 * nk * z.ln_1p() + 0.5 * (0.5 * nk * z).ln(),
                Normalization::Log => {
                    // sinh(z / 2) = e^(z / 2) (1 - e^(-z)) / 2, without
                    // overflow.
                    let ln_sinh = 0.5 * z + (-(-z).exp()).ln_1p() - std::f64::consts::LN_2;
                    -0.5 * z * (nk - 0.5) + 0.5 * (nk.ln() + ln_sinh)
                }
                Normalization::Psd => -z + 0.5 * z.ln(),
            }
    }

    /// The probability at power `z`.
    fn probability(&self, z: f64) -> f64 {
        let top = matches!(self.normalization, Normalization::Standard) && z >= 1.0;
        if z <= 0.0 {
            return 1.0;
        }
        if top {
            return 0.0;
        }
        let single = self.ln_single(z).exp();
        match self.method {
            FalseAlarm::Single => single,
            FalseAlarm::Naive => -(self.independent * (-single).ln_1p()).exp_m1(),
            FalseAlarm::Davies => (single + self.ln_tau(z).exp()).min(1.0),
            FalseAlarm::Baluev => {
                let tau = self.ln_tau(z).exp();
                -(-tau).exp_m1() + single * (-tau).exp()
            }
        }
    }

    /// The power whose single-frequency probability is `e^ln_p`.
    fn single_level(&self, ln_p: f64) -> f64 {
        let k = 2.0 / self.nk;
        match self.normalization {
            Normalization::Standard => -(k * ln_p).exp_m1(),
            Normalization::Model => (-k * ln_p).exp_m1(),
            Normalization::Log => -k * ln_p,
            Normalization::Psd => -ln_p,
        }
    }

    /// The power at which the probability is `p`.
    fn level(&self, p: f64) -> f64 {
        match self.method {
            FalseAlarm::Single => self.single_level(p.ln()),
            FalseAlarm::Naive => {
                let single = -((-p).ln_1p() / self.independent).exp_m1();
                self.single_level(single.ln())
            }
            FalseAlarm::Davies | FalseAlarm::Baluev => {
                // Both are at least the single frequency's probability, so
                // its level is below; above, doubling it (to at most 1 in
                // the standard normalisation) until they are below p. A
                // bracket near the root keeps the stopping width relative
                // to it.
                let low = self.single_level(p.ln());
                let top = if self.normalization == Normalization::Standard {
                    1.0
                } else {
                    f64::INFINITY
                };
                let mut high = (2.0 * low).min(top);
                while self.probability(high) >= p {
                    high = (2.0 * high).min(top);
                }
                let g = |z: f64| self.probability(z) - p;
                itp(g, low, high, g(low), g(high))
            }
        }
    }
}

impl LombScargle {
    /// The probability that noise alone (Gaussian, with the flux's errors,
    /// about the reference fit) gives a highest peak of at least `power`
    /// in `normalization` over frequencies up to `maximum_frequency`, in
    /// cycles per day, by `method` (astropy's `false_alarm_probability`).
    ///
    /// The degrees of freedom are Baluev's (2008, Table 1): one parameter
    /// for the reference fit and three with the sinusoid with a fitted or
    /// subtracted mean, none and two without a mean (where astropy gives
    /// none). Each probability is computed in logarithms, so one far below
    /// `1e-16` keeps its digits, and the ratio of gamma functions in `tau`
    /// from its asymptotic series, not a difference of their logarithms.
    ///
    /// # Errors
    ///
    /// - [`LsError::NonFinite`] if `power` or `maximum_frequency` is NaN
    ///   (or the latter infinite), [`LsError::NotPositive`] unless
    ///   `maximum_frequency` is above zero.
    /// - [`LsError::TooFewPoints`] unless the fit with the sinusoid leaves
    ///   a degree of freedom.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_ls::{FalseAlarm, LightCurve, LombScargle, Mean, Normalization};
    ///
    /// let time: Vec<f64> = (0..100).map(|i| 0.3 * f64::from(i) + 0.01 * f64::from(i % 7)).collect();
    /// let flux = (0..100).map(|i| f64::from((i * 37) % 11) - 5.0).collect();
    /// let lc = LightCurve::from_days(time, flux, vec![1.0; 100])?;
    /// let ls = LombScargle::new(&lc, Mean::Floating)?;
    ///
    /// let n = Normalization::Standard;
    /// let level = ls.false_alarm_level(0.01, n, 2.0, FalseAlarm::Baluev)?;
    /// let p = ls.false_alarm_probability(level, n, 2.0, FalseAlarm::Baluev)?;
    /// assert!((p - 0.01).abs() < 1e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn false_alarm_probability(
        &self,
        power: f64,
        normalization: Normalization,
        maximum_frequency: f64,
        method: FalseAlarm,
    ) -> Result<f64, LsError> {
        if power.is_nan() {
            return Err(LsError::NonFinite { name: "power" });
        }
        Ok(self
            .null(normalization, maximum_frequency, method)?
            .probability(power))
    }

    /// The power of the highest peak that noise alone exceeds with
    /// `probability`: the inverse of
    /// [`false_alarm_probability`](Self::false_alarm_probability), to the
    /// last bits (astropy's `false_alarm_level`).
    ///
    /// # Errors
    ///
    /// [`LsError::NotProbability`] unless `probability` is above 0 and
    /// below 1; the others as
    /// [`false_alarm_probability`](Self::false_alarm_probability).
    pub fn false_alarm_level(
        &self,
        probability: f64,
        normalization: Normalization,
        maximum_frequency: f64,
        method: FalseAlarm,
    ) -> Result<f64, LsError> {
        if !(probability > 0.0 && probability < 1.0) {
            return Err(LsError::NotProbability);
        }
        Ok(self
            .null(normalization, maximum_frequency, method)?
            .level(probability))
    }

    fn null(
        &self,
        normalization: Normalization,
        maximum_frequency: f64,
        method: FalseAlarm,
    ) -> Result<Null, LsError> {
        if !maximum_frequency.is_finite() {
            return Err(LsError::NonFinite {
                name: "maximum_frequency",
            });
        }
        if maximum_frequency <= 0.0 {
            return Err(LsError::NotPositive {
                name: "maximum_frequency",
            });
        }
        let (dh, dk) = if self.mean == Mean::Zero {
            (0, 2)
        } else {
            (1, 3)
        };
        let n = self.time.len();
        if n <= dk {
            return Err(LsError::TooFewPoints { needed: dk + 1 });
        }
        #[expect(clippy::cast_precision_loss, reason = "point counts")]
        let (nh, nk) = ((n - dh) as f64, (n - dk) as f64);
        // The weighted variance of the times, in two compensated passes.
        let (mut w, mut wt) = (Sum::default(), Sum::default());
        for (&wi, &t) in self.weight.iter().zip(&self.time) {
            w.add(wi);
            wt.add(wi * t);
        }
        let (w, mean) = (w.value(), wt.value() / w.value());
        let mut spread = Sum::default();
        for (&wi, &t) in self.weight.iter().zip(&self.time) {
            spread.add(wi * (t - mean) * (t - mean));
        }
        let variance = spread.value() / w;
        let effective = maximum_frequency * (4.0 * PI * variance).sqrt();
        // gamma(N) = sqrt(2 / N) G(N / 2) / G((N - 1) / 2), with N = N_H
        // for the standard normalisation and N_K for the others.
        let gamma_of = if normalization == Normalization::Standard {
            nh
        } else {
            nk
        };
        let gamma = (2.0 / gamma_of).sqrt() * gamma_ratio(0.5 * (gamma_of - 1.0));
        let span = self.time[n - 1] - self.time[0];
        Ok(Null {
            normalization,
            method,
            nh,
            nk,
            ln_w: effective.ln(),
            ln_gamma: gamma.ln(),
            independent: maximum_frequency * span,
        })
    }
}

/// `G(x + 1/2) / G(x)` for `x` at least 1/2, `G` the gamma function: up by
/// `G(x + 3/2) / G(x + 1) = (x + 1/2) / x G(x + 1/2) / G(x)` to at least
/// 20, then `sqrt(x) e^s` with `s` the asymptotic series of
/// `ln(G(x + 1/2) / G(x)) - ln(x) / 2` in odd powers of `1 / x` (from the
/// Bernoulli polynomials at 1/2 and 0), to below a rounding.
pub(crate) fn gamma_ratio(x: f64) -> f64 {
    let mut x = x;
    let mut factor = 1.0;
    while x < 20.0 {
        factor *= x / (x + 0.5);
        x += 1.0;
    }
    let inverse = 1.0 / x;
    let i2 = inverse * inverse;
    let series = inverse
        * (-1.0 / 8.0
            + i2 * (1.0 / 192.0
                + i2 * (-1.0 / 640.0
                    + i2 * (17.0 / 14336.0 + i2 * (-31.0 / 18432.0 + i2 * (691.0 / 180_224.0))))));
    factor * x.sqrt() * series.exp()
}
