//! One box, fitted exactly: the statistics of a single period, duration and
//! mid-transit time. The searches are checked against this.

use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::{BlsError, sums::Sum};

/// A box-shaped transit: a period, a duration and a mid-transit time, with
/// `0 < duration < period`.
///
/// A time is in transit if it lies strictly within `duration / 2` of
/// `transit_time + n * period` for some whole number `n`.
///
/// # Examples
///
/// ```
/// use rustellar_bls::TransitBox;
/// use rustellar_core::{Epoch, TimeDelta};
///
/// let transit = TransitBox::new(
///     TimeDelta::from_days(3.0)?,
///     TimeDelta::from_hours(3.0)?,
///     Epoch::from_btjd(1325.5)?,
/// )?;
/// assert!(transit.contains(Epoch::from_btjd(1331.5)?));
/// assert!(!transit.contains(Epoch::from_btjd(1330.0)?));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitBox {
    period: TimeDelta,
    duration: TimeDelta,
    transit_time: Epoch,
}

impl TransitBox {
    /// Checks and keeps the box.
    ///
    /// # Errors
    ///
    /// - [`BlsError::PeriodNotPositive`] if `period` is zero or negative.
    /// - [`BlsError::DurationNotPositive`] if `duration` is zero or negative.
    /// - [`BlsError::DurationNotShorterThanPeriod`] if `duration` is at least
    ///   `period`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{BlsError, TransitBox};
    /// use rustellar_core::{Epoch, TimeDelta};
    ///
    /// let one_day = TimeDelta::from_days(1.0)?;
    /// let t0 = Epoch::from_btjd(0.0)?;
    /// assert_eq!(
    ///     TransitBox::new(one_day, one_day, t0),
    ///     Err(BlsError::DurationNotShorterThanPeriod)
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        period: TimeDelta,
        duration: TimeDelta,
        transit_time: Epoch,
    ) -> Result<Self, BlsError> {
        if period.as_days() <= 0.0 {
            return Err(BlsError::PeriodNotPositive);
        }
        if duration.as_days() <= 0.0 {
            return Err(BlsError::DurationNotPositive);
        }
        if duration >= period {
            return Err(BlsError::DurationNotShorterThanPeriod);
        }
        Ok(Self {
            period,
            duration,
            transit_time,
        })
    }

    /// The period.
    #[must_use]
    pub fn period(&self) -> TimeDelta {
        self.period
    }

    /// The duration.
    #[must_use]
    pub fn duration(&self) -> TimeDelta {
        self.duration
    }

    /// The mid-transit time.
    #[must_use]
    pub fn transit_time(&self) -> Epoch {
        self.transit_time
    }

    /// Whether `time` is in transit.
    ///
    /// The time from the nearest mid-transit is worked out to within a
    /// rounding of its own size, not of the times' size, so the answer is
    /// right to about a picosecond at the edge of the box, thousands of
    /// orbits from `transit_time`. astropy's
    /// `abs((t - t0 + P/2) % P - P/2) < D/2` rounds at the size of `t`, about
    /// 20 ns at BTJD 3000, so it can put a point on the wrong side of an edge
    /// that close.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::TransitBox;
    /// use rustellar_core::{Epoch, TimeDelta};
    ///
    /// let transit = TransitBox::new(
    ///     TimeDelta::from_days(2.0)?,
    ///     TimeDelta::from_days(0.2)?,
    ///     Epoch::from_btjd(0.0)?,
    /// )?;
    /// assert!(transit.contains(Epoch::from_btjd(-3.95)?));
    /// assert!(!transit.contains(Epoch::from_btjd(0.1)?)); // the edge is outside
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn contains(&self, time: Epoch) -> bool {
        let offset = offset_from_transit(
            time.as_btjd(),
            self.transit_time.as_btjd(),
            self.period.as_days(),
        );
        offset.abs() < 0.5 * self.duration.as_days()
    }

    /// Fits the box to a light curve.
    ///
    /// # Errors
    ///
    /// - [`BlsError::NoPointsInTransit`] or [`BlsError::NoPointsOutOfTransit`]
    ///   if the box holds none, or all, of the points.
    /// - [`BlsError::OutOfRange`] if a weight `1 / flux_err^2` is too large
    ///   or too small for an `f64` to hold to full precision (a flux error
    ///   below about 1e-154 or above about 1e154), or a sum or statistic built
    ///   from the weights overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::TransitBox;
    /// use rustellar_core::{Epoch, LightCurve, TimeDelta};
    ///
    /// // Eight points a day apart, the third and fourth one lower.
    /// let lc = LightCurve::from_btjd(
    ///     (0..8).map(f64::from).collect(),
    ///     vec![0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    ///     vec![1.0; 8],
    /// )?;
    /// let transit = TransitBox::new(
    ///     TimeDelta::from_days(100.0)?,
    ///     TimeDelta::from_days(2.0)?,
    ///     Epoch::from_btjd(2.5)?,
    /// )?;
    /// let fit = transit.fit(&lc)?;
    /// assert_eq!(fit.depth, 1.0);
    /// assert_eq!(fit.delta_chi2, 1.5);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fit(&self, lc: &LightCurve) -> Result<BoxFit, BlsError> {
        let weighted = Weighted::new(lc)?;
        let (mut weight_in, mut weight_out) = (Sum::default(), Sum::default());
        let (mut sum_in, mut sum_out) = (Sum::default(), Sum::default());
        let mut n_in = 0;
        for ((&time, &w), &wy) in lc
            .time()
            .iter()
            .zip(&weighted.weights)
            .zip(&weighted.centred)
        {
            if self.contains(time) {
                n_in += 1;
                weight_in.add(w);
                sum_in.add(wy);
            } else {
                weight_out.add(w);
                sum_out.add(wy);
            }
        }
        if n_in == 0 {
            return Err(BlsError::NoPointsInTransit);
        }
        if n_in == lc.len() {
            return Err(BlsError::NoPointsOutOfTransit);
        }
        BoxFit::from_sums(
            weighted.mean,
            [weight_in.value(), sum_in.value()],
            [weight_out.value(), sum_out.value()],
            n_in,
        )
    }
}

/// A light curve's weights `w = 1 / flux_err^2` and the weighted flux
/// `w (y - mean)`, centred on the weighted mean: the box's levels are then
/// small numbers, and their difference, the depth, loses nothing to
/// cancellation when the flux is near 1 and the depth near 1e-4.
#[derive(Debug, Clone)]
pub(crate) struct Weighted {
    pub(crate) mean: f64,
    pub(crate) weights: Vec<f64>,
    pub(crate) centred: Vec<f64>,
}

impl Weighted {
    pub(crate) fn new(lc: &LightCurve) -> Result<Self, BlsError> {
        let weights: Vec<f64> = lc.flux_err().iter().map(|e| 1.0 / (e * e)).collect();
        // Zero or subnormal weights have lost their digits, or all of them,
        // and an infinite one swamps every other point.
        if !weights.iter().all(|w| w.is_normal()) {
            return Err(BlsError::OutOfRange);
        }
        let flux = lc.flux();
        let weight = weights.iter().copied().collect::<Sum>().value();
        let mean = weights
            .iter()
            .zip(flux)
            .map(|(w, y)| w * y)
            .collect::<Sum>()
            .value()
            / weight;
        let centred: Vec<f64> = weights
            .iter()
            .zip(flux)
            .map(|(w, y)| w * (y - mean))
            .collect();
        if !(mean.is_finite() && centred.iter().all(|x| x.is_finite())) {
            return Err(BlsError::OutOfRange);
        }
        Ok(Self {
            mean,
            weights,
            centred,
        })
    }
}

/// A box fitted to a light curve.
///
/// With weights `w = 1 / flux_err^2`, `W_in` and `W_out` the summed weights
/// in and out of transit and `W` their sum:
///
/// ```text
/// depth      = flux_out - flux_in
/// depth_err  = sqrt(1 / W_in + 1 / W_out)
/// snr        = depth / depth_err
/// delta_chi2 = W_in W_out / W  depth^2      (= snr^2)
/// likelihood = W_in depth^2 / 2             (= delta_chi2 W / (2 W_out))
/// ```
///
/// `delta_chi2` is the fall in chi-squared from a flat line to the box: the
/// BLS statistic of Kovacs, Zucker & Mazeh (2002). `likelihood` is what
/// astropy's `BoxLeastSquares` reports as its power: the log-likelihood
/// gained by the box with the out-of-transit level held fixed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BoxFit {
    /// The weighted mean flux in transit.
    pub flux_in: f64,
    /// The weighted mean flux out of transit.
    pub flux_out: f64,
    /// `flux_out - flux_in`: positive for a dip.
    pub depth: f64,
    /// The depth's standard error.
    pub depth_err: f64,
    /// `depth / depth_err`.
    pub snr: f64,
    /// The fall in chi-squared from a flat line to the box.
    pub delta_chi2: f64,
    /// astropy's log-likelihood power.
    pub likelihood: f64,
    /// The summed weight of the points in transit.
    pub weight_in: f64,
    /// The summed weight of the points out of transit.
    pub weight_out: f64,
    /// How many points are in transit.
    pub n_in: usize,
}

impl BoxFit {
    /// The fit from the summed weight and summed centred flux, `[W, S]`, in
    /// and out of transit, and the mean the flux was centred on.
    pub(crate) fn from_sums(
        mean: f64,
        [weight_in, sum_in]: [f64; 2],
        [weight_out, sum_out]: [f64; 2],
        n_in: usize,
    ) -> Result<Self, BlsError> {
        let level_in = sum_in / weight_in;
        let level_out = sum_out / weight_out;
        let depth = level_out - level_in;
        let depth_err = (1.0 / weight_in + 1.0 / weight_out).sqrt();
        let fit = Self {
            flux_in: mean + level_in,
            flux_out: mean + level_out,
            depth,
            depth_err,
            snr: depth / depth_err,
            // W_out / W first: W_in W_out alone overflows for weights above
            // about 1e154, though the statistic itself is far smaller.
            delta_chi2: weight_in * (weight_out / (weight_in + weight_out)) * depth * depth,
            likelihood: 0.5 * weight_in * depth * depth,
            weight_in,
            weight_out,
            n_in,
        };
        if !fit.is_finite() {
            return Err(BlsError::OutOfRange);
        }
        Ok(fit)
    }

    fn is_finite(&self) -> bool {
        [
            self.flux_in,
            self.flux_out,
            self.depth,
            self.depth_err,
            self.snr,
            self.delta_chi2,
            self.likelihood,
            self.weight_in,
            self.weight_out,
        ]
        .iter()
        .all(|x| x.is_finite())
    }
}

/// `t - t0 - n p` for the whole number `n` nearest `(t - t0) / p`, to within
/// a rounding or two of the result's own size.
///
/// `t - t0` is kept as an exact pair `s + e` (Knuth's two-sum), `n p` as an
/// exact pair `hi + lo`, and `s - hi` is exact because the two are within a
/// factor of two of each other (Sterbenz) or `hi` is zero. Only the last two
/// additions round, at the size of the result.
///
/// For `n p`, `p` is split into two halves of 26 bits (Veltkamp), so while
/// `|n| < 2^26` both `n p_hi` and `n p_lo` are exact products. Only plain
/// arithmetic is used, which is fast and the same on every platform; a fused
/// multiply-add, or `round`, is a slow library call on CPUs without them.
/// Beyond that, or for periods near the ends of the range, the product's
/// error comes from a fused multiply-add.
pub(crate) fn offset_from_transit(t: f64, t0: f64, p: f64) -> f64 {
    let s = t - t0;
    let s_back = s - t;
    let e = (t - (s - s_back)) - (t0 + s_back);
    let n = nearest(s / p);
    let (hi, lo) = if n.abs() < SPLIT_LIMIT && (PERIOD_LOW..PERIOD_HIGH).contains(&p.abs()) {
        let c = SPLITTER * p;
        let p_hi = c - (c - p);
        (n * p_hi, n * (p - p_hi))
    } else {
        let hi = n * p;
        (hi, n.mul_add(p, -hi))
    };
    ((s - hi) - lo) + e
}

/// 2^27 + 1: Veltkamp's splitter for doubles.
const SPLITTER: f64 = 134_217_729.0;
/// 2^26: below this, `n` has at most 26 bits.
const SPLIT_LIMIT: f64 = 67_108_864.0;
/// 2^-900 and 2^900: periods whose split neither underflows nor overflows.
const PERIOD_LOW: f64 = 1.183_052_186_166_774_7e-271;
const PERIOD_HIGH: f64 = 8.452_712_498_170_644e270;

/// The whole number nearest `x`, halves to even. Adding and taking away
/// 2^52 rounds a number below 2^52 to a whole number, with plain arithmetic;
/// from 2^52 up every double is whole.
pub(crate) fn nearest(x: f64) -> f64 {
    const TWO_52: f64 = 4_503_599_627_370_496.0;
    if x.abs() < TWO_52 {
        ((x.abs() + TWO_52) - TWO_52).copysign(x)
    } else {
        x
    }
}
