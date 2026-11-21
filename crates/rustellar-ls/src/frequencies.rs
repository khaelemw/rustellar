//! The trial frequencies: an even grid, which the fast methods need, or any
//! list; and astropy's automatic grid.

use rustellar_core::LightCurve;

use crate::LsError;

/// The most frequencies a grid may hold.
const MAX_FREQUENCIES: usize = 1_000_000_000;

/// The trial frequencies, in cycles per day of the light curve's own time
/// scale: at least one, each greater than zero and finite.
///
/// An even grid (`start + k step`) is kept as its start, step and count,
/// and is what the fast methods need; any other list is kept as given, in
/// its order.
///
/// # Examples
///
/// ```
/// use rustellar_ls::Frequencies;
///
/// let even = Frequencies::uniform(0.1, 0.01, 100)?;
/// assert_eq!(even.len(), 100);
/// assert_eq!(even.get(10), 0.1 + 10.0 * 0.01);
/// assert_eq!(even.uniform_step(), Some(0.01));
///
/// let listed = Frequencies::from_per_day(&[2.0, 0.5, 1.0])?;
/// assert_eq!(listed.iter().collect::<Vec<_>>(), [2.0, 0.5, 1.0]);
/// assert_eq!((listed.min(), listed.max()), (0.5, 2.0));
/// # Ok::<(), rustellar_ls::LsError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Frequencies(Repr);

#[derive(Debug, Clone, PartialEq)]
enum Repr {
    Uniform { start: f64, step: f64, count: usize },
    Listed(Vec<f64>),
}

impl Frequencies {
    /// `count` frequencies from `start`, `step` apart.
    ///
    /// # Errors
    ///
    /// - [`LsError::NonFinite`] if `start` or `step` is NaN or infinite,
    ///   or the last frequency would be infinite.
    /// - [`LsError::NotPositive`] if `start` or `step` is not above zero
    ///   (`step` may be anything finite when `count` is 1).
    /// - [`LsError::NoFrequencies`] if `count` is 0.
    /// - [`LsError::GridTooLarge`] if `count` is above a billion.
    pub fn uniform(start: f64, step: f64, count: usize) -> Result<Self, LsError> {
        positive(start, "frequency")?;
        if count == 0 {
            return Err(LsError::NoFrequencies);
        }
        if count > MAX_FREQUENCIES {
            return Err(LsError::GridTooLarge);
        }
        if count == 1 {
            if !step.is_finite() {
                return Err(LsError::NonFinite { name: "step" });
            }
            return Ok(Self(Repr::Uniform {
                start,
                step: 0.0,
                count,
            }));
        }
        positive(step, "step")?;
        #[expect(clippy::cast_precision_loss, reason = "at most a billion")]
        let last = start + step * (count - 1) as f64;
        if !last.is_finite() {
            return Err(LsError::NonFinite { name: "frequency" });
        }
        Ok(Self(Repr::Uniform { start, step, count }))
    }

    /// Any list of frequencies, in cycles per day, kept in its order.
    ///
    /// # Errors
    ///
    /// - [`LsError::NoFrequencies`] if the list is empty.
    /// - [`LsError::NonFinite`] if one is NaN or infinite.
    /// - [`LsError::NotPositive`] if one is not above zero.
    pub fn from_per_day(frequencies: &[f64]) -> Result<Self, LsError> {
        if frequencies.is_empty() {
            return Err(LsError::NoFrequencies);
        }
        for &f in frequencies {
            positive(f, "frequency")?;
        }
        Ok(Self(Repr::Listed(frequencies.to_vec())))
    }

    /// How many frequencies.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.0 {
            Repr::Uniform { count, .. } => *count,
            Repr::Listed(v) => v.len(),
        }
    }

    /// Never: a grid holds at least one frequency.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// The `k`th frequency.
    ///
    /// # Panics
    ///
    /// If `k` is not below [`len`](Self::len).
    #[must_use]
    pub fn get(&self, k: usize) -> f64 {
        match &self.0 {
            Repr::Uniform { start, step, count } => {
                assert!(k < *count, "frequency {k} of {count}");
                #[expect(clippy::cast_precision_loss, reason = "at most a billion")]
                let k = k as f64;
                // As astropy's `f0 + df * arange(n)`: rounded twice.
                start + step * k
            }
            Repr::Listed(v) => v[k],
        }
    }

    /// Every frequency, in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        (0..self.len()).map(|k| self.get(k))
    }

    /// The step of an even grid made by [`uniform`](Self::uniform), or
    /// `None` for a list (even an evenly spaced one).
    #[must_use]
    pub const fn uniform_step(&self) -> Option<f64> {
        match &self.0 {
            Repr::Uniform { step, .. } => Some(*step),
            Repr::Listed(_) => None,
        }
    }

    /// The lowest frequency.
    #[must_use]
    pub fn min(&self) -> f64 {
        match &self.0 {
            Repr::Uniform { start, .. } => *start,
            Repr::Listed(v) => v.iter().copied().fold(f64::INFINITY, f64::min),
        }
    }

    /// The highest frequency.
    #[must_use]
    pub fn max(&self) -> f64 {
        match &self.0 {
            Repr::Uniform { count, .. } => self.get(count - 1),
            Repr::Listed(v) => v.iter().copied().fold(0.0, f64::max),
        }
    }
}

/// astropy's automatic frequency grid (VanderPlas 2018, section 7.1): an
/// even grid from half a step to a multiple of the average Nyquist
/// frequency, with `samples_per_peak` frequencies across a peak's width.
///
/// For a light curve of `n` points spanning `T` days, the step is
/// `1 / (T samples_per_peak)`, the lowest frequency half a step (or
/// `minimum`) and the highest `nyquist_factor n / (2 T)` (or `maximum`);
/// the count is `1 + round((highest - lowest) / step)`, as astropy's
/// `LombScargle.autofrequency`.
///
/// The defaults (5 samples per peak, Nyquist factor 5) are astropy's. The
/// average Nyquist frequency means nothing for uneven sampling: it is only
/// a scale, and for data with a short regular cadence the highest frequency
/// worth searching is usually much lower; set `maximum` for your data.
///
/// # Examples
///
/// ```
/// use rustellar_core::LightCurve;
/// use rustellar_ls::AutoFrequency;
///
/// // 100 points over 9.9 days.
/// let time: Vec<f64> = (0..100).map(|i| 0.1 * f64::from(i)).collect();
/// let lc = LightCurve::from_days(time, vec![1.0; 100], vec![0.01; 100])?;
/// let grid = AutoFrequency::default().with_maximum(2.01).frequencies(&lc)?;
/// let step = 1.0 / (0.1 * 99.0) / 5.0;
/// assert_eq!(grid.uniform_step(), Some(step));
/// assert_eq!(grid.min(), step / 2.0);
/// assert_eq!(grid.len(), 100);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoFrequency {
    samples_per_peak: f64,
    nyquist_factor: f64,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

impl Default for AutoFrequency {
    /// astropy's: 5 samples per peak, Nyquist factor 5, the lowest and
    /// highest frequencies from the data.
    fn default() -> Self {
        Self {
            samples_per_peak: 5.0,
            nyquist_factor: 5.0,
            minimum: None,
            maximum: None,
        }
    }
}

impl AutoFrequency {
    /// Frequencies across a peak's width (`1 / T`).
    #[must_use]
    pub const fn with_samples_per_peak(self, samples_per_peak: f64) -> Self {
        Self {
            samples_per_peak,
            ..self
        }
    }

    /// The highest frequency as a multiple of the average Nyquist
    /// frequency, `n / (2 T)`; unused with a `maximum`.
    #[must_use]
    pub const fn with_nyquist_factor(self, nyquist_factor: f64) -> Self {
        Self {
            nyquist_factor,
            ..self
        }
    }

    /// The lowest frequency, in cycles per day, instead of half a step.
    #[must_use]
    pub const fn with_minimum(self, minimum: f64) -> Self {
        Self {
            minimum: Some(minimum),
            ..self
        }
    }

    /// The highest frequency, in cycles per day, instead of the Nyquist
    /// factor's.
    #[must_use]
    pub const fn with_maximum(self, maximum: f64) -> Self {
        Self {
            maximum: Some(maximum),
            ..self
        }
    }

    /// The samples per peak.
    #[must_use]
    pub const fn samples_per_peak(&self) -> f64 {
        self.samples_per_peak
    }

    /// The Nyquist factor.
    #[must_use]
    pub const fn nyquist_factor(&self) -> f64 {
        self.nyquist_factor
    }

    /// The lowest frequency, if set.
    #[must_use]
    pub const fn minimum(&self) -> Option<f64> {
        self.minimum
    }

    /// The highest frequency, if set.
    #[must_use]
    pub const fn maximum(&self) -> Option<f64> {
        self.maximum
    }

    /// The grid for `lc`.
    ///
    /// # Errors
    ///
    /// - [`LsError::NoBaseline`] if `lc` spans no time.
    /// - [`LsError::NonFinite`] or [`LsError::NotPositive`] if a setting
    ///   is NaN, infinite, or not above zero.
    /// - [`LsError::FrequencyRangeEmpty`] if the lowest frequency is above
    ///   the highest.
    /// - [`LsError::GridTooLarge`] if the grid would hold more than a
    ///   billion frequencies.
    pub fn frequencies(&self, lc: &LightCurve) -> Result<Frequencies, LsError> {
        positive(self.samples_per_peak, "samples_per_peak")?;
        let time = lc.time();
        let (first, last) = (time[0].as_days(), time[time.len() - 1].as_days());
        let baseline = last - first;
        if baseline <= 0.0 {
            return Err(LsError::NoBaseline);
        }
        let step = 1.0 / baseline / self.samples_per_peak;
        positive(step, "step")?;
        let lowest = match self.minimum {
            Some(f) => positive(f, "minimum")?,
            None => 0.5 * step,
        };
        let highest = match self.maximum {
            Some(f) => positive(f, "maximum")?,
            None => {
                positive(self.nyquist_factor, "nyquist_factor")?;
                #[expect(clippy::cast_precision_loss, reason = "a point count")]
                let n = time.len() as f64;
                self.nyquist_factor * (0.5 * n / baseline)
            }
        };
        if lowest > highest {
            return Err(LsError::FrequencyRangeEmpty);
        }
        let count = ((highest - lowest) / step).round();
        #[expect(clippy::cast_precision_loss, reason = "the cap, exact")]
        if count >= MAX_FREQUENCIES as f64 {
            return Err(LsError::GridTooLarge);
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a whole number from 0 to below a billion"
        )]
        let count = count as usize + 1;
        Frequencies::uniform(lowest, step, count)
    }
}

/// `value` if finite and above zero.
fn positive(value: f64, name: &'static str) -> Result<f64, LsError> {
    if !value.is_finite() {
        Err(LsError::NonFinite { name })
    } else if value <= 0.0 {
        Err(LsError::NotPositive { name })
    } else {
        Ok(value)
    }
}
