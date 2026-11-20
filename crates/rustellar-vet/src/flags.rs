//! Flags from what the user knows about the data: periods a candidate
//! should not share, and times its transits should not fall in.

use std::num::NonZeroU16;

use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::evidence::layout;
use crate::{Candidate, Ratio, VetError};

/// A period a real signal is unlikely to have: that of something the
/// instrument, the data's processing or the observing pattern repeats
/// (an orbit, a rotation, a day). A candidate matches it at `p / q` times
/// the period, for `p` up to `multiples` and `q` up to `fractions`, within
/// a relative `tolerance`.
///
/// None is built in: which periods matter depends on the data.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroU16;
///
/// use rustellar_vet::{SuspectPeriod, TimeDelta};
///
/// // A 1-day systematic, and every whole number of days up to 30, within
/// // 0.1%.
/// let days = SuspectPeriod::new(TimeDelta::from_days(1.0)?, 1e-3)?
///     .with_multiples(NonZeroU16::new(30).unwrap());
/// assert_eq!(days.multiples().get(), 30);
/// assert_eq!(days.fractions().get(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuspectPeriod {
    period: TimeDelta,
    tolerance: f64,
    multiples: NonZeroU16,
    fractions: NonZeroU16,
}

impl SuspectPeriod {
    /// `period` itself (`multiples` and `fractions` 1), matched within a
    /// relative `tolerance`: `|P / (period p / q) - 1| <= tolerance`.
    ///
    /// # Errors
    ///
    /// - [`VetError::NotPositive`] if `period` is not above zero.
    /// - [`VetError::ToleranceOutOfRange`] unless `0 <= tolerance < 1`.
    pub fn new(period: TimeDelta, tolerance: f64) -> Result<Self, VetError> {
        if period.as_days() <= 0.0 {
            return Err(VetError::NotPositive {
                name: "suspect period",
            });
        }
        if !(0.0..1.0).contains(&tolerance) {
            return Err(VetError::ToleranceOutOfRange);
        }
        Ok(Self {
            period,
            tolerance,
            multiples: NonZeroU16::MIN,
            fractions: NonZeroU16::MIN,
        })
    }

    /// Matches at `p` times the period, for every `p` up to `multiples`.
    #[must_use]
    pub const fn with_multiples(self, multiples: NonZeroU16) -> Self {
        Self { multiples, ..self }
    }

    /// Matches at the period over `q`, for every `q` up to `fractions`.
    #[must_use]
    pub const fn with_fractions(self, fractions: NonZeroU16) -> Self {
        Self { fractions, ..self }
    }

    /// The period.
    #[must_use]
    pub const fn period(&self) -> TimeDelta {
        self.period
    }

    /// The relative tolerance.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// The largest multiple matched.
    #[must_use]
    pub const fn multiples(&self) -> NonZeroU16 {
        self.multiples
    }

    /// The largest fraction matched.
    #[must_use]
    pub const fn fractions(&self) -> NonZeroU16 {
        self.fractions
    }
}

/// A candidate's period matched to a [`SuspectPeriod`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PeriodFlag {
    /// Which suspect, its index in the list given.
    pub suspect: usize,
    /// The ratio matched: the candidate's period is about this times the
    /// suspect's.
    pub ratio: Ratio,
    /// The suspect's period times the ratio.
    pub period: TimeDelta,
    /// The candidate's period over that, less 1.
    pub offset: f64,
}

/// Every [`SuspectPeriod`] `candidate`'s period matches, and at which
/// ratio, the closest first (by `|offset|`, then the order given). A ratio
/// counts once, in lowest terms: `2/2` is `1`.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroU16;
///
/// use rustellar_vet::{Candidate, Epoch, SuspectPeriod, TimeDelta, flag_periods};
///
/// let candidate = Candidate::new(
///     TimeDelta::from_days(6.851)?,
///     Epoch::from_days(1.0)?,
///     TimeDelta::from_hours(3.0)?,
/// )?;
/// // An orbit of 13.7 days, and its half, third and quarter, within 0.5%.
/// let orbit = SuspectPeriod::new(TimeDelta::from_days(13.7)?, 5e-3)?
///     .with_fractions(NonZeroU16::new(4).unwrap());
/// let flags = flag_periods(&candidate, &[orbit]);
/// assert_eq!(flags.len(), 1);
/// assert_eq!(flags[0].ratio.to_string(), "1/2");
/// assert!((flags[0].offset - 0.001 / 6.85).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn flag_periods(candidate: &Candidate, suspects: &[SuspectPeriod]) -> Vec<PeriodFlag> {
    let period = candidate.period().as_days();
    let mut flags = Vec::new();
    for (k, s) in suspects.iter().enumerate() {
        let base = s.period.as_days();
        for q in 1..=s.fractions.get() {
            // The multiples nearest P q / base; a neighbour of the nearest
            // only matters for a tolerance near 1 / p.
            let nearest = (period * f64::from(q) / base).round();
            for p in [nearest - 1.0, nearest, nearest + 1.0] {
                if p < 1.0 || p > f64::from(s.multiples.get()) {
                    continue;
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a whole number from 1 to a u16"
                )]
                let p = p as u16;
                let ratio = Ratio::new(p, q).expect("above zero");
                if (ratio.numerator(), ratio.denominator()) != (p, q) {
                    continue;
                }
                let target = base * f64::from(p) / f64::from(q);
                let offset = period / target - 1.0;
                if offset.abs() <= s.tolerance {
                    flags.push(PeriodFlag {
                        suspect: k,
                        ratio,
                        period: TimeDelta::from_days(target).expect("finite"),
                        offset,
                    });
                }
            }
        }
    }
    flags.sort_by(|a, b| a.offset.abs().total_cmp(&b.offset.abs()));
    flags
}

/// A span of time a transit should not fall in: a gap's edge, a known
/// event of the instrument, data flagged as poor.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Epoch, TimeDelta, TimeWindow};
///
/// let event = TimeWindow::around(Epoch::from_days(10.0)?, TimeDelta::from_hours(1.0)?)?;
/// assert!(((event.end() - event.start()).as_hours() - 2.0).abs() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeWindow {
    start: Epoch,
    end: Epoch,
}

impl TimeWindow {
    /// From `start` to `end`.
    ///
    /// # Errors
    ///
    /// [`VetError::WindowReversed`] if `end` is before `start`.
    pub fn new(start: Epoch, end: Epoch) -> Result<Self, VetError> {
        if end < start {
            return Err(VetError::WindowReversed);
        }
        Ok(Self { start, end })
    }

    /// `half` either side of `time`.
    ///
    /// # Errors
    ///
    /// [`VetError::WindowReversed`] if `half` is below zero.
    pub fn around(time: Epoch, half: TimeDelta) -> Result<Self, VetError> {
        Self::new(time - half, time + half)
    }

    /// The start.
    #[must_use]
    pub const fn start(&self) -> Epoch {
        self.start
    }

    /// The end.
    #[must_use]
    pub const fn end(&self) -> Epoch {
        self.end
    }
}

/// A predicted transit that overlaps one or more [`TimeWindow`]s.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TimeFlag {
    /// Its number, as in [`Transit::epoch`](crate::Transit::epoch).
    pub epoch: i64,
    /// Its predicted mid-transit time.
    pub time: Epoch,
    /// The windows it overlaps, their indices in the list given.
    pub windows: Vec<usize>,
}

/// Every transit of `candidate` predicted in `lc`'s span (as
/// [`evidence`](crate::evidence)'s) that overlaps a window, in order. A
/// transit lasts from half a duration before its mid-transit time to half
/// a duration after, and overlaps a window that starts before it ends and
/// ends after it starts.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, TimeDelta, TimeWindow, flag_times};
///
/// let time: Vec<f64> = (0..480).map(|i| f64::from(i) / 48.0).collect();
/// let lc = LightCurve::from_days(time, vec![1.0; 480], vec![1e-3; 480])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_days(1.0)?,
///     TimeDelta::from_hours(3.0)?,
/// )?;
/// // An event at 7.05: the transit at 7.0 overlaps it.
/// let event = TimeWindow::around(Epoch::from_days(7.05)?, TimeDelta::from_hours(0.5)?)?;
/// let flags = flag_times(&lc, &candidate, &[event]);
/// assert_eq!(flags.len(), 1);
/// assert_eq!((flags[0].epoch, flags[0].windows.as_slice()), (2, &[0][..]));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn flag_times(lc: &LightCurve, candidate: &Candidate, windows: &[TimeWindow]) -> Vec<TimeFlag> {
    let times: Vec<f64> = lc.time().iter().map(|t| t.as_days()).collect();
    let period = candidate.period();
    let half = candidate.duration().as_days() / 2.0;
    layout(&times, candidate)
        .predicted
        .into_iter()
        .filter_map(|n| {
            #[expect(clippy::cast_precision_loss, reason = "a transit number")]
            let centre = candidate.transit_time().as_days() + n as f64 * period.as_days();
            let (from, to) = (centre - half, centre + half);
            let hit: Vec<usize> = windows
                .iter()
                .enumerate()
                .filter(|(_, w)| w.start.as_days() < to && w.end.as_days() > from)
                .map(|(k, _)| k)
                .collect();
            (!hit.is_empty()).then(|| TimeFlag {
                epoch: n,
                time: Epoch::from_days(centre).expect("finite"),
                windows: hit,
            })
        })
        .collect()
}
