use std::fmt;

use rustellar_core::{LightCurve, TimeDelta};

use crate::fit::phase;
use crate::grouped::solve;
use crate::{
    Candidate, Evidence, Fit, Shape, SignalToNoise, VetError, evidence, fit_depth, signal_to_noise,
};

/// A ratio of periods, `p / q` in lowest terms: an alias of a candidate's
/// period `P` is `P p / q`.
///
/// # Examples
///
/// ```
/// use rustellar_vet::Ratio;
///
/// let r = Ratio::new(4, 6)?;
/// assert_eq!((r.numerator(), r.denominator()), (2, 3));
/// assert_eq!(r.to_string(), "2/3");
/// assert_eq!(Ratio::new(2, 1)?.to_string(), "2");
/// # Ok::<(), rustellar_vet::VetError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio {
    p: u16,
    q: u16,
}

impl Ratio {
    /// The ratios looked at by default: a third, a half, two thirds, one,
    /// three halves, two and three times the period.
    pub const DEFAULT: [Self; 7] = [
        Self { p: 1, q: 3 },
        Self { p: 1, q: 2 },
        Self { p: 2, q: 3 },
        Self { p: 1, q: 1 },
        Self { p: 3, q: 2 },
        Self { p: 2, q: 1 },
        Self { p: 3, q: 1 },
    ];

    /// `p / q`, in lowest terms.
    ///
    /// # Errors
    ///
    /// [`VetError::RatioZero`] if `p` or `q` is zero.
    pub fn new(p: u16, q: u16) -> Result<Self, VetError> {
        if p == 0 || q == 0 {
            return Err(VetError::RatioZero);
        }
        let (mut a, mut b) = (p, q);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Ok(Self { p: p / a, q: q / a })
    }

    /// `p`.
    #[must_use]
    pub const fn numerator(self) -> u16 {
        self.p
    }

    /// `q`.
    #[must_use]
    pub const fn denominator(self) -> u16 {
        self.q
    }

    /// `p / q`.
    #[must_use]
    pub fn value(self) -> f64 {
        f64::from(self.p) / f64::from(self.q)
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = if self.q == 1 {
            self.p.to_string()
        } else {
            format!("{}/{}", self.p, self.q)
        };
        f.pad(&text)
    }
}

/// One depth fitted to a set of transits, with its error and points.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Depth {
    /// The fractional depth at mid-transit.
    pub depth: f64,
    /// Its standard error.
    pub depth_err: f64,
    /// The points in transit.
    pub points: usize,
}

impl Depth {
    /// How many standard errors the depth is from zero.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.depth / self.depth_err
    }
}

/// The candidate's and an alias's transits split into those both predict,
/// those only one predicts and the times between, each set's depth fitted
/// with one shared baseline (`None` where a set has no points).
///
/// The test of an alias is here: if the alias is the true period, its own
/// transits have the depth and the candidate's alone have none; if the
/// candidate is, the other way round.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Split {
    /// The transits both periods predict.
    pub both: Option<Depth>,
    /// The candidate's transits the alias does not predict.
    pub candidate_only: Option<Depth>,
    /// The alias's transits the candidate does not predict.
    pub alias_only: Option<Depth>,
    /// The times at the same spacing that neither predicts (for a ratio
    /// `p / q` with `p` and `q` both above 1).
    pub neither: Option<Depth>,
}

/// A candidate's alias: its period's fit, evidence and signal to noise, and
/// how it compares with the candidate's.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Alias {
    /// The alias: the candidate's mid-transit time and duration at the
    /// ratio's period.
    pub candidate: Candidate,
    /// Its fit.
    pub fit: Fit,
    /// Its evidence transit by transit.
    pub evidence: Evidence,
    /// Its signal to noise.
    pub snr: SignalToNoise,
    /// Its fit's chi-squared less the candidate's: below zero when the
    /// alias fits the light curve better (both have one depth and one
    /// baseline).
    pub delta_chi2: f64,
    /// Both periods' transits split; `None` where the transits at the finer
    /// spacing would overlap (a duration longer than `P / q`) or a set
    /// cannot be fitted.
    pub split: Option<Split>,
}

/// One row of an [`Aliases`] table: a ratio and its alias, or why it could
/// not be measured.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AliasRow {
    /// The ratio.
    pub ratio: Ratio,
    /// The alias at that ratio, or why there is none.
    pub alias: Result<Alias, VetError>,
}

/// A candidate's aliases side by side: which period the light curve
/// supports.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Aliases {
    /// The candidate's own fit.
    pub fit: Fit,
    /// Each ratio's row, in the order asked for.
    pub rows: Vec<AliasRow>,
}

impl Aliases {
    /// The row whose alias fits best: the lowest chi-squared.
    #[must_use]
    pub fn best(&self) -> Option<&AliasRow> {
        self.rows
            .iter()
            .filter_map(|r| r.alias.as_ref().ok().map(|a| (r, a.delta_chi2)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(r, _)| r)
    }
}

/// The candidate at each of [`Ratio::DEFAULT`]'s periods.
///
/// # Errors
///
/// As [`fit_depth`] for the candidate itself; each alias's own errors are
/// in its row.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, Ratio, Shape, TimeDelta, aliases};
///
/// // A 0.3% box dip every 1.5 days, in alternating noise of 1e-3, found
/// // at 3 days.
/// let time: Vec<f64> = (0..1728).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .enumerate()
///     .map(|(i, &t)| {
///         let noise = if i % 2 == 0 { 1e-3 } else { -1e-3 };
///         let dip = ((t - 0.75 + 0.75).rem_euclid(1.5) - 0.75).abs() < 0.05;
///         1.0 + noise - if dip { 3e-3 } else { 0.0 }
///     })
///     .collect();
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 1728])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_days(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
/// let table = aliases(&lc, &candidate, Shape::BOX)?;
/// assert_eq!(table.best().unwrap().ratio, Ratio::new(1, 2)?);
/// println!("{table}");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn aliases(lc: &LightCurve, candidate: &Candidate, shape: Shape) -> Result<Aliases, VetError> {
    aliases_at(lc, candidate, shape, &Ratio::DEFAULT)
}

/// The candidate at each of `ratios`' periods.
///
/// # Errors
///
/// As [`aliases`].
pub fn aliases_at(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
    ratios: &[Ratio],
) -> Result<Aliases, VetError> {
    let fit = fit_depth(lc, candidate, shape)?;
    let rows = ratios
        .iter()
        .map(|&ratio| AliasRow {
            ratio,
            alias: alias(lc, candidate, shape, ratio, fit.chi2),
        })
        .collect();
    Ok(Aliases { fit, rows })
}

/// The candidate at `ratio`'s period, compared with its chi-squared `chi2`.
fn alias(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
    ratio: Ratio,
    chi2: f64,
) -> Result<Alias, VetError> {
    let period = TimeDelta::from_days(candidate.period().as_days() * ratio.value())
        .map_err(|_| VetError::AliasPeriodNotFinite)?;
    let at = Candidate::new(period, candidate.transit_time(), candidate.duration())?;
    let fit = fit_depth(lc, &at, shape)?;
    let evidence = evidence(lc, &at, shape)?;
    let snr = signal_to_noise(lc, &at, shape)?;
    Ok(Alias {
        candidate: at,
        delta_chi2: fit.chi2 - chi2,
        fit,
        evidence,
        snr,
        split: split(lc, candidate, shape, ratio),
    })
}

/// The candidate's and `ratio`'s transits split, at the finer spacing `P /
/// q` where both fall on whole numbers of it: number `m` is the
/// candidate's when `q` divides it, the alias's when `p` does.
pub(crate) fn split(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
    ratio: Ratio,
) -> Option<Split> {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let fine = period / f64::from(ratio.q);
    if duration >= fine {
        return None;
    }
    let profile = shape.profile(period, duration).ok()?;
    let t0 = candidate.transit_time().as_days();
    let (p, q) = (i64::from(ratio.p), i64::from(ratio.q));
    let (mut shapes, mut group) = (vec![], vec![]);
    for t in lc.time() {
        let dt = t.as_days() - t0;
        let ph = phase(dt, fine);
        #[expect(clippy::cast_possible_truncation, reason = "a whole number of steps")]
        let m = ((dt - ph) / fine).round() as i64;
        let (theirs, ours) = (m.rem_euclid(p) == 0, m.rem_euclid(q) == 0);
        shapes.push(profile.at(ph));
        group.push(Some(match (ours, theirs) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        }));
    }
    let fit = solve(lc.flux(), lc.flux_err(), &shapes, &group, 4).ok()?;
    let depth = |k: usize| {
        Some(Depth {
            depth: fit.depth(k)?,
            depth_err: fit.cov_depth(k, k)?.sqrt(),
            points: fit.points[k],
        })
    };
    Some(Split {
        both: depth(0),
        candidate_only: depth(1),
        alias_only: depth(2),
        neither: depth(3),
    })
}

/// A depth in parts per million with its error, or a dash.
fn ppm(d: Option<&Depth>) -> String {
    d.map_or_else(
        || "-".to_owned(),
        |d| format!("{:.0} ± {:.0}", d.depth * 1e6, d.depth_err * 1e6),
    )
}

impl fmt::Display for Aliases {
    /// The table: each ratio's period, depth (ppm), signal to noise white
    /// and red, transits with data of those predicted, odd-even
    /// significance, depths at the alias's and the candidate's own
    /// transits, and chi-squared against the candidate's.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header = format!(
            "{:>5}  {:>11}  {:>15}  {:>6}  {:>6}  {:>8}  {:>8}  {:>15}  {:>15}  {:>10}",
            "ratio",
            "period (d)",
            "depth (ppm)",
            "SNR",
            "red",
            "transits",
            "odd-even",
            "alias only",
            "candidate only",
            "delta chi2"
        );
        writeln!(f, "{header}")?;
        for row in &self.rows {
            let a = match &row.alias {
                Ok(a) => a,
                Err(e) => {
                    writeln!(f, "{:>5}  {e}", row.ratio)?;
                    continue;
                }
            };
            let fit = a.fit;
            let depth = format!("{:.0} ± {:.0}", fit.depth * 1e6, fit.depth_err * 1e6);
            let transits = format!("{}/{}", a.evidence.with_data(), a.evidence.transits.len());
            let odd_even = a
                .evidence
                .odd_even
                .map_or_else(|| "-".to_owned(), |o| format!("{:+.1}", o.sigma()));
            let split = a.split.as_ref();
            let line = format!(
                "{:>5}  {:>11.6}  {:>15}  {:>6.1}  {:>6.1}  {:>8}  {:>8}  {:>15}  {:>15}  {:>10.1}",
                row.ratio,
                a.candidate.period().as_days(),
                depth,
                a.snr.white,
                a.snr.red,
                transits,
                odd_even,
                ppm(split.and_then(|s| s.alias_only.as_ref())),
                ppm(split.and_then(|s| s.candidate_only.as_ref())),
                a.delta_chi2
            );
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}
