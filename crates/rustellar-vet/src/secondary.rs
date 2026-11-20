use rustellar_core::LightCurve;

use crate::fit::phase;
use crate::sum::Sum;
use crate::{Candidate, Shape, fit_depth};

/// A box of the candidate's duration fitted at one phase.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Dip {
    /// Where, as a fraction of the period after mid-transit.
    pub phase: f64,
    /// The fractional depth: 1 less the box's mean over the baseline's.
    /// Negative for a brightening.
    pub depth: f64,
    /// Its standard error.
    pub depth_err: f64,
    /// The points in the box.
    pub points: usize,
}

impl Dip {
    /// How many standard errors the depth is from zero.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.depth / self.depth_err
    }

    /// The largest depth consistent with the data at `sigmas` standard
    /// errors: the depth (or zero, if negative) plus `sigmas` errors. The
    /// usual limit is at 3.
    #[must_use]
    pub fn upper_limit(&self, sigmas: f64) -> f64 {
        self.depth.max(0.0) + sigmas * self.depth_err
    }
}

/// A search for a second eclipse away from the transit: a box of the
/// candidate's duration slid over every phase at least one duration from
/// mid-transit, in steps of a quarter duration, with the points within one
/// duration of mid-transit left out.
///
/// An eclipsing binary's second star shows a second dip, at half the
/// period for a circular orbit; a planet's own eclipse behind its star is
/// far shallower. Each `None` is for a period too short to leave room (under
/// two durations), or no points there.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Secondary {
    /// The box at mid-transit, every point used: the primary eclipse's
    /// depth on the same terms, for [`depth_ratio`](Self::depth_ratio).
    pub primary: Option<Dip>,
    /// The box at half the period, where a circular orbit's second eclipse
    /// falls.
    pub at_half: Option<Dip>,
    /// The deepest dip of the scan, in standard errors.
    pub deepest: Option<Dip>,
    /// The strongest brightening of the scan, in standard errors: a scale
    /// for how deep the deepest dip would be by chance or by systematics.
    pub brightest: Option<Dip>,
    /// The spread (standard deviation) of every scanned box's significance:
    /// near 1 in white noise, larger in red noise or systematics. The
    /// deepest dip over this is its significance against the light curve's
    /// own noise.
    pub spread: f64,
    /// About how many independent boxes the scan holds: its length over the
    /// duration. Many trials make a 3-sigma dip somewhere likely.
    pub trials: usize,
}

impl Secondary {
    /// The depth at half the period over the primary's: near 1 when the
    /// two eclipses are alike, so the true period may be half the
    /// candidate's (or the star a twin binary); near 0 for a planet.
    #[must_use]
    pub fn depth_ratio(&self) -> Option<f64> {
        Some(self.at_half?.depth / self.primary?.depth)
    }
}

/// The search for a second eclipse of `candidate` in `lc`.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, TimeDelta, secondary};
///
/// // A 0.3% dip every 3 days, and a 0.1% one half a period later.
/// let time: Vec<f64> = (0..1728).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .enumerate()
///     .map(|(i, &t)| {
///         let noise = if i % 2 == 0 { 1e-3 } else { -1e-3 };
///         let at = |c: f64| ((t - c + 1.5).rem_euclid(3.0) - 1.5).abs() < 0.05;
///         let dip = if at(0.75) { 3e-3 } else if at(2.25) { 1e-3 } else { 0.0 };
///         1.0 + noise - dip
///     })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 1728])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_btjd(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
/// let s = secondary(&lc, &candidate);
/// let half = s.at_half.unwrap();
/// assert!(half.sigma() > 5.0);
/// assert!((s.deepest.unwrap().phase - 0.5).abs() < 0.02);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn secondary(lc: &LightCurve, candidate: &Candidate) -> Secondary {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let t0 = candidate.transit_time().as_btjd();
    // Every point at least a duration from mid-transit: its phase in days
    // after it (from `duration` to `period - duration`), weight and flux.
    let mut points: Vec<(f64, f64, f64)> = lc
        .time()
        .iter()
        .zip(lc.flux())
        .zip(lc.flux_err())
        .filter_map(|((t, &y), &e)| {
            let ph = phase(t.as_btjd() - t0, period);
            (ph.abs() >= duration).then(|| (ph.rem_euclid(period), 1.0 / (e * e), y))
        })
        .collect();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    let primary = fit_depth(lc, candidate, Shape::BOX).ok().map(|f| Dip {
        phase: 0.0,
        depth: f.depth,
        depth_err: f.depth_err,
        points: f.points_in_transit,
    });
    let none = Secondary {
        primary,
        at_half: None,
        deepest: None,
        brightest: None,
        spread: f64::NAN,
        trials: 0,
    };
    let room = period - 2.0 * duration;
    if room < 0.0 || points.is_empty() {
        return none;
    }
    // The weighted mean, and the sums of weight and of weighted flux about
    // it.
    let (mut w, mut wy) = (Sum::default(), Sum::default());
    for &(_, wi, y) in &points {
        w.add(wi);
        wy.add(wi * y);
    }
    let mean = wy.get() / w.get();
    let centred: Vec<(f64, f64, f64)> =
        points.iter().map(|&(u, wi, y)| (u, wi, y - mean)).collect();
    let mut total_c = Sum::default();
    for &(_, wi, c) in &centred {
        total_c.add(wi * c);
    }
    let (total_w, total_c) = (w.get(), total_c.get());
    let dip = |centre: f64| -> Option<Dip> {
        let half = 0.5 * duration;
        let lo = centred.partition_point(|p| p.0 <= centre - half);
        let hi = centred.partition_point(|p| p.0 < centre + half);
        let (mut w_in, mut c_in) = (Sum::default(), Sum::default());
        for &(_, wi, c) in &centred[lo..hi] {
            w_in.add(wi);
            c_in.add(wi * c);
        }
        let (w_in, c_in) = (w_in.get(), c_in.get());
        let w_out = total_w - w_in;
        if hi == lo || w_out <= 0.0 {
            return None;
        }
        // The baseline (outside the box) and the box's mean, about the
        // weighted mean.
        let (alpha_c, m_c) = ((total_c - c_in) / w_out, c_in / w_in);
        let (alpha, m) = (mean + alpha_c, mean + m_c);
        let depth = (alpha_c - m_c) / alpha;
        let var = 1.0 / (w_in * alpha * alpha) + m * m / (w_out * alpha.powi(4));
        Some(Dip {
            phase: centre / period,
            depth,
            depth_err: var.sqrt(),
            points: hi - lo,
        })
    };
    let step = 0.25 * duration;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a count of steps, room not negative"
    )]
    let steps = (room / step).floor() as usize;
    let scan: Vec<Dip> = (0..=steps)
        .filter_map(|j| {
            #[expect(clippy::cast_precision_loss, reason = "a step number")]
            let centre = duration + j as f64 * step;
            dip(centre)
        })
        .collect();
    let sigmas: Vec<f64> = scan.iter().map(Dip::sigma).collect();
    let by_sigma = |a: &&Dip, b: &&Dip| a.sigma().total_cmp(&b.sigma());
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a count"
    )]
    let trials = (room / duration).floor() as usize + 1;
    Secondary {
        primary,
        at_half: dip(0.5 * period),
        deepest: scan.iter().max_by(by_sigma).copied(),
        brightest: scan.iter().min_by(by_sigma).copied(),
        spread: spread(&sigmas),
        trials,
    }
}

/// The sample standard deviation; NaN for fewer than two values.
fn spread(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}
