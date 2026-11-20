use rustellar_core::LightCurve;

use crate::evidence::{Layout, layout};
use crate::fit::shapes;
use crate::grouped::solve;
use crate::{Candidate, Shape, VetError};

/// The noise around a candidate's transits, measured out of transit in
/// fractions of the baseline: white, and red on the transit's own time
/// scale.
///
/// The red noise is Pont, Zucker & Queloz's (2006): the residuals out of
/// transit are averaged in bins as long as the transit, and the variance
/// of those averages beyond what white noise alone gives is the red
/// noise's, `sigma_red^2 = sigma_binned^2 - sigma_white^2 <1 / n_bin>`
/// (zero where white noise explains it all).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Noise {
    /// The scatter of single points out of transit.
    pub white: f64,
    /// The scatter of the residuals over their errors out of transit: 1
    /// when the errors given are right. NaN with fewer than two points out
    /// of transit.
    pub error_scale: f64,
    /// The scatter of the averages in bins one transit long.
    pub binned: f64,
    /// The red noise on the transit's time scale.
    pub red: f64,
    /// The bins used: those out of transit holding at least half the
    /// points a full bin would.
    pub bins: usize,
}

/// One transit's signal to noise.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TransitSnr {
    /// Its number, as in [`Transit::epoch`](crate::Transit::epoch).
    pub epoch: i64,
    /// Its depth over its error, with the errors given.
    pub white: f64,
    /// Its depth over its error with the noise measured, as
    /// [`SignalToNoise::red`]: one transit's depth carries all of the red
    /// noise.
    pub red: f64,
}

/// A candidate's signal to noise, white and red, transit by transit, and
/// whether its transits share one depth.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SignalToNoise {
    /// The depth, one for every transit, as [`Fit::depth`](crate::Fit::depth).
    pub depth: f64,
    /// The depth over its error, with the errors given (white noise).
    pub white: f64,
    /// The depth over its error with the noise measured, as Pont, Zucker &
    /// Queloz (2006): the white error scaled to the scatter out of transit
    /// ([`Noise::error_scale`]) where that is larger, and the red noise's
    /// variance over the number of transits added, `err^2 max(1, scale)^2 +
    /// sigma_red^2 / N_transits`. Never above [`white`](Self::white).
    pub red: f64,
    /// The noise it was measured with.
    pub noise: Noise,
    /// Each transit with points.
    pub transits: Vec<TransitSnr>,
    /// The largest single transit's share of the depth: each transit's depth
    /// weighted by its inverse variance, `d_k / e_k^2`, the largest over
    /// their sum. `1 / N` for `N` equal transits, near 1 when one event
    /// makes the signal (and above 1 when the others are negative).
    pub largest_share: f64,
    /// How much better each transit's own depth fits than one depth for all:
    /// the drop in chi-squared, distributed as chi-squared with
    /// [`consistency_dof`](Self::consistency_dof) degrees of freedom when the
    /// transits do share one depth and the errors are right.
    pub consistency_chi2: f64,
    /// The transits with points less one.
    pub consistency_dof: usize,
}

/// The signal to noise of `candidate` in `lc`, for `shape`.
///
/// # Errors
///
/// As [`fit_depth`](crate::fit_depth).
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, signal_to_noise};
///
/// // A 0.3% box dip every 3 days, in alternating noise of 1e-3.
/// let time: Vec<f64> = (0..1728).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .enumerate()
///     .map(|(i, &t)| {
///         let noise = if i % 2 == 0 { 1e-3 } else { -1e-3 };
///         let dip = ((t - 0.75 + 1.5).rem_euclid(3.0) - 1.5).abs() < 0.05;
///         1.0 + noise - if dip { 3e-3 } else { 0.0 }
///     })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 1728])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_btjd(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
/// let snr = signal_to_noise(&lc, &candidate, Shape::BOX)?;
/// assert!(snr.white > 20.0);
/// assert_eq!(snr.transits.len(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn signal_to_noise(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
) -> Result<SignalToNoise, VetError> {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let profile = shape.profile(period, duration)?;
    let s = shapes(lc, candidate, &profile);
    let times: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
    let Layout { epochs, predicted } = layout(&times, candidate);
    let (flux, err) = (lc.flux(), lc.flux_err());
    // One depth for all, then one for each transit.
    let all: Vec<Option<usize>> = vec![Some(0); s.len()];
    let one = solve(flux, err, &s, &all, 1)?;
    let index = |n: i64| predicted.binary_search(&n).ok();
    let group: Vec<Option<usize>> = epochs.iter().map(|&n| index(n)).collect();
    let each = solve(flux, err, &s, &group, predicted.len())?;
    let noise = measure(&times, flux, err, &s, one.alpha, duration);
    let used: Vec<usize> = (0..predicted.len())
        .filter(|&k| each.points[k] > 0)
        .collect();
    // A depth's variance with the noise measured, over `transits` transits:
    // the errors given at least (`max` passes over NaN, too few points to
    // measure).
    let scale = noise.error_scale.max(1.0).powi(2);
    let red_var = noise.red * noise.red;
    #[expect(clippy::cast_precision_loss, reason = "a count of transits")]
    let measured = |var: f64, transits: usize| var * scale + red_var / transits as f64;
    let depth = one.depth(0).expect("points in transit");
    let var = one.cov_depth(0, 0).expect("points in transit");
    let white = depth / var.sqrt();
    let red = depth / measured(var, used.len()).sqrt();
    let (mut weighted, mut largest) = (0.0, f64::NEG_INFINITY);
    let transits: Vec<TransitSnr> = used
        .iter()
        .map(|&k| {
            let d = each.depth(k).expect("points");
            let var = each.cov_depth(k, k).expect("points");
            weighted += d / var;
            largest = largest.max(d / var);
            TransitSnr {
                epoch: predicted[k],
                white: d / var.sqrt(),
                red: d / measured(var, 1).sqrt(),
            }
        })
        .collect();
    Ok(SignalToNoise {
        depth,
        white,
        red,
        noise,
        largest_share: largest / weighted,
        consistency_chi2: (one.chi2 - each.chi2).max(0.0),
        consistency_dof: used.len() - 1,
        transits,
    })
}

/// The white and red noise of the residuals out of transit, `flux /
/// baseline - 1`, in time bins one `duration` long from the first time.
pub(crate) fn measure(
    times: &[f64],
    flux: &[f64],
    flux_err: &[f64],
    shapes: &[f64],
    baseline: f64,
    duration: f64,
) -> Noise {
    // Each point out of transit: its time, residual and residual over its
    // error.
    let out: Vec<(f64, f64, f64)> = times
        .iter()
        .zip(flux)
        .zip(flux_err)
        .zip(shapes)
        .filter(|&(_, &s)| s == 0.0)
        .map(|(((&t, &y), &e), _)| (t, y / baseline - 1.0, (y - baseline) / e))
        .collect();
    let white = scatter(out.iter().map(|&(_, r, _)| r));
    let error_scale = scatter(out.iter().map(|&(_, _, z)| z));
    let first = out.first().map_or(0.0, |&(t, _, _)| t);
    // Bin by time; each bin's count and sum.
    let mut bins = std::collections::BTreeMap::<i64, (usize, f64)>::new();
    for &(t, r, _) in &out {
        #[expect(clippy::cast_possible_truncation, reason = "a bin number")]
        let b = ((t - first) / duration).floor() as i64;
        let e = bins.entry(b).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += r;
    }
    let full = bins.values().map(|&(n, _)| n).max().unwrap_or(0);
    let kept: Vec<(usize, f64)> = bins
        .values()
        .filter(|&&(n, _)| 2 * n >= full && n >= 2)
        .copied()
        .collect();
    let (binned, red) = if kept.len() < 2 {
        (f64::NAN, 0.0)
    } else {
        #[expect(clippy::cast_precision_loss, reason = "counts of points")]
        let binned = scatter(kept.iter().map(|&(n, sum)| sum / n as f64));
        #[expect(clippy::cast_precision_loss, reason = "counts of points")]
        let inverse_n = kept.iter().map(|&(n, _)| 1.0 / n as f64).sum::<f64>() / kept.len() as f64;
        let red_var = binned * binned - white * white * inverse_n;
        (binned, red_var.max(0.0).sqrt())
    };
    Noise {
        white,
        error_scale,
        binned,
        red,
        bins: kept.len(),
    }
}

/// The sample standard deviation (n - 1); NaN for fewer than two values.
fn scatter(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let (n, sum) = values
        .clone()
        .fold((0usize, 0.0), |(n, s), v| (n + 1, s + v));
    if n < 2 {
        return f64::NAN;
    }
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let n = n as f64;
    let mean = sum / n;
    let ss: f64 = values.map(|v| (v - mean) * (v - mean)).sum();
    (ss / (n - 1.0)).sqrt()
}
