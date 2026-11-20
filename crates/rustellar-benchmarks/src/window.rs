//! How long a detrending window should be, measured: limb-darkened transits
//! on spotted, rotating stars, each light curve detrended with the biweight
//! at several multiples of its transit's duration, and at the fixed 0.5 days.
//!
//! A window too short follows the transit and takes some of its depth; one
//! too long leaves the star's variability in. Both cost signal to noise:
//! the depth the detrending keeps over the scatter it leaves, here as a
//! matched filter with the injected transit's own shape, against the same
//! transit in white noise alone.

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::thread;

use rustellar_bls::{Bls, Durations, Objective, SearchOptions};
use rustellar_core::{Epoch, LightCurve, OfirPeriod, TimeDelta};
use rustellar_detrend::{Candidate, Detrend, Flattened, transit_mask};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

use crate::random::Random;
use crate::stats::median;

/// The white noise of every curve.
const NOISE: f64 = 1e-3;

/// One injected curve: its transit and its star's variability.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub seed: u64,
    pub transit: TransitParams,
    /// Each sinusoid's period (days), amplitude and phase.
    pub waves: [(f64, f64, f64); 2],
}

/// One window tried: a multiple of the transit's duration (`None` for the
/// fixed 0.5 days), and whether the transits are masked (over 1.2 times
/// their duration, as for a candidate whose ephemeris is a little off).
#[derive(Debug, Clone, Copy)]
pub struct Window {
    pub factor: Option<f64>,
    pub masked: bool,
}

const fn blind(factor: f64) -> Window {
    Window {
        factor: Some(factor),
        masked: false,
    }
}

const fn masked(factor: f64) -> Window {
    Window {
        factor: Some(factor),
        masked: true,
    }
}

/// The windows tried.
pub const WINDOWS: [Window; 19] = [
    blind(1.5),
    blind(2.0),
    blind(2.5),
    blind(3.0),
    blind(4.0),
    blind(6.0),
    blind(8.0),
    blind(10.0),
    blind(12.0),
    Window {
        factor: None,
        masked: false,
    },
    masked(1.5),
    masked(2.0),
    masked(2.5),
    masked(3.0),
    masked(4.0),
    masked(6.0),
    masked(8.0),
    masked(12.0),
    Window {
        factor: None,
        masked: true,
    },
];

/// `count` cases from seeds 1, 2, ...: period 1 to 9 days, radius ratio
/// 0.03 to 0.06, impact parameter 0 to 0.7, duration 1 to 7 hours; two
/// sinusoids of period 0.5 to 10 days and amplitude 1e-4 to 5e-3 (both
/// spread evenly in their logarithms).
///
/// # Panics
///
/// If a drawn case is not a transit (it always is).
#[must_use]
pub fn cases(count: u64) -> Vec<Case> {
    (1..=count)
        .map(|seed| {
            let mut random = Random::new(0x5eed_d200 + seed);
            let mut between = |lo: f64, hi: f64| random.uniform().mul_add(hi - lo, lo);
            let period = between(1.0, 9.0);
            let k = between(0.03, 0.06);
            let b = between(0.0, 0.7);
            let t14 = between(1.0, 7.0) / 24.0;
            let t0 = between(0.0, period);
            let chord = ((1.0 + k) * (1.0 + k) - b * b).sqrt();
            let a_rs = chord / (PI * t14 / period).sin();
            let mut wave = || {
                let p = (between(0.5f64.ln(), 10f64.ln())).exp();
                let a = (between(1e-4f64.ln(), 5e-3f64.ln())).exp();
                (p, a, between(0.0, 2.0 * PI))
            };
            let waves = [wave(), wave()];
            let transit = TransitParams::new(
                Epoch::from_btjd(t0).expect("finite"),
                TimeDelta::from_days(period).expect("finite"),
                RadiusRatio::new(k).expect("a radius ratio"),
                ScaledSemiMajorAxis::new(a_rs).expect("above 1"),
                ImpactParameter::new(b).expect("below a"),
                LimbDarkening::quadratic(0.4804, 0.1867).expect("valid"),
            )
            .expect("a transit");
            Case {
                seed,
                transit,
                waves,
            }
        })
        .collect()
}

/// A TESS sector at 10 minutes: two 13.2-day orbits a day apart.
fn times() -> Vec<f64> {
    let step = 10.0 / 1440.0;
    (0..1901)
        .map(|i| f64::from(i) * step)
        .chain((0..1901).map(|i| 14.2 + f64::from(i) * step))
        .collect()
}

/// For one case and one window: the depth kept (1 is all of it) and the
/// signal to noise over the ideal's.
#[derive(Debug, Clone, Copy)]
pub struct Kept {
    pub depth: f64,
    pub snr: f64,
    /// The signal to noise of the transit in white noise alone.
    pub ideal: f64,
}

/// The curve for `case`, and its transit's dimming at each time (the
/// matched filter's shape).
pub(crate) fn inject(case: &Case) -> (LightCurve, Vec<f64>) {
    inject_with(case, 0.0, true)
}

/// [`inject`] with the noise correlated from one cadence to the next by
/// `red` (an AR(1) process of the same standard deviation; 0 is white),
/// and the transit left out unless `transit`.
pub(crate) fn inject_with(case: &Case, red: f64, transit: bool) -> (LightCurve, Vec<f64>) {
    let times = times();
    let mut random = Random::new(case.seed);
    let epochs: Vec<Epoch> = times
        .iter()
        .map(|&t| Epoch::from_btjd(t).expect("finite"))
        .collect();
    let model = if transit {
        case.transit.light_curve(&epochs)
    } else {
        vec![1.0; times.len()]
    };
    let shape: Vec<f64> = model.iter().map(|f| 1.0 - f).collect();
    let kick = (1.0 - red * red).sqrt();
    let mut noise = 0.0;
    let flux: Vec<f64> = times
        .iter()
        .zip(&model)
        .enumerate()
        .map(|(i, (&t, &m))| {
            let star: f64 = case
                .waves
                .iter()
                .map(|&(p, a, phase)| a * (2.0 * PI * t / p + phase).sin())
                .sum();
            let draw = random.noise();
            noise = if i == 0 {
                draw
            } else {
                red * noise + kick * draw
            };
            m * (1.0 + star) + NOISE * noise
        })
        .collect();
    let lc = LightCurve::from_btjd(times.clone(), flux, vec![NOISE; times.len()])
        .expect("a light curve");
    (lc, shape)
}

/// `lc` detrended with window `w`, for a transit `t14` days long.
fn flatten(lc: &LightCurve, w: Window, t14: f64, mask: &[bool]) -> Flattened {
    let window = w.factor.map_or(0.5, |f| f * t14);
    Detrend::default()
        .with_window(TimeDelta::from_days(window).expect("finite"))
        .with_threads(NonZeroUsize::MIN)
        .flatten(lc, w.masked.then_some(mask))
        .expect("detrended")
}

/// Each window's result for `case`, in the order of [`WINDOWS`].
///
/// # Panics
///
/// If the curve cannot be built or detrended (it always can).
#[must_use]
pub fn measure(case: &Case) -> Vec<Kept> {
    let (lc, shape) = inject(case);
    let ideal = ideal(&shape);
    let t14 = case.transit.t14().expect("a transit").as_days();
    let mask = transit_mask(
        &lc,
        case.transit.period(),
        case.transit.t0(),
        TimeDelta::from_days(1.2 * t14).expect("finite"),
    )
    .expect("a mask");
    WINDOWS
        .iter()
        .map(|&w| kept(&flatten(&lc, w, t14, &mask), &shape, ideal))
        .collect()
}

/// The depth `flat` keeps of the transit `shape` (by a matched filter) and
/// its signal to noise over `ideal`, the scatter from the points out of
/// transit.
pub(crate) fn kept(flat: &Flattened, shape: &[f64], ideal: f64) -> Kept {
    let (mut rs, mut ss) = (0.0, 0.0);
    let mut outside = vec![];
    for (&i, &f) in flat.indices().iter().zip(flat.light_curve().flux()) {
        let s = shape[i];
        if s > 0.0 {
            rs += (1.0 - f) * s;
            ss += s * s;
        } else {
            outside.push(f);
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let scatter = {
        let mean = outside.iter().sum::<f64>() / outside.len() as f64;
        (outside.iter().map(|f| (f - mean) * (f - mean)).sum::<f64>() / (outside.len() - 1) as f64)
            .sqrt()
    };
    let depth = rs / ss;
    Kept {
        depth,
        snr: depth * ss.sqrt() / scatter / ideal,
        ideal,
    }
}

/// The ideal signal to noise of `shape` in the white noise alone.
pub(crate) fn ideal(shape: &[f64]) -> f64 {
    shape.iter().map(|s| s * s).sum::<f64>().sqrt() / NOISE
}

/// [`measure`] for every case, on `threads` threads.
#[must_use]
pub fn measure_all(cases: &[Case], threads: NonZeroUsize) -> Vec<Vec<Kept>> {
    let chunk = cases.len().div_ceil(threads.get()).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(measure).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a study thread panicked"))
            .collect()
    })
}

/// The durations the blind search tries: 0.75 to 8 hours.
const DURATIONS: [f64; 8] = [0.03, 0.045, 0.065, 0.09, 0.125, 0.17, 0.23, 0.33];

/// For one case and one unmasked window: a blind BLS search's best peak.
#[derive(Debug, Clone, Copy)]
pub struct Found {
    pub sde: f64,
    /// Whether the best peak is at the injected period, within 1%.
    pub right: bool,
}

/// A blind BLS search (objective signal to noise, Ofir's grid from 0.6
/// days) on `case` detrended with each unmasked window, in the order of
/// [`WINDOWS`].
///
/// # Panics
///
/// If the curve cannot be built, detrended or searched (it always can).
#[must_use]
pub fn search(case: &Case) -> Vec<Found> {
    let (lc, _) = inject(case);
    let t14 = case.transit.t14().expect("a transit").as_days();
    let truth = case.transit.period().as_days();
    WINDOWS
        .iter()
        .filter(|w| !w.masked)
        .map(|&w| found(flatten(&lc, w, t14, &[]).light_curve(), truth))
        .collect()
}

/// A blind BLS search on `lc`: its best peak's SDE, and whether it is at
/// `truth` days within 1%.
pub(crate) fn found(lc: &LightCurve, truth: f64) -> Found {
    best(lc, truth).0
}

/// The blind BLS search of [`found`], with its best peak as a candidate.
pub(crate) fn best(lc: &LightCurve, truth: f64) -> (Found, Option<Candidate>) {
    best_with(lc, truth, None)
}

/// [`best`], or with `information` the matched search of a whitened light
/// curve (`Bls::search_whitened`).
pub(crate) fn best_with(
    lc: &LightCurve,
    truth: f64,
    information: Option<&dyn Fn(TimeDelta) -> f64>,
) -> (Found, Option<Candidate>) {
    let periods = OfirPeriod {
        minimum_period: Some(TimeDelta::from_days(0.6).expect("finite")),
        ..OfirPeriod::default()
    }
    .periods(lc)
    .expect("a grid");
    let bls = Bls::new(lc).expect("a light curve");
    let durations = Durations::from_days(&DURATIONS).expect("durations");
    let options = SearchOptions::default()
        .with_objective(Objective::Snr)
        .with_threads(NonZeroUsize::MIN);
    let result = match information {
        None => bls.search(&periods, &durations, &options),
        Some(f) => bls.search_whitened(&periods, &durations, &options, f),
    }
    .expect("a search");
    let candidate = result.best().map(|b| Candidate {
        period: b.transit.period(),
        t0: b.transit.transit_time(),
        duration: b.transit.duration(),
    });
    let period = candidate.map_or(f64::NAN, |c| c.period.as_days());
    let found = Found {
        sde: sde(result.power()),
        right: (period / truth - 1.0).abs() < 0.01,
    };
    (found, candidate)
}

/// The SDE of a BLS power spectrum: the peak's height above the mean, in
/// standard deviations.
fn sde(power: &[f64]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "a count of periods")]
    let n = power.len() as f64;
    let mean = power.iter().sum::<f64>() / n;
    let variance = power.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / n;
    let peak = power.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (peak - mean) / variance.sqrt()
}

/// [`search`] for every case, on `threads` threads.
#[must_use]
pub fn search_all(cases: &[Case], threads: NonZeroUsize) -> Vec<Vec<Found>> {
    let chunk = cases.len().div_ceil(threads.get()).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(search).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a search thread panicked"))
            .collect()
    })
}

/// The share of `cases` that `keep` picks, whose `found` passes `pass`.
pub(crate) fn share(
    cases: &[Case],
    found: &[Found],
    keep: impl Fn(&Case) -> bool,
    pass: impl Fn(&Found) -> bool,
) -> f64 {
    let (mut hits, mut all) = (0_u32, 0_u32);
    for (c, f) in cases.iter().zip(found) {
        if keep(c) {
            all += 1;
            hits += u32::from(pass(f));
        }
    }
    f64::from(hits) / f64::from(all)
}

/// The blind search's table: per unmasked window, how often the best peak
/// is at the injected period, how often with SDE above 7 too (for all
/// stars, then quiet and active ones apart), and the median SDE.
#[must_use]
pub fn search_report(cases: &[Case], results: &[Vec<Found>]) -> String {
    let mut out = String::from(
        "| Window | Best peak right | Right, SDE > 7 | Quiet stars: right, SDE > 7 \
         | Active stars: right, SDE > 7 | Median SDE |\n\
         |---|--:|--:|--:|--:|--:|\n",
    );
    let detected = |f: &Found| f.right && f.sde > 7.0;
    for (j, w) in WINDOWS.iter().filter(|w| !w.masked).enumerate() {
        let found: Vec<Found> = results.iter().map(|r| r[j]).collect();
        let sdes: Vec<f64> = found.iter().map(|f| f.sde).collect();
        let name = w
            .factor
            .map_or_else(|| "0.5 d".to_owned(), |f| format!("{f} x T14"));
        out += &format!(
            "| {name} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} |\n",
            share(cases, &found, |_| true, |f| f.right),
            share(cases, &found, |_| true, detected),
            share(cases, &found, |c| !active(c), detected),
            share(cases, &found, active, detected),
            median(&sdes),
        );
    }
    out
}

/// The `q` quantile of `values` (sorted copy, nearest rank); NaN for none.
pub(crate) fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "an index into a short list"
    )]
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

/// Whether a case's star is active: either sinusoid above 1e-3.
pub(crate) fn active(case: &Case) -> bool {
    case.waves.iter().any(|w| w.1 > 1e-3)
}

/// The table: per window, the median depth kept and the signal to noise
/// over the ideal's (median, tenth percentile), for all stars and then the
/// quiet and active ones apart.
#[must_use]
pub fn report(cases: &[Case], results: &[Vec<Kept>]) -> String {
    let quiet = cases.iter().filter(|c| !active(c)).count();
    let ideal: Vec<f64> = results.iter().map(|r| r[0].ideal).collect();
    let mut out = format!(
        "{} curves, {quiet} on quiet stars (both sinusoids at most 1e-3), {} on active ones; \
         the transits' signal to noise in white noise alone {:.1} to {:.1} (median {:.1}, \
         10th and 90th percentiles {:.1} and {:.1}).\n\n\
         | Window | Masked | Depth kept | SNR / ideal | SNR / ideal, 10th percentile \
         | Quiet stars: SNR / ideal | Active stars: SNR / ideal |\n\
         |---|---|--:|--:|--:|--:|--:|\n",
        cases.len(),
        cases.len() - quiet,
        quantile(&ideal, 0.0),
        quantile(&ideal, 1.0),
        median(&ideal),
        quantile(&ideal, 0.1),
        quantile(&ideal, 0.9),
    );
    for (j, w) in WINDOWS.iter().enumerate() {
        let column = |keep: &dyn Fn(&Case) -> bool, value: fn(&Kept) -> f64| -> Vec<f64> {
            cases
                .iter()
                .zip(results)
                .filter(|(c, _)| keep(c))
                .map(|(_, r)| value(&r[j]))
                .collect()
        };
        let all = |_: &Case| true;
        let depth = column(&all, |k| k.depth);
        let snr = column(&all, |k| k.snr);
        let snr_quiet = column(&|c| !active(c), |k| k.snr);
        let snr_active = column(&active, |k| k.snr);
        let name = w
            .factor
            .map_or_else(|| "0.5 d".to_owned(), |f| format!("{f} x T14"));
        out += &format!(
            "| {name} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            if w.masked { "yes" } else { "no" },
            median(&depth),
            median(&snr),
            quantile(&snr, 0.1),
            group_median(&snr_quiet),
            group_median(&snr_active),
        );
    }
    out
}

/// The median of a group of stars, NaN if there are none.
pub(crate) fn group_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        median(values)
    }
}
