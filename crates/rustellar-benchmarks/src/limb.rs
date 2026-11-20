//! `rustellar-bls` against `rustellar-tls` on limb-darkened transits: the
//! injections of `rustellar-bls`'s limb-darkened study, each search at its
//! own 1% false-alarm rate, with transitleastsquares 1.32 on its subset.
//!
//! The 2,000 curves: 1,000 transits (quadratic limb darkening u = 0.4,
//! 0.26, a Sun-like star, impact parameter uniform in [0, 1)) in the noise
//! of four survey families, and 1,000 of noise alone. A search detects a
//! transit when its SDE is above the level 1% of its own noise-only curves
//! reach, and recovers it when it detects it within 1% of its period.

use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::thread;

use crate::recipe::Survey;
use rustellar_bls::{Bls, Durations, Objective};
use rustellar_core::{Epoch, LightCurve, OfirPeriod, Periods, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};
use rustellar_tls::{SearchOptions, Tls};

use crate::stats::median;

const FAMILIES: &str = include_str!("../../rustellar-bls/tests/reference/limb_families.csv");
const CASES: &str = include_str!("../../rustellar-bls/tests/reference/limb_cases.csv");
const TLS: &str = include_str!("../../rustellar-bls/tests/reference/limb_tls.csv");

/// The study's box durations for BLS, in days.
const DURATIONS: [f64; 6] = [0.04, 0.06, 0.08, 0.11, 0.15, 0.2];

/// The families' names, in order.
const NAMES: [&str; 4] = [
    "TESS 30 min, 1 sector",
    "TESS 2 min, 10% kept",
    "TESS 30 min, 2 sectors",
    "ground, nights",
];

/// rustellar-model against batman on the reference data's sample points.
const MODEL_TOLERANCE: f64 = 2.1e-8;

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

/// A search's SDE, and its best candidate's period and mid-transit time
/// in days (NaN where it has none).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// The signal detection efficiency.
    pub sde: f64,
    /// The period.
    pub period: f64,
    /// A mid-transit time.
    pub t0: f64,
}

/// One curve of the study.
#[derive(Debug, Clone)]
pub struct Case {
    /// Its family: 0 to 3, in the order of the report's rows.
    pub family: usize,
    /// Whether it holds a transit.
    pub injected: bool,
    /// The light curve.
    pub lc: LightCurve,
    /// The injected period, in days.
    pub period: f64,
    /// The injected mid-transit time.
    pub tc: f64,
    /// The injected transit's signal to noise.
    pub snr: f64,
    /// The periods searched: Ofir's grid from 0.6 days.
    pub periods: Periods,
    /// transitleastsquares 1.32's result, on every fourth pair of groups.
    pub tls: Option<Hit>,
}

/// A family's segments (start, points), and its cadence, noise and the
/// fraction of points kept.
type Family = (Vec<(f64, u32)>, [f64; 3]);

/// The curves of every `step`-th block of eight (each family with a
/// transit, then without; 1 for all), built again from their seeds.
///
/// # Panics
///
/// If a curve is not the reference data's: its times and errors to the bit, its
/// fluxes within the transit model's agreement with batman.
#[must_use]
pub fn cases(step: usize) -> Vec<Case> {
    let families: Vec<Family> = rows(FAMILIES)
        .map(|v| {
            let segments = v[2]
                .split('|')
                .map(|s| {
                    let (start, n) = s.split_once(':').expect("start:count");
                    (start.parse().expect("a time"), n.parse().expect("a count"))
                })
                .collect();
            let f = |i: usize| v[i].parse::<f64>().expect("a number");
            (segments, [f(3), f(4), f(5)])
        })
        .collect();
    let tls: Vec<(usize, Hit)> = rows(TLS)
        .map(|v| {
            let f = |i: usize| v[i].parse::<f64>().expect("a number");
            let hit = Hit {
                sde: f(5),
                period: f(1),
                t0: f(2),
            };
            (v[0].parse().expect("a case"), hit)
        })
        .collect();
    rows(CASES)
        .enumerate()
        // Whole blocks of eight: each family with a transit, then without.
        .filter(|(i, _)| (i / 8) % step == 0)
        .map(|(index, v)| {
            let f = |i: usize| v[i].parse::<f64>().expect("a number");
            let family: usize = v[1].parse().expect("a family");
            let injected = v[3] == "1";
            let (period, tc) = (f(4), f(5));
            let (segments, [cadence, noise, keep]) = &families[family];
            let lc = Survey {
                seed: v[2].parse().expect("a seed"),
                segments: segments.clone(),
                cadence: *cadence,
                noise: *noise,
                keep: *keep,
                period: 1.0,
                tc: 0.0,
                duration: 0.0,
                depth: 0.0,
            }
            .light_curve();
            let lc = if injected {
                let transit = TransitParams::new(
                    Epoch::from_days(tc).expect("finite"),
                    TimeDelta::from_days(period).expect("finite"),
                    RadiusRatio::new(f(6)).expect("a radius ratio"),
                    ScaledSemiMajorAxis::new(f(7)).expect("a semi-major axis"),
                    ImpactParameter::new(f(8)).expect("an impact parameter"),
                    LimbDarkening::quadratic(0.4, 0.26).expect("the study's"),
                )
                .expect("a transit");
                let model = transit.light_curve(lc.time());
                let flux = lc.flux().iter().zip(&model).map(|(y, m)| y + (m - 1.0));
                LightCurve::from_days(
                    lc.time().iter().map(|t| t.as_days()).collect(),
                    flux.collect(),
                    lc.flux_err().to_vec(),
                )
                .expect("a light curve")
            } else {
                lc
            };
            check_samples(&lc, v[15]);
            let periods = OfirPeriod {
                minimum_period: Some(TimeDelta::from_days(0.6).expect("finite")),
                ..OfirPeriod::default()
            }
            .periods(&lc)
            .expect("a grid");
            assert_eq!(
                periods.as_slice().len(),
                v[10].parse::<usize>().expect("a count"),
                "the reference grid"
            );
            Case {
                family,
                injected,
                lc,
                period,
                tc,
                snr: f(9),
                periods,
                tls: tls.iter().find(|t| t.0 == index).map(|t| t.1),
            }
        })
        .collect()
}

/// The reference data's sample points: "index:time:flux:error;...".
fn check_samples(lc: &LightCurve, samples: &str) {
    for sample in samples.split(';') {
        let v: Vec<f64> = sample
            .split(':')
            .map(|x| x.parse().expect("a number"))
            .collect();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "an index"
        )]
        let i = v[0] as usize;
        assert_eq!(lc.time()[i].as_days().to_bits(), v[1].to_bits(), "time {i}");
        assert_eq!(lc.flux_err()[i].to_bits(), v[3].to_bits(), "error {i}");
        assert!((lc.flux()[i] - v[2]).abs() <= MODEL_TOLERANCE, "flux {i}");
    }
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

/// `rustellar-bls` as in its study (objective snr, the study's durations)
/// and `rustellar-tls` with `options`, on one thread each.
///
/// # Panics
///
/// If a search fails, which a study curve never makes it do.
#[must_use]
pub fn search(case: &Case, options: &SearchOptions) -> [Hit; 2] {
    let one = NonZeroUsize::MIN;
    let bls = Bls::new(&case.lc)
        .expect("a light curve")
        .search(
            &case.periods,
            &Durations::from_days(&DURATIONS).expect("durations"),
            &rustellar_bls::SearchOptions::default()
                .with_objective(Objective::Snr)
                .with_threads(one),
        )
        .expect("a search");
    let best = bls.best().expect("a box with a dip");
    let bls = Hit {
        sde: sde(bls.power()),
        period: best.transit.period().as_days(),
        t0: best.transit.transit_time().as_days(),
    };
    // The study's flux is around 0, which a box does not mind; TLS's
    // template dims from 1, so it gets the flux plus 1, as
    // transitleastsquares 1.32 did.
    let normalised = LightCurve::from_days(
        case.lc.time().iter().map(|t| t.as_days()).collect(),
        case.lc.flux().iter().map(|f| f + 1.0).collect(),
        case.lc.flux_err().to_vec(),
    )
    .expect("a light curve");
    let result = Tls::new(&normalised)
        .expect("a light curve")
        .search(&case.periods, &options.with_threads(one))
        .expect("a search");
    let tls = Hit {
        sde: result.spectra().map_or(0.0, |s| s.sde),
        period: result.best().map_or(f64::NAN, |b| b.period.as_days()),
        t0: result.best().map_or(f64::NAN, |b| b.transit_time.as_days()),
    };
    [bls, tls]
}

/// Both searches on every case, the cases shared out over `threads`.
#[must_use]
pub fn search_all(cases: &[Case], options: &SearchOptions, threads: NonZeroUsize) -> Vec<[Hit; 2]> {
    let chunk = cases.len().div_ceil(threads.get()).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || part.iter().map(|c| search(c, options)).collect::<Vec<_>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a search thread panicked"))
            .collect()
    })
}

/// The SDE a search must beat: above all but 1% of its noise-only curves.
fn threshold(noise: &[f64]) -> f64 {
    let mut sorted = noise.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[sorted.len() / 100]
}

/// The period's relative error and the mid-transit time's error in days (at
/// the transit nearest the middle of the data), if `hit` recovers the case.
fn errors(case: &Case, hit: Hit, threshold: f64) -> Option<[f64; 2]> {
    if hit.sde <= threshold || (hit.period / case.period - 1.0).abs() >= 0.01 {
        return None;
    }
    let time = case.lc.time();
    let middle = 0.5 * (time[0].as_days() + time[time.len() - 1].as_days());
    let near = hit.t0 + ((middle - hit.t0) / hit.period).round() * hit.period;
    let r = (near - case.tc) / case.period;
    Some([
        (hit.period / case.period - 1.0).abs(),
        (r - r.round()).abs() * case.period,
    ])
}

/// "recovered/of | median period error | median mid-transit time error".
fn tally(cases: &[(&Case, Hit)], threshold: f64) -> String {
    let found: Vec<[f64; 2]> = cases
        .iter()
        .filter_map(|(c, h)| errors(c, *h, threshold))
        .collect();
    if found.is_empty() {
        return format!("0/{} | | ", cases.len());
    }
    let p: Vec<f64> = found.iter().map(|e| e[0]).collect();
    let t: Vec<f64> = found.iter().map(|e| e[1] * 1440.0).collect();
    format!(
        "{}/{} | {:.2e} | {:.2}",
        found.len(),
        cases.len(),
        median(&p),
        median(&t)
    )
}

type Pick = Box<dyn Fn(&Case) -> bool>;

/// The study as Markdown tables: each search's threshold, and by family,
/// signal to noise and in all, how many transits each recovers with its
/// median errors (period relative, mid-transit time in minutes); then
/// transitleastsquares 1.32 on its subset, where the cases include it.
#[must_use]
pub fn report(cases: &[Case], hits: &[[Hit; 2]]) -> String {
    let both: Vec<(&Case, [Hit; 2])> = cases.iter().zip(hits.iter().copied()).collect();
    let noise = |j: usize, pick: &dyn Fn(&Case) -> bool| {
        let sdes: Vec<f64> = both
            .iter()
            .filter(|(c, _)| !c.injected && pick(c))
            .map(|(_, h)| h[j].sde)
            .collect();
        threshold(&sdes)
    };
    let (bls, tls) = (noise(0, &|_| true), noise(1, &|_| true));
    let mut out = String::new();
    let _ = writeln!(
        out,
        "## rustellar-bls against rustellar-tls on limb-darkened transits\n\n\
         SDE at 1% false alarms: rustellar-bls {bls:.2}, rustellar-tls {tls:.2}.\n\n\
         | Transits | rustellar-bls recovered | P error | t0 error (min) \
         | rustellar-tls recovered | P error | t0 error (min) |\n\
         |---|---|---|---|---|---|---|"
    );
    let injected: Vec<(&Case, [Hit; 2])> =
        both.iter().copied().filter(|(c, _)| c.injected).collect();
    let mut groups: Vec<(&str, Pick)> = NAMES
        .iter()
        .enumerate()
        .map(|(k, name)| (*name, Box::new(move |c: &Case| c.family == k) as Pick))
        .collect();
    groups.push(("snr below 10", Box::new(|c: &Case| c.snr < 10.0)));
    groups.push((
        "snr 10 to 20",
        Box::new(|c: &Case| (10.0..20.0).contains(&c.snr)),
    ));
    groups.push(("snr 20 and up", Box::new(|c: &Case| c.snr >= 20.0)));
    groups.push(("all", Box::new(|_: &Case| true)));
    for (name, pick) in &groups {
        let side = |j: usize, threshold: f64| {
            let picked: Vec<(&Case, Hit)> = injected
                .iter()
                .filter(|(c, _)| pick(c))
                .map(|(c, h)| (*c, h[j]))
                .collect();
            tally(&picked, threshold)
        };
        let _ = writeln!(out, "| {name} | {} | {} |", side(0, bls), side(1, tls));
    }

    // transitleastsquares 1.32 on its subset, each search at its own
    // threshold from the subset's noise-only curves.
    let reference: Vec<f64> = both
        .iter()
        .filter(|(c, _)| !c.injected)
        .filter_map(|(c, _)| c.tls.map(|t| t.sde))
        .collect();
    if reference.is_empty() {
        return out;
    }
    let subset = |c: &Case| c.tls.is_some();
    let thresholds = [noise(0, &subset), noise(1, &subset), threshold(&reference)];
    let _ = writeln!(
        out,
        "\nOn transitleastsquares 1.32's subset (SDE thresholds {:.2}, {:.2}, {:.2}):\n\n\
         | Search | recovered | P error | t0 error (min) |\n|---|---|---|---|",
        thresholds[0], thresholds[1], thresholds[2]
    );
    for (j, name) in ["rustellar-bls", "rustellar-tls", "transitleastsquares 1.32"]
        .iter()
        .enumerate()
    {
        let picked: Vec<(&Case, Hit)> = injected
            .iter()
            .filter_map(|(c, h)| c.tls.map(|t| (*c, if j < 2 { h[j] } else { t })))
            .collect();
        let _ = writeln!(out, "| {name} | {} |", tally(&picked, thresholds[j]));
    }
    out
}
