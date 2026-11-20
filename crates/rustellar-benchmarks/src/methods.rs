//! Detrending methods compared on the window study's curves (limb-darkened
//! transits on spotted, rotating stars, a TESS sector at 10 minutes): the
//! biweight at 2.5 transit durations (the crate's recommended window) beside
//! the splines with knots 1, 1.5, 2.5 and 4 durations apart, plain, clipped
//! and Huber. For each: the depth kept and signal to noise over the ideal
//! (a matched filter), and a blind BLS search; wotan 1.10's `biweight`,
//! `rspline` and `hspline` on the same curves from a recorded file, depth
//! and signal to noise only.

use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::thread;

use rustellar_core::TimeDelta;
use rustellar_detrend::{Detrend, Method};

use crate::stats::median;
use crate::window::{
    Case, Found, Kept, active, found, group_median, ideal, inject, kept, quantile, share,
};

/// One method tried, with its window (or knot spacing) in transit
/// durations.
#[derive(Debug, Clone, Copy)]
pub struct Trial {
    pub name: &'static str,
    pub method: Method,
    pub factor: f64,
}

const fn trial(name: &'static str, method: Method, factor: f64) -> Trial {
    Trial {
        name,
        method,
        factor,
    }
}

const ROBUST: Method = Method::RobustSpline { sigma: 3.0 };
const HUBER: Method = Method::HuberSpline { c: 1.345 };

/// The methods tried.
pub const TRIALS: [Trial; 13] = [
    trial("biweight", Method::Biweight { c: 5.0 }, 2.5),
    trial("spline", Method::Spline, 1.0),
    trial("spline", Method::Spline, 1.5),
    trial("spline", Method::Spline, 2.5),
    trial("spline", Method::Spline, 4.0),
    trial("robust spline", ROBUST, 1.0),
    trial("robust spline", ROBUST, 1.5),
    trial("robust spline", ROBUST, 2.5),
    trial("robust spline", ROBUST, 4.0),
    trial("Huber spline", HUBER, 1.0),
    trial("Huber spline", HUBER, 1.5),
    trial("Huber spline", HUBER, 2.5),
    trial("Huber spline", HUBER, 4.0),
];

/// For one case and one method: the matched filter's result and the blind
/// search's.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    pub kept: Kept,
    pub found: Found,
}

/// Each method's outcome on `case`, in the order of [`TRIALS`].
///
/// # Panics
///
/// If the curve cannot be built, detrended or searched (it always can).
#[must_use]
pub fn measure(case: &Case) -> Vec<Outcome> {
    let (lc, shape) = inject(case);
    let ideal = ideal(&shape);
    let t14 = case.transit.t14().expect("a transit").as_days();
    let truth = case.transit.period().as_days();
    TRIALS
        .iter()
        .map(|t| {
            let flat = Detrend::default()
                .with_method(t.method)
                .with_window(TimeDelta::from_days(t.factor * t14).expect("finite"))
                .with_threads(NonZeroUsize::MIN)
                .flatten(&lc, None)
                .expect("detrended");
            Outcome {
                kept: kept(&flat, &shape, ideal),
                found: found(flat.light_curve(), truth),
            }
        })
        .collect()
}

/// [`measure`] for every case, on `threads` threads.
#[must_use]
pub fn measure_all(cases: &[Case], threads: NonZeroUsize) -> Vec<Vec<Outcome>> {
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

/// wotan's results, recorded: seed, method, factor, depth kept, signal to
/// noise over the ideal.
const WOTAN: &str = include_str!("../data/methods_wotan.csv");

/// The blind search on wotan's flattened curves ([`search_flattened`]):
/// seed, method, factor, right (1 or 0), SDE.
const WOTAN_SEARCH: &str = include_str!("../data/methods_wotan_search.csv");

/// Whether the search on wotan's `method` at `factor` found `seed`'s
/// transit (right period, SDE above 7).
fn wotan_detected(seed: u64, method: &str, factor: f64) -> bool {
    WOTAN_SEARCH.lines().skip(1).any(|line| {
        let r: Vec<&str> = line.split(',').collect();
        r[0].parse() == Ok(seed)
            && r[1] == method
            && r[2]
                .parse::<f64>()
                .is_ok_and(|f| f.to_bits() == factor.to_bits())
            && r[3] == "1"
            && r[4].parse::<f64>().is_ok_and(|sde| sde > 7.0)
    })
}

/// One recorded wotan method: its name, its factor, and per seed the depth
/// kept and the signal to noise over the ideal.
type Recorded = (String, f64, Vec<(u64, f64, f64)>);

/// The recorded wotan rows for the seeds of `cases`, per (method, factor)
/// in the file's order.
fn wotan(cases: &[Case]) -> Vec<Recorded> {
    let mut groups: Vec<Recorded> = vec![];
    for line in WOTAN.lines().skip(1) {
        let r: Vec<&str> = line.split(',').collect();
        let seed: u64 = r[0].parse().expect("a seed");
        if !cases.iter().any(|c| c.seed == seed) {
            continue;
        }
        let factor: f64 = r[2].parse().expect("a factor");
        let row = (
            seed,
            r[3].parse().expect("a depth"),
            r[4].parse().expect("an SNR"),
        );
        match groups
            .iter_mut()
            .find(|g| g.0 == r[1] && g.1.to_bits() == factor.to_bits())
        {
            Some(g) => g.2.push(row),
            None => groups.push((r[1].to_owned(), factor, vec![row])),
        }
    }
    groups
}

/// The table: per method, the median depth kept, the signal to noise over
/// the ideal (median, 10th percentile, quiet and active stars apart), and
/// how often a blind search's best peak is right with SDE above 7; then
/// wotan's rows.
#[must_use]
pub fn report(cases: &[Case], results: &[Vec<Outcome>]) -> String {
    let mut out = String::from(
        "| Method | Knots or window | Depth kept | SNR / ideal | SNR / ideal, 10th percentile \
         | Quiet stars | Active stars | Search right, SDE > 7 |\n\
         |---|---|--:|--:|--:|--:|--:|--:|\n",
    );
    let detected = |f: &Found| f.right && f.sde > 7.0;
    for (j, t) in TRIALS.iter().enumerate() {
        let outcome: Vec<Outcome> = results.iter().map(|r| r[j]).collect();
        let snr: Vec<f64> = outcome.iter().map(|o| o.kept.snr).collect();
        let group = |keep: &dyn Fn(&Case) -> bool| -> Vec<f64> {
            cases
                .iter()
                .zip(&outcome)
                .filter(|(c, _)| keep(c))
                .map(|(_, o)| o.kept.snr)
                .collect()
        };
        let depth: Vec<f64> = outcome.iter().map(|o| o.kept.depth).collect();
        let found: Vec<Found> = outcome.iter().map(|o| o.found).collect();
        let _ = writeln!(
            out,
            "| {} | {} x T14 | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            t.name,
            t.factor,
            median(&depth),
            median(&snr),
            quantile(&snr, 0.1),
            group_median(&group(&|c| !active(c))),
            group_median(&group(&active)),
            share(cases, &found, |_| true, detected),
        );
    }
    for (name, factor, rows) in wotan(cases) {
        let pick = |keep: &dyn Fn(&Case) -> bool, value: fn(&(u64, f64, f64)) -> f64| {
            rows.iter()
                .filter(|r| cases.iter().any(|c| c.seed == r.0 && keep(c)))
                .map(value)
                .collect::<Vec<f64>>()
        };
        let snr = pick(&|_| true, |r| r.2);
        let detected = rows
            .iter()
            .filter(|r| wotan_detected(r.0, &name, factor))
            .count();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let share = detected as f64 / rows.len() as f64;
        let line = format!(
            "| wotan {name} | {factor} x T14 | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {share:.3} |",
            median(&pick(&|_| true, |r| r.1)),
            median(&snr),
            quantile(&snr, 0.1),
            group_median(&pick(&|c| !active(c), |r| r.2)),
            group_median(&pick(&active, |r| r.2)),
        );
        let _ = writeln!(out, "{line}");
    }
    out
}

/// Every case's light curve and transit shape, for wotan to be run on:
/// `seed,time,flux,shape` rows.
///
/// # Panics
///
/// If a curve cannot be built (it always can).
#[must_use]
pub fn export(cases: &[Case]) -> String {
    let mut out = String::from("seed,t14,time,flux,shape\n");
    for case in cases {
        let (lc, shape) = inject(case);
        let t14 = case.transit.t14().expect("a transit").as_days();
        for ((t, f), s) in lc.time().iter().zip(lc.flux()).zip(&shape) {
            let _ = writeln!(out, "{},{t14:?},{:?},{f:?},{s:?}", case.seed, t.as_btjd());
        }
    }
    out
}

/// One recorded curve flattened by wotan: its seed, method and factor, and
/// its times and fluxes.
type Flat = (u64, String, String, Vec<f64>, Vec<f64>);

/// A blind BLS search, as on this crate's own curves, on each curve wotan
/// flattened: `lines` are `seed,method,factor,time,flux` rows (a header
/// first), each curve's rows together. `seed,method,factor,right,sde`
/// rows back, `right` 1 where the best peak is the injected period.
///
/// # Panics
///
/// On a malformed row, or a curve that cannot be searched.
#[must_use]
pub fn search_flattened(lines: impl Iterator<Item = String>, threads: NonZeroUsize) -> String {
    let mut curves: Vec<Flat> = vec![];
    for line in lines.skip(1) {
        let r: Vec<&str> = line.split(',').collect();
        let seed: u64 = r[0].parse().expect("a seed");
        let (t, f): (f64, f64) = (r[3].parse().expect("a time"), r[4].parse().expect("a flux"));
        match curves.last_mut() {
            Some(c) if c.0 == seed && c.1 == r[1] && c.2 == r[2] => {
                c.3.push(t);
                c.4.push(f);
            }
            _ => curves.push((seed, r[1].to_owned(), r[2].to_owned(), vec![t], vec![f])),
        }
    }
    let most = curves.iter().map(|c| c.0).max().unwrap_or(0);
    let truths: Vec<f64> = crate::window::cases(most)
        .iter()
        .map(|c| c.transit.period().as_days())
        .collect();
    let chunk = curves.len().div_ceil(threads.get()).max(1);
    let lines: Vec<String> = thread::scope(|scope| {
        let handles: Vec<_> = curves
            .chunks(chunk)
            .map(|part| {
                let truths = &truths;
                scope.spawn(move || {
                    part.iter()
                        .map(|(seed, method, factor, t, f)| {
                            let lc = rustellar_core::LightCurve::from_btjd(
                                t.clone(),
                                f.clone(),
                                vec![1e-3; t.len()],
                            )
                            .expect("a light curve");
                            let index = usize::try_from(*seed).expect("small") - 1;
                            let hit = found(&lc, truths[index]);
                            format!(
                                "{seed},{method},{factor},{},{:?}\n",
                                u8::from(hit.right),
                                hit.sde
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a search thread panicked"))
            .collect()
    });
    let mut out = String::from("seed,method,factor,right,sde\n");
    out.extend(lines);
    out
}
