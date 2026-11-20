//! Transit masking on the window study's curves: each curve flattened
//! blind with the biweight at 2.5 transit durations and searched (BLS),
//! then flattened again around the candidate the search found
//! (`Detrend::flatten_around`) with several mask and window lengths, once,
//! or searched again until the search finds the same transits (at most 5
//! times). Beside them, the transits
//! masked where they truly are (the best a mask can do). For each: the
//! depth kept and signal to noise over the ideal (a matched filter), and
//! the last search's result.

use std::cell::Cell;
use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::thread;

use rustellar_core::TimeDelta;
use rustellar_detrend::{Detrend, Masking, transit_mask};

use crate::stats::median;
use crate::window::{Case, Found, Kept, active, best, group_median, ideal, inject, kept, quantile};

/// The blind window, in transit durations (the crate's recommended one).
const BLIND: f64 = 2.5;

/// One way of flattening: blind, around the search's candidate (mask and
/// window in its durations), or around the true transits (in theirs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variant {
    Blind,
    Iterative {
        mask: f64,
        window: f64,
        rounds: usize,
    },
    Truth {
        mask: f64,
        window: f64,
    },
}

/// The variants tried.
pub const VARIANTS: [Variant; 8] = [
    Variant::Blind,
    iterative(1.2, 2.0, 1),
    iterative(1.5, 2.5, 1),
    iterative(2.0, 3.0, 1),
    iterative(1.2, 2.0, 5),
    iterative(1.5, 2.5, 5),
    Variant::Truth {
        mask: 1.2,
        window: 2.0,
    },
    Variant::Truth {
        mask: 1.5,
        window: 2.5,
    },
];

const fn iterative(mask: f64, window: f64, rounds: usize) -> Variant {
    Variant::Iterative {
        mask,
        window,
        rounds,
    }
}

/// For one case and one variant: the matched filter's result, the last
/// search's, how many searches ran and whether they settled.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    pub kept: Kept,
    pub found: Found,
    pub searches: usize,
    pub converged: bool,
}

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).expect("finite")
}

/// Each variant's outcome on `case`, in the order of [`VARIANTS`].
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
    let detrend = Detrend::default()
        .with_window(days(BLIND * t14))
        .with_threads(NonZeroUsize::MIN);
    VARIANTS
        .iter()
        .map(|v| match *v {
            Variant::Blind => {
                let flat = detrend.flatten(&lc, None).expect("detrended");
                Outcome {
                    kept: kept(&flat, &shape, ideal),
                    found: best(flat.light_curve(), truth).0,
                    searches: 1,
                    converged: true,
                }
            }
            Variant::Iterative {
                mask,
                window,
                rounds,
            } => {
                let last = Cell::new(None);
                let search = |lc: &rustellar_core::LightCurve| {
                    let (found, candidate) = best(lc, truth);
                    last.set(Some(found));
                    candidate
                };
                let masking = Masking::default()
                    .with_mask_per_duration(mask)
                    .with_window_per_duration(window)
                    .with_rounds(NonZeroUsize::new(rounds).expect("not zero"));
                let done = detrend
                    .flatten_around(&lc, &masking, search)
                    .expect("detrended");
                // The last search ran on the last curve only if it settled.
                let found = if done.converged() {
                    last.get().expect("searched")
                } else {
                    best(done.flattened().light_curve(), truth).0
                };
                Outcome {
                    kept: kept(done.flattened(), &shape, ideal),
                    found,
                    searches: done.searches(),
                    converged: done.converged(),
                }
            }
            Variant::Truth { mask, window } => {
                let masked = transit_mask(
                    &lc,
                    case.transit.period(),
                    case.transit.t0(),
                    days(mask * t14),
                )
                .expect("a mask");
                let flat = detrend
                    .with_window(days(window * t14))
                    .flatten(&lc, Some(&masked))
                    .expect("detrended");
                Outcome {
                    kept: kept(&flat, &shape, ideal),
                    found: best(flat.light_curve(), truth).0,
                    searches: 0,
                    converged: true,
                }
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

#[expect(clippy::cast_precision_loss, reason = "counts")]
fn fraction(hits: usize, all: usize) -> f64 {
    hits as f64 / all as f64
}

/// The table: per variant, the median depth kept, the signal to noise over
/// the ideal (median, 10th percentile, quiet and active stars apart), how
/// often the last search's best peak is right (with SDE above 7, and at
/// all), how often it is right where the blind search's was wrong, the
/// mean number of searches and how often they settled.
#[must_use]
pub fn report(cases: &[Case], results: &[Vec<Outcome>]) -> String {
    let mut out = String::from(
        "| Flattening | Mask | Window | Depth kept | SNR / ideal | SNR / ideal, 10th percentile \
         | Quiet stars | Active stars | SNR / ideal, blind right | Search right, SDE > 7 | Right | Right where blind was not \
         | Searches | Settled |\n\
         |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n",
    );
    let blind_wrong: Vec<bool> = results.iter().map(|r| !r[0].found.right).collect();
    let wrong = blind_wrong.iter().filter(|&&w| w).count();
    for (j, v) in VARIANTS.iter().enumerate() {
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
        let right_snr: Vec<f64> = outcome
            .iter()
            .zip(&blind_wrong)
            .filter(|(_, w)| !**w)
            .map(|(o, _)| o.kept.snr)
            .collect();
        let n = outcome.len();
        let detected = outcome
            .iter()
            .filter(|o| o.found.right && o.found.sde > 7.0)
            .count();
        let right = outcome.iter().filter(|o| o.found.right).count();
        let recovered = outcome
            .iter()
            .zip(&blind_wrong)
            .filter(|(o, w)| **w && o.found.right)
            .count();
        let searches: usize = outcome.iter().map(|o| o.searches).sum();
        let settled = outcome.iter().filter(|o| o.converged).count();
        let (name, mask, window) = match *v {
            Variant::Blind => ("blind", String::from("-"), format!("{BLIND} x T14")),
            Variant::Iterative {
                mask,
                window,
                rounds,
            } => (
                if rounds == 1 {
                    "masked once"
                } else {
                    "iterative"
                },
                format!("{mask} x duration"),
                format!("{window} x duration"),
            ),
            Variant::Truth { mask, window } => (
                "true transits masked",
                format!("{mask} x T14"),
                format!("{window} x T14"),
            ),
        };
        let line = format!(
            "| {name} | {mask} | {window} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} \
             | {:.3} | {:.3} | {:.2} | {:.3} |",
            median(&depth),
            median(&snr),
            quantile(&snr, 0.1),
            group_median(&group(&|c| !active(c))),
            group_median(&group(&active)),
            group_median(&right_snr),
            fraction(detected, n),
            fraction(right, n),
            fraction(recovered, wrong),
            fraction(searches, n),
            fraction(settled, n),
        );
        let _ = writeln!(out, "{line}");
    }
    out
}
