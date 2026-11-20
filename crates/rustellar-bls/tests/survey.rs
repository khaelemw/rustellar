// Survey light curves (TESS 2 and 30 minute, several sectors over two years,
// Kepler-like quarters, nights from the ground, thinned data): the exact
// search never below astropy's slow search, the binned search never above
// the exact one, and how the binned search compares with astropy's fast one.

#[expect(dead_code, reason = "the survey tests use only some of the recipes")]
#[path = "../src/tests/recipe.rs"]
mod recipe;

use std::num::NonZeroUsize;
use std::thread;

use recipe::{Survey, check_samples};
use rustellar_bls::{Bls, Durations, Objective, Periods, SearchOptions, TransitBox};
use rustellar_core::{Epoch, LightCurve, TimeDelta};

// ==========================
// Helper Functions
// ==========================

const SURVEY_CURVES: &str = include_str!("reference/survey_curves.csv");
const SURVEY_POWER: &str = include_str!("reference/survey_power.csv");

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

struct Curve {
    family: String,
    lc: LightCurve,
    durations: Vec<f64>,
}

// astropy at one period of one curve.
struct Astropy {
    curve: usize,
    period: f64,
    fast: f64,
    fast_duration: f64,
    fast_transit_time: f64,
    slow: f64,
}

fn curves() -> Vec<Curve> {
    rows(SURVEY_CURVES)
        .map(|v| {
            let f = |i: usize| v[i].parse::<f64>().unwrap();
            let lc = Survey {
                seed: v[2].parse().unwrap(),
                segments: v[3]
                    .split('|')
                    .map(|s| {
                        let (start, n) = s.split_once(':').unwrap();
                        (start.parse().unwrap(), n.parse().unwrap())
                    })
                    .collect(),
                cadence: f(4),
                noise: f(5),
                keep: f(6),
                period: f(7),
                tc: f(8),
                duration: f(9),
                depth: f(10),
            }
            .light_curve();
            assert_eq!(lc.len(), v[12].parse::<usize>().unwrap(), "curve {}", v[0]);
            check_samples(&lc, v[13]);
            Curve {
                family: v[1].to_string(),
                lc,
                durations: v[11].split(':').map(|d| d.parse().unwrap()).collect(),
            }
        })
        .collect()
}

fn astropy() -> Vec<Astropy> {
    rows(SURVEY_POWER)
        .map(|v| {
            let f = |i: usize| v[i].parse::<f64>().unwrap();
            Astropy {
                curve: v[0].parse().unwrap(),
                period: f(1),
                fast: f(2),
                fast_duration: f(3),
                fast_transit_time: f(4),
                slow: f(5),
            }
        })
        .collect()
}

// One curve's comparison at each of its periods.
struct Row {
    // Our exact search's best snr over the durations.
    exact: f64,
    // Our binned search's snr, its box's duration, and the exact search's
    // best snr at that duration.
    binned: f64,
    binned_duration: f64,
    exact_at_binned: f64,
    // The exact search's best snr at astropy fast's duration, and astropy
    // fast's own box fitted here.
    exact_at_fast: f64,
    fast_refit: f64,
}

fn best_snr(lc: &LightCurve, period: f64, duration: f64) -> f64 {
    TransitBox::best(lc, days(period), days(duration))
        .unwrap()
        .map_or(0.0, |b| b.fit.snr)
}

fn days(value: f64) -> TimeDelta {
    TimeDelta::from_days(value).unwrap()
}

fn compare(curve: &Curve, astropy: &[&Astropy]) -> Vec<Row> {
    let periods: Vec<f64> = astropy.iter().map(|a| a.period).collect();
    let binned = Bls::new(&curve.lc)
        .unwrap()
        .search(
            &Periods::from_days(&periods).unwrap(),
            &Durations::from_days(&curve.durations).unwrap(),
            &SearchOptions::default()
                .with_objective(Objective::Snr)
                .with_refine_peaks(0)
                .with_threads(NonZeroUsize::MIN),
        )
        .unwrap();
    astropy
        .iter()
        .zip(binned.power().iter().zip(binned.boxes()))
        .map(|(a, (&power, found))| {
            let found = found.expect("every period has a box with a dip");
            let binned_duration = found.transit.duration().as_days();
            let fast_box = TransitBox::new(
                days(a.period),
                days(a.fast_duration),
                Epoch::from_btjd(a.fast_transit_time).unwrap(),
            )
            .unwrap();
            Row {
                exact: curve
                    .durations
                    .iter()
                    .map(|&d| best_snr(&curve.lc, a.period, d))
                    .fold(0.0, f64::max),
                binned: power,
                binned_duration,
                exact_at_binned: best_snr(&curve.lc, a.period, binned_duration),
                exact_at_fast: best_snr(&curve.lc, a.period, a.fast_duration),
                fast_refit: fast_box.fit(&curve.lc).unwrap().snr,
            }
        })
        .collect()
}

// Every curve compared, in parallel.
fn survey() -> (Vec<Curve>, Vec<Astropy>, Vec<Row>) {
    let (curves, astropy) = (curves(), astropy());
    let rows = thread::scope(|scope| {
        let handles: Vec<_> = curves
            .iter()
            .enumerate()
            .map(|(c, curve)| {
                let mine: Vec<&Astropy> = astropy.iter().filter(|a| a.curve == c).collect();
                scope.spawn(move || compare(curve, &mine))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    (curves, astropy, rows)
}

// ==========================
// Against astropy
// ==========================

// The light curves are the reference data's, bit for bit: 36 curves of 1,776 to
// 18,000 points, five families.
#[test]
fn survey_curves_are_the_reference_curves() {
    let curves = curves();
    assert_eq!(curves.len(), 36);
    let families: std::collections::BTreeSet<&str> =
        curves.iter().map(|c| c.family.as_str()).collect();
    assert_eq!(families.len(), 5);
    assert_eq!(astropy().len(), 36 * 24);
}

#[test]
fn survey_against_astropy() {
    let (_, astropy, rows) = survey();
    let mut slow_better = 0;
    let mut fast_not_its_box = 0;
    for (a, row) in astropy.iter().zip(&rows) {
        let at = format!("curve {}, P {}", a.curve, a.period);
        // astropy's slow search tries a grid of mid-transit times; the exact
        // search tries every set of points a box can hold.
        assert!(
            row.exact >= a.slow * (1.0 - 1e-12),
            "{at}: {} vs {}",
            row.exact,
            a.slow
        );
        if row.exact > a.slow * (1.0 + 1e-9) {
            slow_better += 1;
        }
        // The binned box is one the exact search tries at its own duration
        // (the searched one rounded to whole bins).
        assert!(
            row.binned <= row.exact_at_binned * (1.0 + 1e-12),
            "{at}: binned {} above exact {} at {}",
            row.binned,
            row.exact_at_binned,
            row.binned_duration
        );
        // astropy's fast search also rounds each duration to whole bins. Its
        // power is that of a box of its duration, which the exact search
        // tries too, except where its padding has overwritten its last real
        // bin: then its power is not its own box's (which fits worse).
        if a.fast > row.exact_at_fast * (1.0 + 1e-9) {
            assert!(
                row.fast_refit < a.fast * (1.0 - 1e-9),
                "{at}: fast {} above exact {}, and its box fits to {}",
                a.fast,
                row.exact_at_fast,
                row.fast_refit
            );
            fast_not_its_box += 1;
        }
    }
    // A grid of mid-transit times misses the best box at most periods.
    assert_eq!(slow_better, 788, "exact above slow");
    // Four, all low snr (under 2.4, noise), each box running through
    // astropy's last, part bin and past phase zero: curve 6 at 1.27 days,
    // 15 at 2.57, 21 at 1.38 and 30 at 3.62.
    assert_eq!(fast_not_its_box, 4, "astropy fast's power not its box's");
}
