//! Survey-scale light curves (a TESS sector at 2-minute cadence, a fast
//! rotator, counts, a faint star, a Kepler quarter) with spots, drift,
//! flares, noise and transits: the trend at sampled points against the
//! biweight and Huber roots at 50 digits and against wotan's, and the
//! transits found after detrending, ours and wotan's. Heavy: run with
//! `cargo test --release -p rustellar-detrend --test survey -- --ignored`.

use std::num::NonZeroUsize;
use std::time::Instant;

use rustellar_bls::{Bls, Durations, Objective, OfirPeriod, SearchOptions};
use rustellar_core::{LightCurve, TimeDelta};
use rustellar_detrend::Method;

#[path = "common/recipe.rs"]
mod recipe;

use recipe::{Built, Curve, curves, fingerprint};

const WINDOWS: &str = include_str!("reference/survey_windows.csv");
const WOTAN: &str = include_str!("reference/survey_wotan.csv");
const HUBER: &str = include_str!("reference/survey_huber.csv");

/// Each curve built and checked against the generator's fingerprint.
fn built() -> Vec<(Curve, Built)> {
    curves()
        .into_iter()
        .map(|c| {
            let b = c.build();
            assert_eq!(b.time.len(), c.points, "{}: points", c.name);
            assert_eq!(
                fingerprint(&b.time, &b.flux),
                c.fingerprint,
                "{}: not the generator's light curve",
                c.name
            );
            (c, b)
        })
        .collect()
}

/// wotan's trend over each curve, from trend / model - 1.
fn wotan_trend(name: &str, model: &[f64]) -> Vec<f64> {
    let line = WOTAN
        .lines()
        .find(|l| l.split(',').next() == Some(name))
        .expect("wotan's trend for each curve");
    let delta: Vec<f64> = line
        .split_once(',')
        .unwrap()
        .1
        .split(';')
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(delta.len(), model.len(), "{name}");
    model
        .iter()
        .zip(delta)
        .map(|(m, d)| m * (1.0 + d))
        .collect()
}

fn ulp(x: f64) -> f64 {
    let x = x.abs();
    f64::from_bits(x.to_bits() + 1) - x
}

// The rounding bound's share, as in the crate's own reference tests: about
// eight roundings per term, summed in double-double.
const ROUNDINGS: f64 = 8.0;

// ==========================
// Tests
// ==========================

/// At every sampled point, our trend is within half a unit (plus the
/// roundings' doubt) of the root at 50 digits, and the nearest double
/// wherever that doubt is small. wotan's is never nearer where its window
/// holds the same points; elsewhere (its window is half open, its edges
/// rounded) it is measured against its own window's root.
#[test]
#[ignore = "heavy: survey-scale light curves"]
fn the_survey_trend_at_fifty_digits() {
    let (mut seen, mut certain, mut nearest) = (0, 0, 0);
    for (curve, b) in built() {
        let lc = curve.light_curve(&b.time, &b.flux);
        let start = Instant::now();
        let trend = curve.detrend().trend(&lc, None).unwrap();
        let elapsed = start.elapsed();
        let (mut ours_worst, mut wotan_worst, mut wotan_rel) = (0.0f64, 0.0f64, 0.0f64);
        let (mut same, mut wotan_worse, mut wotan_nearest) = (0, 0, 0);
        let (mut own_window, mut our_window) = (0.0, 0.0);
        for line in WINDOWS.lines().filter(|l| !l.starts_with('#')).skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            if f[0] != curve.name {
                continue;
            }
            let x = |k: usize| f[k].parse::<f64>().unwrap();
            let i: usize = f[1].parse().unwrap();
            let e: i32 = f[2].parse().unwrap();
            let (hi, lo, sensitivity, wotan) = (x(3), x(4), x(6), x(7));
            let scale = 2f64.powi(-e);
            let ours = trend[i].expect("a trend at every point");
            let error = |t: f64| ((t * scale - hi) - lo).abs();
            let unit = ulp(hi);
            let rounding = ROUNDINGS * f64::EPSILON * f64::EPSILON * sensitivity;
            assert!(
                error(ours) <= 0.5 * unit + rounding,
                "{} point {i}: {:.3} ulp",
                curve.name,
                error(ours) / unit
            );
            seen += 1;
            nearest += usize::from(error(ours) <= 0.5 * unit);
            if rounding <= 0.01 * unit {
                certain += 1;
                assert!(error(ours) <= 0.5 * unit, "{} point {i}", curve.name);
            }
            ours_worst = ours_worst.max(error(ours) / unit);
            if f[8] == "1" {
                same += 1;
                assert!(
                    error(wotan) >= error(ours),
                    "{} point {i}: wotan nearer",
                    curve.name
                );
                wotan_worse += usize::from(error(wotan) > error(ours));
            }
            // wotan against the root of its own window.
            let (w_hi, w_lo) = (x(9), x(10));
            let w_error = ((wotan * scale - w_hi) - w_lo).abs();
            wotan_nearest += usize::from(w_error <= 0.5 * ulp(w_hi));
            wotan_worst = wotan_worst.max(w_error / ulp(w_hi));
            wotan_rel = wotan_rel.max(w_error / w_hi.abs());
            if f[8] == "0" {
                // Where the windows differ, wotan nearer its own window's
                // root than ours, on the whole: the roots are its windows'.
                own_window += w_error;
                our_window += error(wotan);
            }
        }
        assert!(own_window < our_window, "{}", curve.name);
        // Never the nearest double, of either.
        assert_eq!(wotan_nearest, 0, "{}", curve.name);
        // Everywhere, wotan's trend against ours (wotan's to six digits; its
        // windows differ at the edges, and most at a segment's ends).
        let wotan = wotan_trend(&curve.name, &b.model);
        let apart: Vec<f64> = trend
            .iter()
            .zip(&wotan)
            .map(|(o, w)| (o.unwrap() / w - 1.0).abs())
            .collect();
        let most = apart.iter().copied().fold(0.0, f64::max);
        let mut sorted = apart.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        eprintln!(
            "{}: {} points in {:.3} s; worst {ours_worst:.3} ulp; wotan worse at {wotan_worse} \
             of {same} same windows; wotan the nearest double of its own window's root at \
             {wotan_nearest}, worst {wotan_worst:.2e} ulp ({wotan_rel:.2e} relative); trends \
             apart by {median:.1e} (median), at most {most:.1e}",
            curve.name,
            b.time.len(),
            elapsed.as_secs_f64()
        );
        assert!(most < 1e-3, "{}: {most:e}", curve.name);
    }
    eprintln!("{seen} windows: {nearest} the nearest double, {certain} certain");
    assert!(
        seen > 1_000 && certain * 100 >= seen * 99,
        "{certain} of {seen}"
    );
}

/// Huber's M-estimator (c = 1.28) at the same sampled points: ours within
/// half a unit (plus the roundings' doubt) of its root at 50 digits, and the
/// nearest double wherever that doubt is small. wotan's `huber_psi` (c 1.28)
/// is one step from the median, so it is measured against its own window's
/// root, and is never nearer where the windows agree.
#[test]
#[ignore = "heavy: survey-scale light curves"]
fn the_survey_huber_at_fifty_digits() {
    let (mut seen, mut certain, mut nearest) = (0, 0, 0);
    let (mut wotan_nearest, mut wotan_rel) = (0, 0.0f64);
    for (curve, b) in built() {
        let lc = curve.light_curve(&b.time, &b.flux);
        let trend = curve
            .detrend()
            .with_method(Method::Huber { c: 1.28 })
            .trend(&lc, None)
            .unwrap();
        let mut ours_worst = 0.0f64;
        for line in HUBER.lines().filter(|l| !l.starts_with('#')).skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            if f[0] != curve.name {
                continue;
            }
            let x = |k: usize| f[k].parse::<f64>().unwrap();
            let i: usize = f[1].parse().unwrap();
            let e: i32 = f[2].parse().unwrap();
            let (hi, lo, sensitivity, wotan) = (x(3), x(4), x(6), x(7));
            let scale = 2f64.powi(-e);
            let ours = trend[i].expect("a trend at every point");
            let error = |t: f64| ((t * scale - hi) - lo).abs();
            let unit = ulp(hi);
            let rounding = ROUNDINGS * f64::EPSILON * f64::EPSILON * sensitivity;
            assert!(
                error(ours) <= 0.5 * unit + rounding,
                "{} point {i}: {:.3} ulp",
                curve.name,
                error(ours) / unit
            );
            seen += 1;
            nearest += usize::from(error(ours) <= 0.5 * unit);
            if rounding <= 0.01 * unit {
                certain += 1;
                assert!(error(ours) <= 0.5 * unit, "{} point {i}", curve.name);
            }
            ours_worst = ours_worst.max(error(ours) / unit);
            if f[8] == "1" {
                assert!(
                    error(wotan) >= error(ours),
                    "{} point {i}: wotan nearer",
                    curve.name
                );
            }
            let (w_hi, w_lo) = (x(9), x(10));
            let w_error = ((wotan * scale - w_hi) - w_lo).abs();
            wotan_nearest += usize::from(w_error <= 0.5 * ulp(w_hi));
            wotan_rel = wotan_rel.max(w_error / w_hi.abs());
        }
        eprintln!("{}: Huber worst {ours_worst:.3} ulp", curve.name);
    }
    eprintln!(
        "Huber, {seen} windows: ours the nearest double at {nearest}, {certain} certain; \
         wotan's huber_psi the nearest of its own window's root at {wotan_nearest}, \
         worst {wotan_rel:.2e} relative"
    );
    assert!(
        seen > 1_000 && certain * 100 >= seen * 99,
        "{certain} of {seen}"
    );
    assert_eq!(wotan_nearest, 0);
}

// The search's durations, days.
const DURATIONS: [f64; 7] = [0.04, 0.06, 0.08, 0.11, 0.15, 0.2, 0.25];

/// A box found: period, mid-transit time, depth.
#[derive(Debug, Clone, Copy)]
struct Found {
    period: f64,
    t0: f64,
    depth: f64,
}

fn search(lc: &LightCurve) -> Found {
    let periods = OfirPeriod {
        minimum_period: Some(TimeDelta::from_days(0.6).unwrap()),
        ..OfirPeriod::default()
    }
    .periods(lc)
    .unwrap();
    let threads = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let result = Bls::new(lc)
        .unwrap()
        .search(
            &periods,
            &Durations::from_days(&DURATIONS).unwrap(),
            &SearchOptions::default()
                .with_objective(Objective::Snr)
                .with_threads(threads),
        )
        .unwrap();
    let best = result.best().expect("a box with a dip");
    Found {
        period: best.transit.period().as_days(),
        t0: best.transit.transit_time().as_btjd(),
        depth: best.fit.depth,
    }
}

/// Each curve flattened by our trend and by wotan's, then searched: both
/// find the injected transits, with the same period and mid-transit time
/// and depths that agree.
#[test]
#[ignore = "heavy: survey-scale searches"]
fn detrend_then_search() {
    for (curve, b) in built() {
        let lc = curve.light_curve(&b.time, &b.flux);
        let flat = curve.detrend().flatten(&lc, None).unwrap();
        assert_eq!(flat.indices().len(), lc.len(), "{}", curve.name);
        let wotan = wotan_trend(&curve.name, &b.model);
        let flux: Vec<f64> = b.flux.iter().zip(&wotan).map(|(f, w)| f / w).collect();
        let error: Vec<f64> = wotan
            .iter()
            .map(|w| curve.scale * curve.noise / w)
            .collect();
        let theirs = LightCurve::from_btjd(b.time.clone(), flux, error).unwrap();
        let tc = curve.t_start + curve.tc;
        // Errors: period relative, mid-transit time in days (to the nearest
        // injected transit), depth relative.
        let off = |found: Found| {
            let k = ((found.t0 - tc) / curve.period).round();
            [
                (found.period / curve.period - 1.0).abs(),
                (found.t0 - tc - k * curve.period).abs(),
                (found.depth / curve.depth - 1.0).abs(),
            ]
        };
        let (ours, theirs) = (off(search(flat.light_curve())), off(search(&theirs)));
        eprintln!(
            "{}: ours P {:.2e} t0 {:.2e} d depth {:.3}; wotan's P {:.2e} t0 {:.2e} d depth {:.3}",
            curve.name, ours[0], ours[1], ours[2], theirs[0], theirs[1], theirs[2]
        );
        for found in [ours, theirs] {
            assert!(
                found[0] < 2e-3 && found[1] < curve.duration / 4.0 && found[2] < 0.5,
                "{}: {found:?}",
                curve.name
            );
        }
        // The two agree: the trends differ by a few ppm, mostly where the
        // windows' edges differ, far below the noise.
        assert!(
            (ours[0] - theirs[0]).abs() < 1e-4
                && (ours[1] - theirs[1]).abs() < 0.01
                && (ours[2] - theirs[2]).abs() < 0.02,
            "{}: {ours:?} against {theirs:?}",
            curve.name
        );
    }
}
