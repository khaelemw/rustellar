// Compares exposure-averaged light curves with batman (Kreidberg 2015) and
// PyTransit (Parviainen 2015) for seven kinds of transit (from a small planet
// to an eclipsing binary), every common Kepler and TESS cadence (20 s to
// 30 min), and 1 to 30 samples. The data file describes how it was made in
// its header comments.
//
// The two packages place their samples differently. PyTransit uses the middle
// of equal slices of the exposure (the midpoint rule of Kipping 2010), as this
// crate does; batman spreads its samples from edge to edge. These tests check
// that this crate matches PyTransit sample for sample, that batman's rule
// applied to this crate's flux reproduces batman, and that the midpoint rule
// is the more accurate of the two.

use std::{collections::BTreeMap, num::NonZeroUsize, sync::OnceLock};

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

// ==========================
// Helper Functions
// ==========================

const CSV: &str = include_str!("reference/exposure_points.csv");

const T0: f64 = 3_000.123_456;

// The sets where PyTransit's series approximation to the orbit is off even
// with no averaging: by 4.5e-6 at a_rs = 4 and 9.6e-7 for the eclipsing
// binary (k = 0.5, a_rs = 8). See the last test.
const PYTRANSIT_ORBIT_APPROXIMATE: [usize; 2] = [3, 4];

// Measured worst difference from PyTransit with the same samples, on the
// other sets and with every sample clear of the case lines: 3.6e-8, the error
// of PyTransit's own flux.
const PYTRANSIT_TOLERANCE: f64 = 6e-8;

// PyTransit is off by up to about 3e-6 within 1e-4 of the lines where Mandel
// & Agol's formula changes case (see tests/flux_reference.rs), so points with
// a sample that close are left out of the comparison with it.
const CASE_LINE_MARGIN: f64 = 1e-4;

// Measured worst difference between batman and batman's rule applied to this
// crate's flux: 1.2e-8, the error of batman's own flux.
const BATMAN_TOLERANCE: f64 = 3e-8;

#[derive(Debug, Clone, Copy)]
struct Point {
    set: usize,
    k: f64,
    b: f64,
    a_rs: f64,
    period: f64,
    exposure: f64,
    samples: usize,
    t: f64,
    batman: f64,
    pytransit: f64,
    // batman's instantaneous flux averaged over 2001 midpoint samples: the
    // exposure average with negligible sampling error.
    dense: f64,
}

impl Point {
    fn params(&self) -> TransitParams {
        TransitParams::new(
            Epoch::from_days(T0).unwrap(),
            TimeDelta::from_days(self.period).unwrap(),
            RadiusRatio::new(self.k).unwrap(),
            ScaledSemiMajorAxis::new(self.a_rs).unwrap(),
            ImpactParameter::new(self.b).unwrap(),
            LimbDarkening::quadratic(0.4, 0.25).unwrap(),
        )
        .unwrap()
    }

    fn time(&self) -> Epoch {
        Epoch::from_days(self.t).unwrap()
    }

    fn exposure(&self) -> Exposure {
        Exposure::new(
            TimeDelta::from_days(self.exposure).unwrap(),
            NonZeroUsize::new(self.samples).unwrap(),
        )
        .unwrap()
    }

    fn binned(&self) -> f64 {
        self.params().binned_flux_at(self.time(), self.exposure())
    }

    // batman's rule: samples at linspace(-e/2, e/2, n), or the time stamp
    // alone for n = 1.
    fn batman_rule(&self) -> f64 {
        let params = self.params();
        if self.samples == 1 {
            return params.flux_at(self.time());
        }
        let n = self.samples as f64;
        let sum: f64 = (0..self.samples)
            .map(|i| {
                let offset = self.exposure * (i as f64 / (n - 1.0) - 0.5);
                params.flux_at(self.time() + TimeDelta::from_days(offset).unwrap())
            })
            .sum();
        sum / n
    }

    // Whether any sample lies within `CASE_LINE_MARGIN` of a case line.
    fn sample_near_case_line(&self) -> bool {
        let params = self.params();
        self.exposure().offsets().any(|offset| {
            let z = params.separation(self.time() + offset);
            [self.k, (1.0 - self.k).abs(), 1.0 + self.k]
                .iter()
                .any(|line| (z - line).abs() < CASE_LINE_MARGIN)
        })
    }

    fn cadence_seconds(&self) -> u64 {
        (self.exposure * 86_400.0).round() as u64
    }
}

fn points() -> &'static [Point] {
    static POINTS: OnceLock<Vec<Point>> = OnceLock::new();
    POINTS.get_or_init(|| {
        CSV.lines()
            .filter(|line| !line.starts_with('#'))
            .skip(1)
            .map(|line| {
                let v: Vec<f64> = line
                    .split(',')
                    .map(|field| field.parse().unwrap_or_else(|_| panic!("{line}")))
                    .collect();
                assert_eq!(v.len(), 11, "{line}");
                Point {
                    set: v[0] as usize,
                    k: v[1],
                    b: v[2],
                    a_rs: v[3],
                    period: v[4],
                    exposure: v[5],
                    samples: v[6] as usize,
                    t: v[7],
                    batman: v[8],
                    pytransit: v[9],
                    dense: v[10],
                }
            })
            .collect()
    })
}

fn count_in_header() -> usize {
    CSV.lines()
        .next()
        .and_then(|line| line.strip_prefix("# "))
        .and_then(|line| line.split(' ').next())
        .and_then(|count| count.parse().ok())
        .expect("first line should be '# <count> ...'")
}

// Worst error against the dense average, per (set, cadence, samples), for
// this crate and for batman.
fn worst_errors() -> BTreeMap<(usize, u64, usize), (f64, f64)> {
    let mut worst = BTreeMap::new();
    for p in points() {
        let entry = worst
            .entry((p.set, p.cadence_seconds(), p.samples))
            .or_insert((0.0_f64, 0.0_f64));
        entry.0 = entry.0.max((p.binned() - p.dense).abs());
        entry.1 = entry.1.max((p.batman - p.dense).abs());
    }
    worst
}

// ==========================
// Tests
// ==========================

#[test]
fn reference_file_is_complete() {
    assert_eq!(points().len(), count_in_header());
}

#[test]
fn matches_pytransit_sample_for_sample() {
    let mut checked = 0;
    for p in points() {
        if PYTRANSIT_ORBIT_APPROXIMATE.contains(&p.set) || p.sample_near_case_line() {
            continue;
        }
        let difference = (p.binned() - p.pytransit).abs();
        assert!(difference < PYTRANSIT_TOLERANCE, "{difference:e} at {p:?}");
        checked += 1;
    }
    assert!(checked > 10_000, "only {checked} points");
}

// Confirms both that the orbit geometry matches batman's and that batman's
// samples run from edge to edge of the exposure.
#[test]
fn batman_rule_on_this_flux_reproduces_batman() {
    for p in points() {
        let difference = (p.batman_rule() - p.batman).abs();
        assert!(difference < BATMAN_TOLERANCE, "{difference:e} at {p:?}");
    }
}

// The midpoint rule's error falls as 1 / samples^2. With 30 samples it is
// within 7e-6 of the exposure average for every set and cadence (measured
// worst 6.9e-6, for the eclipsing binary at 30 minutes), and for every
// planet (k up to 0.2) within 3e-6.
#[test]
fn converges_to_the_exposure_average() {
    for ((set, cadence, samples), (ours, _)) in worst_errors() {
        if samples == 30 {
            assert!(ours < 1e-5, "set {set}, {cadence} s: {ours:e}");
        }
    }
}

// Doubling the samples from 15 to 30 cuts the error by about four, wherever
// the error is large enough to measure above the reference's own (about
// 1e-8).
#[test]
fn error_falls_as_one_over_samples_squared() {
    let worst = worst_errors();
    let mut checked = 0;
    for (&(set, cadence, samples), &(at_15, _)) in &worst {
        if samples != 15 || at_15 < 1e-6 {
            continue;
        }
        let (at_30, _) = worst[&(set, cadence, 30)];
        let ratio = at_15 / at_30;
        assert!(
            (3.0..5.5).contains(&ratio),
            "set {set}, {cadence} s: ratio {ratio}"
        );
        checked += 1;
    }
    assert!(checked > 10, "only {checked} cases");
}

// At the same number of samples, the midpoint rule is the more accurate,
// wherever batman's error is large enough to measure: by 2.6 to 35 times.
#[test]
fn more_accurate_than_batman_at_the_same_samples() {
    let mut checked = 0;
    for ((set, cadence, samples), (ours, batman)) in worst_errors() {
        if samples == 1 || batman < 1e-6 {
            continue;
        }
        assert!(
            ours < batman / 2.0,
            "set {set}, {cadence} s, {samples} samples: {ours:e} vs batman {batman:e}"
        );
        checked += 1;
    }
    assert!(checked > 100, "only {checked} cases");
}

// batman's error falls only as 1 / samples: doubling the samples from 15 to
// 30 roughly halves it, where the midpoint rule's falls by four.
#[test]
fn batman_error_falls_as_one_over_samples() {
    let worst = worst_errors();
    let mut checked = 0;
    for (&(set, cadence, samples), &(_, at_15)) in &worst {
        if samples != 15 || at_15 < 1e-6 {
            continue;
        }
        let (_, at_30) = worst[&(set, cadence, 30)];
        let ratio = at_15 / at_30;
        assert!(
            (1.6..2.5).contains(&ratio),
            "set {set}, {cadence} s: ratio {ratio}"
        );
        checked += 1;
    }
    assert!(checked > 10, "only {checked} cases");
}

// PyTransit finds the planet's position from a series approximation to the
// orbit, which loses accuracy on tight orbits and for eclipsing binaries:
// its flux is off by up to 4.5e-6 even with no averaging, while batman agrees
// with this crate.
#[test]
fn pytransit_orbit_is_approximate_for_some_orbits() {
    for set in PYTRANSIT_ORBIT_APPROXIMATE {
        let unbinned: Vec<&Point> = points()
            .iter()
            .filter(|p| p.set == set && p.samples == 1)
            .collect();
        let worst = |f: fn(&Point) -> f64| {
            unbinned
                .iter()
                .map(|p| (p.params().flux_at(p.time()) - f(p)).abs())
                .fold(0.0, f64::max)
        };
        assert!(worst(|p| p.pytransit) > 5e-7, "set {set}");
        assert!(worst(|p| p.batman) < BATMAN_TOLERANCE, "set {set}");
    }
}
