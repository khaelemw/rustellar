// Compares exposure-averaged light curves with batman (Kreidberg 2015) and
// PyTransit (Parviainen 2015) for four transits and eight exposures. The data
// file describes how it was made in its header comments.
//
// The two packages place their samples differently. PyTransit uses the middle
// of equal slices of the exposure (the midpoint rule of Kipping 2010), as this
// crate does; batman spreads its samples from edge to edge. These tests check
// that this crate matches PyTransit sample for sample, that batman's rule
// applied to this crate's flux reproduces batman, and that the midpoint rule
// is the more accurate of the two.

use std::{collections::BTreeMap, num::NonZeroUsize};

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

// ==========================
// Helper Functions
// ==========================

const CSV: &str = include_str!("reference/exposure_points.csv");

const T0: f64 = 3_000.123_456;

// (k, b, a_rs, period), as in the data file's header.
const SETS: [(f64, f64, f64, f64); 4] = [
    (0.1, 0.3, 10.0, 3.0),
    (0.1, 0.95, 10.0, 3.0),
    (0.02, 0.5, 20.0, 10.0),
    (0.2, 0.0, 4.0, 1.0),
];

// The set with a_rs = 4, where PyTransit's series approximation to the orbit
// is off by about 4.5e-6 in flux (see the last test).
const TIGHT_ORBIT: usize = 3;

// Measured worst difference from PyTransit with the same samples, away from
// the tight orbit: 3.2e-8, the error of PyTransit's own flux.
const PYTRANSIT_TOLERANCE: f64 = 5e-8;

// Measured worst difference between batman and batman's rule applied to this
// crate's flux: 1.0e-8, the error of batman's own flux.
const BATMAN_TOLERANCE: f64 = 2e-8;

#[derive(Debug, Clone, Copy)]
struct Point {
    set: usize,
    exposure: f64,
    samples: usize,
    t: f64,
    batman: f64,
    pytransit: f64,
    // PyTransit with 2001 samples: the exposure average with negligible
    // sampling error.
    dense: f64,
}

impl Point {
    fn params(&self) -> TransitParams {
        let (k, b, a_rs, period) = SETS[self.set];
        TransitParams::new(
            Epoch::from_btjd(T0).unwrap(),
            TimeDelta::from_days(period).unwrap(),
            RadiusRatio::new(k).unwrap(),
            ScaledSemiMajorAxis::new(a_rs).unwrap(),
            ImpactParameter::new(b).unwrap(),
            LimbDarkening::quadratic(0.4, 0.25).unwrap(),
        )
        .unwrap()
    }

    fn time(&self) -> Epoch {
        Epoch::from_btjd(self.t).unwrap()
    }

    fn binned(&self) -> f64 {
        let exposure = Exposure::new(
            TimeDelta::from_days(self.exposure).unwrap(),
            NonZeroUsize::new(self.samples).unwrap(),
        )
        .unwrap();
        self.params().binned_flux_at(self.time(), exposure)
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
}

fn points() -> Vec<Point> {
    CSV.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| {
            let v: Vec<f64> = line
                .split(',')
                .map(|field| field.parse().unwrap_or_else(|_| panic!("{line}")))
                .collect();
            assert_eq!(v.len(), 7, "{line}");
            Point {
                set: v[0] as usize,
                exposure: v[1],
                samples: v[2] as usize,
                t: v[3],
                batman: v[4],
                pytransit: v[5],
                dense: v[6],
            }
        })
        .collect()
}

fn count_in_header() -> usize {
    CSV.lines()
        .next()
        .and_then(|line| line.strip_prefix("# "))
        .and_then(|line| line.split(' ').next())
        .and_then(|count| count.parse().ok())
        .expect("first line should be '# <count> ...'")
}

// Worst error against the dense average, per (set, exposure, samples), for
// this crate and for batman.
fn worst_errors() -> BTreeMap<(usize, u64, usize), (f64, f64)> {
    let mut worst = BTreeMap::new();
    for p in points().into_iter().filter(|p| p.set != TIGHT_ORBIT) {
        let entry = worst
            .entry((p.set, p.exposure.to_bits(), p.samples))
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
    for p in points().into_iter().filter(|p| p.set != TIGHT_ORBIT) {
        let difference = (p.binned() - p.pytransit).abs();
        assert!(difference < PYTRANSIT_TOLERANCE, "{difference:e} at {p:?}");
        checked += 1;
    }
    assert!(checked > 1000, "only {checked} points");
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

// The midpoint rule's error falls as 1 / samples^2. Measured worst errors for
// Kepler long cadence (29.4 minutes): 4.7e-6 at 15 samples and 1.0e-6 at 30.
#[test]
fn converges_to_the_exposure_average() {
    for ((set, exposure, samples), (ours, _)) in worst_errors() {
        let limit = match samples {
            15 => 6e-6,
            30 => 1.5e-6,
            _ => continue,
        };
        assert!(
            ours < limit,
            "set {set}, exposure {}, {samples} samples: {ours:e}",
            f64::from_bits(exposure)
        );
    }
}

// At the same number of samples, the midpoint rule is the more accurate:
// 2.6 to 35 times closer to the exposure average than batman's rule.
#[test]
fn more_accurate_than_batman_at_the_same_samples() {
    for ((set, exposure, samples), (ours, batman)) in worst_errors() {
        if samples == 1 {
            continue;
        }
        assert!(
            ours < batman / 2.0,
            "set {set}, exposure {}, {samples} samples: {ours:e} vs batman {batman:e}",
            f64::from_bits(exposure)
        );
    }
}

// batman's error falls only as 1 / samples, and with two samples (both on the
// edges of the exposure) it is worse than not averaging at all.
#[test]
fn batman_with_two_samples_is_worse_than_none() {
    let worst = worst_errors();
    let kepler = (29.4_f64 / 1440.0).to_bits();
    let (none, _) = worst[&(0, kepler, 1)];
    let (_, batman_two) = worst[&(0, kepler, 2)];
    assert!(batman_two > none, "{batman_two:e} vs {none:e}");
}

// PyTransit finds the planet's position from a series approximation to the
// orbit, which loses accuracy on tight orbits: at a_rs = 4 its flux is off by
// about 4.5e-6 even with no averaging, while batman agrees with this crate.
#[test]
fn pytransit_orbit_is_approximate_for_tight_orbits() {
    let unbinned: Vec<Point> = points()
        .into_iter()
        .filter(|p| p.set == TIGHT_ORBIT && p.samples == 1)
        .collect();
    let worst = |f: fn(&Point) -> f64| {
        unbinned
            .iter()
            .map(|p| (p.params().flux_at(p.time()) - f(p)).abs())
            .fold(0.0, f64::max)
    };
    assert!(worst(|p| p.pytransit) > 1e-6);
    assert!(worst(|p| p.batman) < BATMAN_TOLERANCE);
}
