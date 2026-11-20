// Compares light curves with references from two independent codes over 1,984
// transits: batman's orbit geometry (Kreidberg 2015) and exoplanet-core's
// flux (Agol, Luger & Foreman-Mackey 2020) at batman's separation. The grid
// runs from small planets to eclipsing binaries (k up to 10), with central,
// inner-contact, grazing and missing transits, a_rs from 1.5 to 300, and half
// the sets observed about 1,400 orbits from t0. The data files describe how
// they were made in their header comments.

use std::sync::OnceLock;

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams, flux,
};

// ==========================
// Helper Functions
// ==========================

const SETS_CSV: &str = include_str!("reference/lightcurve_sets.csv");
const POINTS_CSV: &str = include_str!("reference/lightcurve_points.csv");

// batman's separation when the planet is behind the star.
const BEHIND: f64 = 100.0;

// Measured worst difference from batman's separation: 7.3e-11, from rounding
// times about 4,300 days from t0 (a few 1e-13 days, times the planet's speed
// across the sky of up to 600 stellar radii a day).
const SEPARATION_TOLERANCE: f64 = 2e-10;

// The flux at batman's separation against exoplanet-core: as for the random
// points (tests/flux_random.rs), at most 5e-12 for k up to 10.
const FLUX_TOLERANCE: f64 = 1e-11;

// The light curve against exoplanet-core, so with the separation's rounding
// included: measured worst 1.9e-11.
const LIGHT_CURVE_TOLERANCE: f64 = 5e-11;

#[derive(Debug, Clone, Copy)]
struct Set {
    k: f64,
    b: f64,
    a_rs: f64,
    period: f64,
    t0: f64,
    u1: f64,
    u2: f64,
}

impl Set {
    fn params(&self) -> TransitParams {
        TransitParams::new(
            Epoch::from_days(self.t0).unwrap(),
            TimeDelta::from_days(self.period).unwrap(),
            RadiusRatio::new(self.k).unwrap(),
            ScaledSemiMajorAxis::new(self.a_rs).unwrap(),
            ImpactParameter::new(self.b).unwrap(),
            LimbDarkening::quadratic(self.u1, self.u2).unwrap(),
        )
        .unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
struct Point {
    set: usize,
    t: f64,
    z: f64,
    batman: f64,
    agol: f64,
}

impl Point {
    fn set(&self) -> Set {
        sets()[self.set]
    }

    fn time(&self) -> Epoch {
        Epoch::from_days(self.t).unwrap()
    }

    fn behind(&self) -> bool {
        self.z == BEHIND
    }
}

fn data_rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

fn number(field: &str) -> f64 {
    field.parse().unwrap_or_else(|_| panic!("{field}"))
}

fn sets() -> &'static [Set] {
    static SETS: OnceLock<Vec<Set>> = OnceLock::new();
    SETS.get_or_init(|| {
        data_rows(SETS_CSV)
            .enumerate()
            .map(|(i, f)| {
                assert_eq!(f.len(), 9, "{f:?}");
                assert_eq!(number(f[0]) as usize, i);
                // f[2] is the name of the impact parameter's rule, e.g. "1+k/2".
                Set {
                    k: number(f[1]),
                    b: number(f[3]),
                    a_rs: number(f[4]),
                    period: number(f[5]),
                    t0: number(f[6]),
                    u1: number(f[7]),
                    u2: number(f[8]),
                }
            })
            .collect()
    })
}

fn points() -> &'static [Point] {
    static POINTS: OnceLock<Vec<Point>> = OnceLock::new();
    POINTS.get_or_init(|| {
        data_rows(POINTS_CSV)
            .map(|f| {
                assert_eq!(f.len(), 5, "{f:?}");
                Point {
                    set: number(f[0]) as usize,
                    t: number(f[1]),
                    z: number(f[2]),
                    batman: number(f[3]),
                    agol: number(f[4]),
                }
            })
            .collect()
    })
}

fn count_in_header(csv: &str) -> usize {
    csv.lines()
        .next()
        .and_then(|line| line.strip_prefix("# "))
        .and_then(|line| line.split(' ').next())
        .and_then(|count| count.parse().ok())
        .expect("first line should be '# <count> ...'")
}

// ==========================
// Tests
// ==========================

#[test]
fn reference_files_are_complete() {
    assert_eq!(sets().len(), count_in_header(SETS_CSV));
    assert_eq!(points().len(), count_in_header(POINTS_CSV));
}

#[test]
fn separation_matches_batman() {
    for p in points().iter().filter(|p| !p.behind()) {
        let ours = p.set().params().separation(p.time());
        let difference = (ours - p.z).abs();
        assert!(difference < SEPARATION_TOLERANCE, "{difference:e} at {p:?}");
    }
}

// Where batman puts the planet behind the star, so does this crate, and the
// flux is exactly 1; everywhere else the planet is in front.
#[test]
fn behind_the_star_matches_batman() {
    let mut behind = 0;
    for p in points() {
        let params = p.set().params();
        assert_eq!(params.in_front(p.time()), !p.behind(), "{p:?}");
        if p.behind() {
            assert_eq!(params.flux_at(p.time()), 1.0, "{p:?}");
            behind += 1;
        }
    }
    assert!(behind > 3000, "only {behind} points behind the star");
}

#[test]
fn flux_at_batman_separation_matches_agol() {
    for p in points().iter().filter(|p| !p.behind()) {
        let s = p.set();
        let ours = flux(
            p.z,
            RadiusRatio::new(s.k).unwrap(),
            LimbDarkening::quadratic(s.u1, s.u2).unwrap(),
        );
        let difference = (ours - p.agol).abs();
        assert!(difference < FLUX_TOLERANCE, "{difference:e} at {p:?}");
    }
}

#[test]
fn light_curve_matches_agol() {
    for p in points() {
        let ours = p.set().params().flux_at(p.time());
        let difference = (ours - p.agol).abs();
        assert!(
            difference < LIGHT_CURVE_TOLERANCE,
            "{difference:e} at {p:?}"
        );
    }
}

// Clear of the star (beyond 1e-9 of first or fourth contact) and behind it
// the flux is exactly 1, and under full cover by a body larger than the star
// exactly 0. Right at contact, rounding decides.
#[test]
fn exact_where_nothing_or_everything_is_covered() {
    let mut checked = 0;
    for p in points() {
        let k = p.set().k;
        let clear = p.behind() || p.z > 1.0 + k + 1e-9;
        let covered = p.z < k - 1.0 - 1e-9;
        if clear || covered {
            let expected = if clear { 1.0 } else { 0.0 };
            assert_eq!(p.set().params().flux_at(p.time()), expected, "{p:?}");
            checked += 1;
        }
    }
    assert!(checked > 10_000, "only {checked} points");
}

// batman agrees to its stated accuracy for planets (k up to 0.3), except in
// the transits whose mid-point lies exactly on a line where its formula
// fails (see tests/flux_reference.rs): touching the limb from inside
// (b = |1 - k|), or the planet's edge on the star's centre (b = k).
#[test]
fn agrees_with_batman_for_planets() {
    let mut checked = 0;
    for p in points() {
        let s = p.set();
        if s.k > 0.3 || s.b == (1.0 - s.k).abs() || s.b == s.k {
            continue;
        }
        let difference = (p.set().params().flux_at(p.time()) - p.batman).abs();
        assert!(difference < 5e-8, "{difference:e} at {p:?}");
        checked += 1;
    }
    assert!(checked > 30_000, "only {checked} points");
}
