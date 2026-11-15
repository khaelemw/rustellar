// Compares `flux` with reference values at about 28,000 random points spread
// evenly over the whole valid range: k from 1e-4 to 100, every separation
// across the transit and some outside it, and limb darkening from Kipping's
// (2013) physical prior plus the rest of the allowed range. The data file
// describes how it was made in its header comments.
//
// Two references: the exact flux by quadrature, with an estimate of its own
// error, and exoplanet-core (Agol, Luger & Foreman-Mackey 2020), the most
// precise published method. The tolerances are this crate's measured error
// in each range of k, with room; they are a record of how accurate `flux` is,
// so a change that makes it worse anywhere fails here.

use rustellar_model::{LimbDarkening, RadiusRatio, flux};

// ==========================
// Helper Functions
// ==========================

const CSV: &str = include_str!("reference/flux_random.csv");

#[derive(Debug, Clone, Copy)]
struct Point {
    k: f64,
    z: f64,
    u1: f64,
    u2: f64,
    exact: f64,
    // Half the difference between two quadratures: an estimate of the error
    // in `exact`.
    spread: f64,
    agol: f64,
}

impl Point {
    fn flux(&self) -> f64 {
        flux(
            self.z,
            RadiusRatio::new(self.k).unwrap(),
            LimbDarkening::quadratic(self.u1, self.u2).unwrap(),
        )
    }
}

// The allowed difference from exoplanet-core for each range of k, from the
// measured worst cases: 9.2e-14 below 1, 1.4e-12 to 10, 1.3e-11 to 20,
// 3.3e-11 to 30, and 1.5e-9 to 100 (where the error grows as k^3).
fn agol_tolerance(k: f64) -> f64 {
    match k {
        k if k < 1.0 => 3e-13,
        k if k < 10.0 => 5e-12,
        k if k < 20.0 => 5e-11,
        k if k < 30.0 => 1e-10,
        _ => 3e-9,
    }
}

// The quadrature is sometimes off by up to 1.5e-12 even where its two forms
// agree, so this is added to its estimated error.
const QUADRATURE_FLOOR: f64 = 3e-12;

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
                k: v[0],
                z: v[1],
                u1: v[2],
                u2: v[3],
                exact: v[4],
                spread: v[5],
                agol: v[6],
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

// ==========================
// Tests
// ==========================

#[test]
fn reference_file_is_complete() {
    assert_eq!(points().len(), count_in_header());
}

// The sample really does cover the range: every decade of k, both sides of
// the limb-darkening prior, and points outside the transit.
#[test]
fn reference_covers_the_whole_range() {
    let points = points();
    for decade in -4..2 {
        let lo = 10_f64.powi(decade);
        let count = points
            .iter()
            .filter(|p| p.k >= lo && p.k < 10.0 * lo)
            .count();
        assert!(count > 3000, "k in [{lo}, {}): {count}", 10.0 * lo);
    }
    assert!(points.iter().any(|p| p.k > 99.0));
    assert!(points.iter().filter(|p| p.u1 < 0.0).count() > 1000);
    assert!(points.iter().filter(|p| p.u1 + p.u2 > 0.99).count() > 10);
    assert!(points.iter().filter(|p| p.z >= 1.0 + p.k).count() > 300);
    assert!(points.iter().filter(|p| p.exact == 0.0).count() > 10);
}

#[test]
fn matches_agol_everywhere() {
    for p in points() {
        let difference = (p.flux() - p.agol).abs();
        assert!(difference < agol_tolerance(p.k), "{difference:e} at {p:?}");
    }
}

#[test]
fn matches_exact_values_everywhere() {
    for p in points() {
        let difference = (p.flux() - p.exact).abs();
        let tolerance = agol_tolerance(p.k) + p.spread + QUADRATURE_FLOOR;
        assert!(difference < tolerance, "{difference:e} at {p:?}");
    }
}

// Outside the transit, and when the planet covers the whole star, the flux is
// exactly 1 and 0, not merely close.
#[test]
fn exact_outside_the_transit_and_under_full_cover() {
    let mut checked = 0;
    for p in points()
        .into_iter()
        .filter(|p| p.exact == 0.0 || p.exact == 1.0)
    {
        assert_eq!(p.flux(), p.exact, "{p:?}");
        checked += 1;
    }
    assert!(checked > 300, "only {checked} points");
}

// The two references agree with each other, which checks the reference data
// itself: two unrelated methods. exoplanet-core loses some precision at large
// k too (2.6e-10 at k = 100).
#[test]
fn references_agree_with_each_other() {
    for p in points() {
        let tolerance = if p.k < 30.0 { 2e-11 } else { 5e-10 };
        let difference = (p.agol - p.exact).abs();
        assert!(
            difference < tolerance + p.spread + QUADRATURE_FLOOR,
            "{difference:e} at {p:?}"
        );
    }
}
