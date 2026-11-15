// Compares `flux` with reference values at 1,932 separations: the exact flux
// from the defining integral, and the analytic models of batman (Kreidberg
// 2015), PyTransit (Parviainen 2015) and exoplanet-core (Agol, Luger &
// Foreman-Mackey 2020). The data file describes how it was made in its
// header comments.
//
// The exact values decide what is correct. The other models are compared away
// from the lines where Mandel & Agol's formula changes case; on those lines
// each is measurably wrong somewhere, and the last tests record where.

use rustellar_model::{LimbDarkening, RadiusRatio, flux};

// ==========================
// Helper Functions
// ==========================

const CSV: &str = include_str!("reference/flux_points.csv");

// Measured worst difference from the exact values: 1.5e-11.
const EXACT_TOLERANCE: f64 = 1e-10;

// Measured worst differences of batman and PyTransit from the exact values,
// at least 1e-3 from a case line: 4.2e-7, at k = 20. For planet-sized k
// (up to 0.3) both are within 5.3e-8.
const OTHER_MODEL_TOLERANCE: f64 = 1e-6;
const PLANET_TOLERANCE: f64 = 1e-7;

// exoplanet-core is accurate to about 1e-13 away from case lines, so the
// difference from it is our own error: at worst 1.4e-11, within 1e-6 of
// z = k, where this crate interpolates.
const AGOL_TOLERANCE: f64 = 3e-11;

// Measured worst difference between the exact values and exoplanet-core,
// at least 1e-3 from a case line: 1.0e-13. Two independent methods agreeing
// this closely is the check on the exact values themselves.
const EXACT_AGREES_WITH_AGOL: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
struct Point {
    k: f64,
    z: f64,
    u1: f64,
    u2: f64,
    exact: f64,
    batman: f64,
    // NaN where PyTransit divides by zero.
    pytransit: f64,
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

    // Distance from the nearest line where the formula changes case.
    fn distance_from_case_line(&self) -> f64 {
        [self.k, (1.0 - self.k).abs(), 1.0 + self.k]
            .into_iter()
            .map(|line| (self.z - line).abs())
            .fold(f64::INFINITY, f64::min)
    }

    // Below z = k by less than 1e-15: where exoplanet-core fails.
    fn just_below_centre_line(&self) -> bool {
        self.z < self.k && self.k - self.z < 1e-15
    }

    fn on_inner_contact(&self) -> bool {
        self.k < 1.0 && self.z == 1.0 - self.k
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
            assert_eq!(v.len(), 8, "{line}");
            Point {
                k: v[0],
                z: v[1],
                u1: v[2],
                u2: v[3],
                exact: v[4],
                batman: v[5],
                pytransit: v[6],
                agol: v[7],
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

fn assert_within(points: impl Iterator<Item = (Point, f64)>, tolerance: f64, what: &str) {
    let mut checked = 0;
    for (point, other) in points {
        let difference = (point.flux() - other).abs();
        assert!(
            difference < tolerance,
            "{what}: {difference:e} at {point:?}"
        );
        checked += 1;
    }
    assert!(checked > 100, "{what}: only {checked} points");
}

// ==========================
// Tests
// ==========================

#[test]
fn reference_file_is_complete() {
    assert_eq!(points().len(), count_in_header());
}

#[test]
fn matches_exact_values() {
    assert_within(
        points().into_iter().map(|p| (p, p.exact)),
        EXACT_TOLERANCE,
        "exact",
    );
}

// The lines where the formula changes case, on and within 1e-5 of them, are
// where analytic implementations go wrong; covered by the test above, but
// checked here on their own so a failure names the cause.
#[test]
fn matches_exact_values_on_and_near_case_lines() {
    assert_within(
        points()
            .into_iter()
            .filter(|p| p.distance_from_case_line() <= 1.01e-5)
            .map(|p| (p, p.exact)),
        EXACT_TOLERANCE,
        "exact near case lines",
    );
}

#[test]
fn agrees_with_batman_away_from_case_lines() {
    assert_within(
        points()
            .into_iter()
            .filter(|p| p.distance_from_case_line() > 1e-6)
            .map(|p| (p, p.batman)),
        OTHER_MODEL_TOLERANCE,
        "batman",
    );
}

#[test]
fn agrees_with_pytransit_away_from_case_lines() {
    assert_within(
        points()
            .into_iter()
            .filter(|p| p.distance_from_case_line() > 1e-3 && !p.pytransit.is_nan())
            .map(|p| (p, p.pytransit)),
        OTHER_MODEL_TOLERANCE,
        "pytransit",
    );
}

#[test]
fn agrees_with_both_for_planet_sized_k() {
    let planets = || {
        points()
            .into_iter()
            .filter(|p| p.k <= 0.3 && p.distance_from_case_line() > 1e-3)
    };
    assert_within(planets().map(|p| (p, p.batman)), PLANET_TOLERANCE, "batman");
    assert_within(
        planets().map(|p| (p, p.pytransit)),
        PLANET_TOLERANCE,
        "pytransit",
    );
}

// At second and third contact (z = 1 - k exactly) with k > 1/2, batman's flux
// is far off, even negative, and PyTransit divides by zero or is far off.
// This crate matches the exact value there (checked above).
#[test]
fn batman_and_pytransit_are_wrong_at_inner_contact() {
    let on_line: Vec<Point> = points()
        .into_iter()
        .filter(|p| p.on_inner_contact() && p.k > 0.5)
        .collect();
    assert!(on_line.len() >= 4, "only {} points", on_line.len());

    for p in &on_line {
        assert!((p.batman - p.exact).abs() > 0.5, "batman {p:?}");
        assert!(p.batman < 0.0, "batman not negative: {p:?}");
        let pytransit_off = p.pytransit.is_nan() || (p.pytransit - p.exact).abs() > 0.05;
        assert!(pytransit_off, "pytransit {p:?}");
    }
}

// For a very small planet near the limb, batman and PyTransit agree with each
// other but not with the exact value: both give about 12 times the blocked
// light (5e-8 in flux, below any measurement noise).
#[test]
fn batman_and_pytransit_overstate_tiny_planets_near_limb() {
    let p = points()
        .into_iter()
        .find(|p| p.k == 1e-4 && p.z == 0.9989 && p.u1 == 0.4)
        .expect("point k = 1e-4, z = 0.9989");

    let exact_blocked = 1.0 - p.exact;
    assert!((1.0 - p.batman) / exact_blocked > 10.0, "{p:?}");
    assert!((p.batman - p.pytransit).abs() < 1e-15, "{p:?}");
    assert!((p.flux() - p.exact).abs() < 1e-14, "{p:?}");
}

// The exact values are not from this crate's code, but they are all from one
// integrand. exoplanet-core is a fully independent analytic method, so its
// agreement here checks the reference data itself.
#[test]
fn exact_values_agree_with_agol() {
    let away: Vec<Point> = points()
        .into_iter()
        .filter(|p| p.distance_from_case_line() > 1e-3)
        .collect();
    assert!(away.len() > 500, "only {} points", away.len());
    for p in away {
        assert!((p.exact - p.agol).abs() < EXACT_AGREES_WITH_AGOL, "{p:?}");
    }
}

// Agol, Luger & Foreman-Mackey (2020) is the most precise published method, so
// matching it everywhere it is right checks this crate against the literature.
#[test]
fn agrees_with_agol_off_its_failing_points() {
    assert_within(
        points()
            .into_iter()
            .filter(|p| !p.just_below_centre_line())
            .map(|p| (p, p.agol)),
        AGOL_TOLERANCE,
        "agol",
    );
}

// One step below z = k (the planet's edge just short of the star's centre),
// exoplanet-core leaves out the 2/3 that B1 gains when the planet covers the
// centre, so its flux is off by (2/3) c2 / disk_integral: 0.73 for the
// Sun-like law. This crate matches the exact value there (checked above).
#[test]
fn agol_is_wrong_one_step_below_centre_line() {
    let failing: Vec<Point> = points()
        .into_iter()
        .filter(Point::just_below_centre_line)
        .collect();
    assert!(!failing.is_empty());
    for p in failing {
        let c2 = p.u1 + 2.0 * p.u2;
        let disk_integral = 1.0 - p.u1 / 3.0 - p.u2 / 6.0;
        let missing = 2.0 / 3.0 * c2 / disk_integral;
        assert!((p.exact - p.agol - missing).abs() < 1e-12, "{p:?}");
        assert!((p.flux() - p.exact).abs() < 1e-14, "{p:?}");
    }
}
