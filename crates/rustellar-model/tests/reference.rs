// Compares the light curve with reference values from batman (Kreidberg
// 2015) over a grid of 840 transits. The data files describe how they were
// made in their header comments.

use std::sync::OnceLock;

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

// ==========================
// Helper Functions
// ==========================

const SETS_CSV: &str = include_str!("reference/batman_sets.csv");
const POINTS_CSV: &str = include_str!("reference/batman_points.csv");

// The largest difference from batman allowed at any point. The measured
// worst case is about 2e-8, and comes from batman: the ring sum moves by less
// than 3e-9 there when its ring count is raised to 100,000.
const TOLERANCE: f64 = 1e-7;

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
            Epoch::from_btjd(self.t0).unwrap(),
            TimeDelta::from_days(self.period).unwrap(),
            RadiusRatio::new(self.k).unwrap(),
            ScaledSemiMajorAxis::new(self.a_rs).unwrap(),
            ImpactParameter::new(self.b).unwrap(),
            LimbDarkening::quadratic(self.u1, self.u2).unwrap(),
        )
        .unwrap()
    }
}

// One point: batman's flux and this crate's flux at the same time.
#[derive(Debug, Clone, Copy)]
struct Point {
    set: usize,
    t: f64,
    batman: f64,
    model: f64,
}

impl Point {
    fn difference(&self) -> f64 {
        (self.model - self.batman).abs()
    }
}

// The count in the first comment line, for example "# 840 parameter sets ...".
fn count_in_header(csv: &str) -> usize {
    csv.lines()
        .next()
        .and_then(|line| line.strip_prefix("# "))
        .and_then(|line| line.split(' ').next())
        .and_then(|count| count.parse().ok())
        .expect("first line should be '# <count> ...'")
}

// The data rows, without comment lines or the column names, split on commas.
fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

fn number(field: &str) -> f64 {
    field
        .parse()
        .unwrap_or_else(|_| panic!("not a number: {field}"))
}

fn sets() -> Vec<Set> {
    rows(SETS_CSV)
        .enumerate()
        .map(|(index, row)| {
            assert_eq!(row.len(), 9, "set row {index}: {row:?}");
            assert_eq!(row[0].parse::<usize>().unwrap(), index, "sets out of order");

            Set {
                k: number(row[1]),
                b: number(row[2]),
                a_rs: number(row[3]),
                period: number(row[5]),
                t0: number(row[6]),
                u1: number(row[7]),
                u2: number(row[8]),
            }
        })
        .collect()
}

// Every point with the model's flux beside batman's, computed once and shared
// by the tests below.
fn points() -> &'static [Point] {
    static POINTS: OnceLock<Vec<Point>> = OnceLock::new();

    POINTS.get_or_init(|| {
        let sets = sets();

        // (set, time, batman flux) for each row, grouped by set below so each
        // set's light curve is one call.
        let mut by_set: Vec<Vec<(f64, f64)>> = vec![Vec::new(); sets.len()];
        for row in rows(POINTS_CSV) {
            assert_eq!(row.len(), 3, "point row: {row:?}");
            let set: usize = row[0].parse().unwrap();
            by_set[set].push((number(row[1]), number(row[2])));
        }

        let mut points = Vec::new();
        for (set, (params, rows)) in sets.iter().map(Set::params).zip(&by_set).enumerate() {
            let times: Vec<Epoch> = rows
                .iter()
                .map(|&(t, _)| Epoch::from_btjd(t).unwrap())
                .collect();
            let model = params.light_curve(&times);

            for (&(t, batman), model) in rows.iter().zip(model) {
                points.push(Point {
                    set,
                    t,
                    batman,
                    model,
                });
            }
        }

        points
    })
}

fn worst(points: impl Iterator<Item = Point>) -> Point {
    points
        .max_by(|a, b| a.difference().total_cmp(&b.difference()))
        .expect("no points")
}

// ==========================
// Tests
// ==========================

// Catches a half-written or hand-edited file before the comparison runs.
#[test]
fn reference_files_are_complete() {
    let sets = sets();

    assert_eq!(sets.len(), count_in_header(SETS_CSV));
    assert_eq!(points().len(), count_in_header(POINTS_CSV));
    assert!(points().iter().all(|p| p.set < sets.len()));
}

#[test]
fn light_curve_matches_batman() {
    let worst = worst(points().iter().copied());
    let set = sets()[worst.set];

    // Seen with `cargo test -- --nocapture`.
    println!(
        "worst difference from batman: {:.2e} at t - t0 = {:+.6} d in {set:?}",
        worst.difference(),
        worst.t - set.t0,
    );

    assert!(
        worst.difference() < TOLERANCE,
        "{:.2e} at t = {} in {set:?}",
        worst.difference(),
        worst.t,
    );
}

// Where batman gives exactly 1 the planet is clear of the star, and this crate
// returns exactly 1 there too. (At first and fourth contact batman gives up to
// 4e-10 above 1, so those points are not exactly 1 and are not checked here.)
#[test]
fn flux_is_exactly_one_where_batman_is() {
    let out_of_transit: Vec<&Point> = points().iter().filter(|p| p.batman == 1.0).collect();

    assert!(
        out_of_transit.len() > 1000,
        "too few: {}",
        out_of_transit.len()
    );
    for p in out_of_transit {
        assert_eq!(p.model, 1.0, "{p:?}");
    }
}

// b = 1 - k: the planet touches the limb from inside at mid-transit, a
// special case of the analytic formula. batman warns that one of its
// elliptic integrals does not converge here, but its flux is still accurate.
#[test]
fn limb_touching_transits_match_batman() {
    let sets = sets();
    let touching = |p: &&Point| {
        let set = sets[p.set];
        (set.b - (1.0 - set.k)).abs() < 1e-12
    };

    let count = points().iter().filter(touching).count();
    assert!(count > 0, "no b = 1 - k sets in the data");

    let worst = worst(points().iter().filter(touching).copied());
    assert!(worst.difference() < TOLERANCE, "{worst:?}");
}
