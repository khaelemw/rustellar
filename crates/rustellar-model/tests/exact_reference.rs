// Compares `flux` with the exact flux at 36,742 points: 3,974 on and near
// every line where the formulas change case, for k from 1e-6 to 1e4, and
// 32,768 random points over the whole range. The data files describe how the
// exact values were made in their header comments.

use rustellar_model::{LimbDarkening, RadiusRatio, flux};

// ==========================
// Helper Functions
// ==========================

const LINES: &str = include_str!("reference/exact_lines.csv");
const RANDOM: &str = include_str!("reference/exact_random.csv");

// Measured worst error over max(1, k): 6.7e-16 for k below 1, and between
// 1.4e-16 and 4.6e-16 for k above 1, where the planet's own terms are of size k.
const TOLERANCE: f64 = 1e-15;

#[derive(Debug, Clone, Copy)]
struct Point {
    k: f64,
    z: f64,
    u1: f64,
    u2: f64,
    exact: f64,
}

impl Point {
    fn flux(&self) -> f64 {
        flux(
            self.z,
            RadiusRatio::new(self.k).unwrap(),
            LimbDarkening::quadratic(self.u1, self.u2).unwrap(),
        )
    }

    fn error(&self) -> f64 {
        (self.flux() - self.exact).abs()
    }

    fn tolerance(&self) -> f64 {
        TOLERANCE * self.k.max(1.0)
    }
}

fn points(csv: &str) -> Vec<Point> {
    let points: Vec<Point> = csv
        .lines()
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
                exact: v[7],
            }
        })
        .collect();
    assert_eq!(points.len(), count_in_header(csv), "file is incomplete");
    points
}

fn count_in_header(csv: &str) -> usize {
    csv.lines()
        .next()
        .and_then(|line| line.strip_prefix("# "))
        .and_then(|line| line.split(' ').next())
        .and_then(|count| count.parse().ok())
        .expect("first line should be '# <count> ...'")
}

fn assert_all_within_tolerance(points: &[Point], what: &str) {
    for point in points {
        let error = point.error();
        assert!(
            error <= point.tolerance(),
            "{what}: error {error:e} at {point:?}, flux {}",
            point.flux()
        );
    }
}

// The points of `points` whose k is within a factor of 1.0001 of `k`.
fn at_k(points: &[Point], k: f64) -> Vec<Point> {
    points
        .iter()
        .copied()
        .filter(|point| (point.k / k).ln().abs() < 1e-4)
        .collect()
}

// ==========================
// Tests
// ==========================

#[test]
fn matches_exact_values_on_and_near_case_lines() {
    assert_all_within_tolerance(&points(LINES), "case lines");
}

#[test]
fn matches_exact_values_at_random_points() {
    assert_all_within_tolerance(&points(RANDOM), "random");
}

// The line file must reach every line it claims to, for every k, at every
// distance; otherwise the test above could pass with whole cases missing.
#[test]
fn case_lines_are_covered_at_every_distance() {
    let lines = points(LINES);
    let mut ks: Vec<f64> = lines.iter().map(|point| point.k).collect();
    ks.sort_by(f64::total_cmp);
    ks.dedup();
    assert!(ks.len() >= 33, "only {} values of k", ks.len());
    assert_eq!(ks.first(), Some(&1e-6));
    assert_eq!(ks.last(), Some(&1e4));

    for k in [1e-6, 0.1, 0.5, 1.0, 2.0, 1e4] {
        let near_k = at_k(&lines, k);
        let mut case_lines = vec![k, (1.0 - k).abs(), 1.0 + k];
        if k > 1.0 {
            case_lines.push(k - 1.0);
        }
        for line in case_lines {
            // On the line, within 10 rounding steps either side, and 1e-15 to
            // 0.1 away.
            let on = near_k.iter().any(|point| point.z == line);
            // (Next to z = 0 the rounding steps are subnormal.)
            let ten_steps = (10.0 * line * f64::EPSILON).max(10.0 * 5e-324);
            let ulps = near_k
                .iter()
                .filter(|point| point.z != line && (point.z - line).abs() <= ten_steps)
                .count();
            let far = near_k
                .iter()
                .any(|point| ((point.z - line).abs() - 0.1).abs() < 1e-9);
            assert!(
                on && ulps >= 4 && far,
                "k = {k}, line z = {line}: on {on}, {ulps} within 10 ulps, far {far}"
            );
        }
        assert!(near_k.iter().any(|point| point.z == 0.0), "k = {k}: z = 0");
    }
}

// The random points must spread over the whole range of k, inside the disk
// and across the limb, so that no region is left to the line points alone.
// (Outside the transit the flux is exactly 1; the line points check that from
// 1 rounding step beyond first contact.)
#[test]
fn random_points_cover_every_region() {
    let random = points(RANDOM);
    for decade in -6..4 {
        let (low, high) = (10f64.powi(decade), 10f64.powi(decade + 1));
        let in_decade: Vec<&Point> = random
            .iter()
            .filter(|p| (low..high).contains(&p.k))
            .collect();
        assert!(
            in_decade.len() > 2000,
            "k in [{low}, {high}): {}",
            in_decade.len()
        );
        let inside = in_decade.iter().filter(|p| p.z + p.k < 1.0).count();
        let partial = in_decade
            .iter()
            .filter(|p| (p.k - 1.0).abs() < p.z && p.z < 1.0 + p.k)
            .count();
        assert!(partial > 100, "k in [{low}, {high}): {partial} partial");
        if high <= 1.0 {
            assert!(inside > 100, "k in [{low}, {high}): {inside} inside");
        }
    }
}

// The flux is exactly 1 when the planet misses the star, and exactly 0 when a
// planet larger than the star covers it, whatever the limb darkening.
#[test]
fn exact_where_the_answer_is_exact() {
    let mut checked = 0;
    for point in points(LINES).iter().chain(&points(RANDOM)) {
        let exact_answer = point.exact == 1.0 || point.exact == 0.0;
        let outside = point.z >= 1.0 + point.k;
        let covered = point.k > 1.0 && point.z <= point.k - 1.0;
        if exact_answer && (outside || covered) {
            assert_eq!(point.flux(), point.exact, "{point:?}");
            checked += 1;
        }
    }
    assert!(checked > 1000, "only {checked} points");
}
