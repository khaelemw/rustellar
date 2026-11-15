// B0, B1 and B2 on their own, against the exact values on and near every
// line where the formulas change case. The flux tests check the same values
// through the limb-darkening sum; this names which moment is wrong.

use std::f64::consts::PI;

use crate::{
    analytic::{blocked_mu, blocked_mu_centre_line, blocked_r2},
    uniform::blocked_area,
};

const CSV: &str = include_str!("../../tests/reference/exact_lines.csv");

// Measured worst error of each moment over max(1, k): 3.9e-16, a little
// under 2 units in the last place of 1. The error grows in proportion to k
// above k = 1, where the planet's own terms are of size k.
const TOLERANCE: f64 = 5e-16;

struct Row {
    k: f64,
    z: f64,
    exact: [f64; 3],
}

fn rows() -> Vec<Row> {
    let rows: Vec<Row> = CSV
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| {
            let v: Vec<f64> = line
                .split(',')
                .map(|field| field.parse().unwrap_or_else(|_| panic!("{line}")))
                .collect();
            assert_eq!(v.len(), 8, "{line}");
            Row {
                k: v[0],
                z: v[1],
                exact: [v[4], v[5], v[6]],
            }
        })
        .collect();
    assert_eq!(rows.len(), 3974);
    rows
}

// The moments as `flux` uses them, with its two early returns.
fn moments(z: f64, p: f64) -> [f64; 3] {
    if z >= 1.0 + p {
        return [0.0; 3];
    }
    if p > 1.0 && z <= p - 1.0 {
        return [1.0, 2.0 / 3.0, 0.5];
    }
    [blocked_area(z, p), blocked_mu(z, p), blocked_r2(z, p)]
}

#[test]
fn moments_match_exact_values_on_and_near_case_lines() {
    for row in rows() {
        let got = moments(row.z, row.k);
        for (i, (got, exact)) in got.into_iter().zip(row.exact).enumerate() {
            let error = (got - exact).abs();
            assert!(
                error <= TOLERANCE * row.k.max(1.0),
                "B{i} at k = {}, z = {}: {got} vs {exact}, error {error:e}",
                row.k,
                row.z
            );
        }
    }
}

// Where the planet misses the star exactly, nothing is blocked, not even a
// rounding error's worth: the rows just outside first contact, and outside
// it by 1 to 10 rounding steps.
#[test]
fn no_overlap_blocks_nothing() {
    let mut checked = 0;
    for row in rows().into_iter().filter(|row| row.exact == [0.0; 3]) {
        assert_eq!(
            moments(row.z, row.k),
            [0.0; 3],
            "k = {}, z = {}",
            row.k,
            row.z
        );
        checked += 1;
    }
    assert!(checked > 200, "only {checked} rows");
}

// Next to the line z = p the general formulas run right up to a relative
// distance of 1e-100, then the value on the line takes over. Both sides agree
// with it to rounding, however close.
#[test]
fn b1_is_continuous_across_z_equals_p() {
    for p in [1e-6, 0.1, 0.4999, 0.5001, 0.9, 1.0, 3.0, 1e4] {
        let on_line = blocked_mu_centre_line(p);
        for distance in [1e-8, 1e-15, 1e-50, 1.01e-100, 0.99e-100, 1e-200] {
            for z in [p * (1.0 + distance), p * (1.0 - distance)] {
                let got = blocked_mu(z, p);
                assert!(
                    (got - on_line).abs() <= 4e-16 * p.max(1.0) + p * distance,
                    "p = {p}, z = {z}: {got} vs {on_line}"
                );
            }
        }
    }
}

// A planet the size of the star almost exactly on its centre leaves a sliver
// of width z uncovered. The old formulas returned NaN at z = 5e-324.
#[test]
fn equal_disks_nearly_concentric() {
    for z in [5e-324, 1e-300, 1e-100, 1e-16] {
        let [b0, b1, b2] = moments(z, 1.0);
        // The uncovered sliver has area about 4 z, and weight about z^(3/2)
        // against mu.
        assert!(
            (b0 - (1.0 - 4.0 * z / PI)).abs() <= 2e-16,
            "B0 at {z}: {b0}"
        );
        assert!((b1 - 2.0 / 3.0).abs() <= 2e-16, "B1 at {z}: {b1}");
        assert!((b2 - 0.5).abs() <= 2e-16 + 4.0 * z, "B2 at {z}: {b2}");
    }
}
