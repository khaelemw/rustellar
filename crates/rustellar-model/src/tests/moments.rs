// B0, B1 and B2 on their own: continuity across z = p and nearly concentric
// equal disks. (Against the exact values on and near every case line, each
// moment through its own limb-darkening law: the integration tests.)

use std::f64::consts::PI;

use crate::{
    analytic::{blocked_mu, blocked_mu_centre_line, blocked_r2},
    uniform::blocked_area,
};

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
