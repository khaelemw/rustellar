// The oracle: noiseless transits found to rounding wherever they sit in
// the fold (across its ends too), and never beaten by a far denser grid.

use super::{Random, light_curve};
use crate::exact::best;
use crate::fit::{Points, Position, fit};
use crate::template::Template;

/// The distance between two mid-transit times on a period.
fn apart(a: f64, b: f64, period: f64) -> f64 {
    let d = (a - b).rem_euclid(period);
    d.min(period - d)
}

#[test]
fn noiseless_transits_are_found_to_rounding() {
    let template = Template::tls_default();
    let mut random = Random::new(1);
    // Mid-transit times anywhere in the fold, including at the first
    // point and just before it a period later (the window wraps).
    for (t0, period, duration) in [
        (1.3, 3.7, 0.13),
        (0.0, 2.0, 0.1),
        (3.7 - 1e-9, 3.7, 0.2),
        (0.01, 1.1, 0.05),
        (10.4, 7.3, 0.3),
    ] {
        let position = Position {
            period,
            duration,
            t0,
        };
        let points = light_curve(
            &mut random,
            &template,
            position,
            0.01,
            0.0,
            (0.0, 27.0, 1.0 / 720.0),
        );
        let found = best(&points, &template, period, duration, 50);
        assert!(apart(found.t0, t0, period) < 1e-8, "{t0}: {}", found.t0);
        assert!(
            (found.fit.depth / 0.01 - 1.0).abs() < 1e-9,
            "{}",
            found.fit.depth
        );
        assert!(found.t0 >= 0.0 && found.t0 <= period, "{}", found.t0);
    }
}

#[test]
fn never_beaten_by_a_far_denser_grid() {
    let template = Template::tls_default();
    let mut random = Random::new(2);
    for case in 0..8 {
        let period = random.between(0.7, 6.0);
        let duration = random.between(0.03, 0.1) * period;
        let t0 = random.between(0.0, 30.0);
        let position = Position {
            period,
            duration,
            t0,
        };
        let depth = random.between(0.0005, 0.003);
        let start = random.between(-5.0, 5.0);
        let points = light_curve(
            &mut random,
            &template,
            position,
            depth,
            1e-3,
            (start, 20.0, 1.0 / 144.0),
        );
        let found = best(&points, &template, period, duration, 20);
        // Every 1/500 of a duration over the whole period: 25 times the
        // oracle's grid, which its polishing must make up.
        let steps = (period / duration * 500.0).ceil();
        let mut dense = f64::INFINITY;
        for k in 0..steps as u32 {
            let t0 = points.time[0] + f64::from(k) * period / steps;
            let chi2 = fit(
                &points,
                &template,
                Position {
                    period,
                    duration,
                    t0,
                },
            )
            .chi2;
            dense = dense.min(chi2);
        }
        assert!(
            found.fit.chi2 <= dense * (1.0 + 1e-12),
            "case {case}: {} against {dense}",
            found.fit.chi2
        );
    }
}

#[test]
fn no_dip_anywhere_gives_no_depth() {
    // Every residual negative: no template position gains anything.
    let template = Template::tls_default();
    let time: Vec<f64> = (0..500).map(|i| f64::from(i) * 0.01).collect();
    let points = Points {
        residual: vec![-1e-3; time.len()],
        weight: vec![1.0; time.len()],
        time,
    };
    let found = best(&points, &template, 1.0, 0.1, 10);
    assert_eq!(found.fit.depth, 0.0);
    assert_eq!(found.fit.chi2, points.flat_chi2());
    assert_eq!(found.t0, 0.0);
}
