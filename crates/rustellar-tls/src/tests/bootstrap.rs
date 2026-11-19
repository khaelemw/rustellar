// The bootstrap's errors: the curvature's where the signal is strong,
// larger in correlated noise, and the same on any number of threads.

use std::num::NonZeroUsize;

use super::{Random, light_curve};
use crate::bootstrap::{Spread, bootstrap};
use crate::errors::errors;
use crate::fit::{Points, Position, fit};
use crate::refine::{Reach, refine};
use crate::template::{SAMPLED_INTERVALS, Sampled, Template};

const TRUTH: Position = Position {
    period: 3.1,
    duration: 0.12,
    t0: 1.3,
};

/// The bootstrap's errors over the curvature's for period, t0, duration
/// and depth, with the bootstrap's own, from `count` resamples.
fn ratios(points: &Points, count: usize, threads: usize) -> ([f64; 4], Spread) {
    let template = Template::tls_default();
    let sampled = Sampled::new(&template, SAMPLED_INTERVALS);
    let reach = Reach {
        t0: 0.02,
        duration: 1.3,
        period: 5e-3,
    };
    let p = refine(points, &sampled, TRUTH, reach).position;
    let depth = fit(points, &template, p).depth;
    let e = errors(points, &template, p, depth).unwrap();
    let b = bootstrap(
        points,
        &sampled,
        &template,
        p,
        depth,
        [e.period, e.t0, e.duration],
        count,
        NonZeroUsize::new(threads).unwrap(),
    );
    (
        [
            b.period / e.period,
            b.t0 / e.t0,
            b.duration / e.duration,
            b.depth / e.depth,
        ],
        b,
    )
}

// About 100 times the noise in all, at 10 minutes: the chi-squared a
// quadratic, and the bootstrap's errors the curvature's, to the 7% a
// spread from 100 resamples has.
#[test]
fn where_the_signal_is_strong_the_curvatures() {
    let mut random = Random::new(5);
    let points = light_curve(
        &mut random,
        &Template::tls_default(),
        TRUTH,
        2e-3,
        2e-4,
        (0.0, 27.0, 1.0 / 144.0),
    );
    let (r, _) = ratios(&points, 100, 4);
    for (i, r) in r.iter().enumerate() {
        assert!((r - 1.0).abs() < 0.2, "{i}: {r}");
    }
}

// Noise correlated over 2.4 hours, as much as the white: the curvature
// takes it as white, the blocks a transit long keep it, and the
// bootstrap's errors are well above the curvature's.
#[test]
fn in_correlated_noise_larger() {
    let mut random = Random::new(6);
    let cadence = 1.0 / 48.0;
    let mut points = light_curve(
        &mut random,
        &Template::tls_default(),
        TRUTH,
        2e-3,
        6e-4,
        (0.0, 27.0, cadence),
    );
    let phi = f64::exp(-cadence / 0.1);
    let mut ar = 0.0;
    for r in &mut points.residual {
        let n = (random.uniform() - 0.5) * 12_f64.sqrt();
        ar = phi * ar + (1.0 - phi * phi).sqrt() * 6e-4 * n;
        *r += ar;
    }
    let (r, _) = ratios(&points, 100, 4);
    for (i, r) in r.iter().enumerate() {
        assert!(*r > 1.2, "{i}: {r}");
    }
}

// Each resample drawn from its own seed: the same errors on one thread and
// on three, and every run.
#[test]
fn the_same_on_any_number_of_threads() {
    let mut random = Random::new(7);
    let points = light_curve(
        &mut random,
        &Template::tls_default(),
        TRUTH,
        2e-3,
        1e-3,
        (0.0, 27.0, 1.0 / 48.0),
    );
    let (_, one) = ratios(&points, 20, 1);
    let (_, three) = ratios(&points, 20, 3);
    assert_eq!(one, three);
    assert_eq!(ratios(&points, 20, 3).1, three);
    // More threads than resamples.
    assert_eq!(ratios(&points, 2, 8).1, ratios(&points, 2, 1).1);
}
