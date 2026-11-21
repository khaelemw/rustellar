// The stretch move: Gaussians recovered (one dimension, three correlated
// and badly scaled, ten), the chain the same under an affine map of the
// space (the move's defining property), the same from a seed on any
// number of threads, never outside the support; the chain's accessors and
// the moves' checks.

use std::num::NonZeroUsize;

use rustellar_mcmc::{Ensemble, McmcError, Move};

mod common;
use common::{assert_moments, ball, gaussian, invert3};

#[test]
fn one_dimension() {
    let target = gaussian(vec![3.0], vec![1.0 / 4.0]);
    let start = ball(16, &[3.0], &[2.0], 1);
    let chain = Ensemble::new(start)
        .unwrap()
        .with_seed(11)
        .run(&target, 20_000)
        .unwrap();
    assert_moments(&chain, 1000, &[3.0], &[4.0], 0.03);
    let f = chain.acceptance_fraction();
    assert!(f > 0.5 && f < 0.9, "{f}");
}

#[test]
fn three_correlated_badly_scaled() {
    // Standard deviations 1e-3, 1 and 1e3; correlations 0.9, -0.3, 0.
    let sd = [1e-3, 1.0, 1e3];
    let rho = [[1.0, 0.9, -0.3], [0.9, 1.0, 0.0], [-0.3, 0.0, 1.0]];
    let mut cov = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            cov[i * 3 + j] = rho[i][j] * sd[i] * sd[j];
        }
    }
    let precision = invert3(&cov);
    let mu = vec![1.0, -2.0, 5e3];
    let target = gaussian(mu.clone(), precision);
    let start = ball(32, &mu, &sd, 2);
    let chain = Ensemble::new(start)
        .unwrap()
        .with_seed(12)
        .run(&target, 20_000)
        .unwrap();
    assert_moments(&chain, 2000, &mu, &cov, 0.05);
}

#[test]
fn ten_dimensions() {
    let d = 10;
    let mut precision = vec![0.0; d * d];
    let mut cov = vec![0.0; d * d];
    for i in 0..d {
        let var = 1.0 + i as f64;
        precision[i * d + i] = 1.0 / var;
        cov[i * d + i] = var;
    }
    let mu: Vec<f64> = (0..d).map(|i| i as f64).collect();
    let target = gaussian(mu.clone(), precision);
    let start = ball(40, &mu, &vec![1.0; d], 3);
    let chain = Ensemble::new(start)
        .unwrap()
        .with_seed(13)
        .run(&target, 20_000)
        .unwrap();
    assert_moments(&chain, 4000, &mu, &cov, 0.1);
}

#[test]
fn affine_invariance() {
    // y = A x + b: the chain in y is the image of the chain in x, step by
    // step, from the image of the start with the same seed: every proposal
    // accepted or rejected alike, the positions apart only by rounding.
    // The moves compound rounding exponentially, as any chaotic map does
    // (1e-9 after 170 steps here, 1e-6 after 250), so 150 steps.
    let a = [[2.0, 0.5], [-1.0, 3.0]];
    let b = [10.0, -4.0];
    let map = |x: &[f64]| -> Vec<f64> {
        (0..2)
            .map(|i| a[i][0] * x[0] + a[i][1] * x[1] + b[i])
            .collect()
    };
    let px = gaussian(vec![0.0, 0.0], vec![1.0, 0.3, 0.3, 2.0]);
    // log p_y(y) = log p_x(A^-1 (y - b)) + constant.
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let back = move |y: &[f64]| -> Vec<f64> {
        let (u, v) = (y[0] - b[0], y[1] - b[1]);
        vec![
            (a[1][1] * u - a[0][1] * v) / det,
            (-a[1][0] * u + a[0][0] * v) / det,
        ]
    };
    let py = |y: &[f64]| px(&back(y));
    let start = ball(8, &[0.0, 0.0], &[1.0, 1.0], 4);
    let mapped: Vec<Vec<f64>> = start.iter().map(|x| map(x)).collect();
    let cx = Ensemble::new(start)
        .unwrap()
        .with_seed(5)
        .run(&px, 150)
        .unwrap();
    let cy = Ensemble::new(mapped)
        .unwrap()
        .with_seed(5)
        .run(&py, 150)
        .unwrap();
    for s in 0..150 {
        for w in 0..8 {
            let want = map(cx.position(s, w));
            let have = cy.position(s, w);
            for i in 0..2 {
                assert!(
                    (want[i] - have[i]).abs() < 1e-8 * want[i].abs().max(1.0),
                    "{s} {w}"
                );
            }
        }
    }
    assert_eq!(cx.acceptance_fractions(), cy.acceptance_fractions());
}

#[test]
fn seeds_and_threads() {
    let target = gaussian(
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.5, 3.0],
    );
    let start = ball(12, &[0.0; 3], &[1.0; 3], 5);
    let ensemble = Ensemble::new(start).unwrap().with_seed(9);
    let one = ensemble.run(&target, 500).unwrap();
    assert_eq!(one, ensemble.run(&target, 500).unwrap());
    for threads in [2, 3, 5, 64] {
        let many = ensemble
            .clone()
            .with_threads(NonZeroUsize::new(threads).unwrap())
            .run(&target, 500)
            .unwrap();
        assert_eq!(one, many, "{threads} threads");
    }
    assert_ne!(one, ensemble.with_seed(10).run(&target, 500).unwrap());
}

#[test]
fn never_outside_the_support() {
    // A unit normal on x > 0 only, and NaN beyond x > 3 (treated as
    // outside); walkers start near the edge.
    let target = |x: &[f64]| {
        if x[0] <= 0.0 {
            f64::NEG_INFINITY
        } else if x[0] > 3.0 {
            f64::NAN
        } else {
            -0.5 * x[0] * x[0]
        }
    };
    let start: Vec<Vec<f64>> = (1..=8).map(|i| vec![0.01 * f64::from(i)]).collect();
    let chain = Ensemble::new(start)
        .unwrap()
        .with_seed(1)
        .run(&target, 5000)
        .unwrap();
    assert!(chain.samples(0).all(|x| x[0] > 0.0 && x[0] <= 3.0));
    // The half-normal's mean sqrt(2 / pi), truncated at 3 (0.7899).
    assert!((chain.mean(500)[0] - 0.7899).abs() < 0.03);
    assert_eq!(chain.evaluations(), 8 + 8 * 5000);
}

#[test]
fn chains_and_restarts() {
    let target = gaussian(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]);
    let start = ball(4, &[0.0, 0.0], &[1.0, 1.0], 6);
    let ensemble = Ensemble::new(start.clone()).unwrap();
    let empty = ensemble.run(&target, 0).unwrap();
    assert_eq!(empty.steps(), 0);
    assert_eq!((empty.walkers(), empty.dimensions()), (4, 2));
    assert_eq!(empty.last(), start);
    assert_eq!(empty.acceptance_fractions(), vec![0.0; 4]);
    assert_eq!(empty.samples(0).count(), 0);
    let chain = ensemble.run(&target, 10).unwrap();
    assert_eq!(chain.samples(3).count(), 7 * 4);
    assert_eq!(chain.samples(99).count(), 0);
    assert_eq!(chain.trace(2, 1).len(), 10);
    assert_eq!(chain.trace(2, 1)[9], chain.position(9, 2)[1]);
    assert_eq!(chain.last()[3], chain.position(9, 3));
    assert_eq!(chain.log_probability(9, 0), target(chain.position(9, 0)));
    let more = Ensemble::new(chain.last())
        .unwrap()
        .run(&target, 5)
        .unwrap();
    assert_eq!(more.steps(), 5);
    assert_eq!(ensemble.check(&target), Ok(()));
    let outside = |_: &[f64]| f64::NEG_INFINITY;
    assert_eq!(
        ensemble.run(&outside, 1).unwrap_err(),
        McmcError::OutsideSupport { walker: 0 }
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn position_out_of_range() {
    let target = |x: &[f64]| -x[0] * x[0];
    let chain = Ensemble::new(vec![vec![0.0], vec![1.0]])
        .unwrap()
        .run(&target, 2)
        .unwrap();
    let _ = chain.position(2, 0);
}

#[test]
#[should_panic(expected = "out of range")]
fn log_probability_out_of_range() {
    let target = |x: &[f64]| -x[0] * x[0];
    let chain = Ensemble::new(vec![vec![0.0], vec![1.0]])
        .unwrap()
        .run(&target, 2)
        .unwrap();
    let _ = chain.log_probability(0, 2);
}

#[test]
fn moves() {
    assert_eq!(Move::stretch(3.0), Ok(Move::Stretch { a: 3.0 }));
    for bad in [1.0, 0.5, f64::NAN, f64::INFINITY] {
        assert_eq!(
            Move::stretch(bad),
            Err(McmcError::InvalidMove { name: "a" })
        );
    }
    let ensemble = Ensemble::new(vec![vec![0.0], vec![1.0]]).unwrap();
    assert_eq!(ensemble.moves(), &[(Move::default(), 1.0)]);
    let narrow = ensemble.clone().with_move(Move::stretch(1.5).unwrap());
    assert_eq!(narrow.moves(), &[(Move::Stretch { a: 1.5 }, 1.0)]);
    assert_eq!(
        ensemble.clone().with_moves(Vec::new()).unwrap_err(),
        McmcError::InvalidMove { name: "moves" }
    );
    for w in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            ensemble
                .clone()
                .with_moves(vec![(Move::default(), w)])
                .unwrap_err(),
            McmcError::InvalidMove { name: "weight" }
        );
    }
    // A mixture samples the target too, the narrower move accepted more.
    let target = gaussian(vec![1.0], vec![1.0]);
    let start = ball(16, &[1.0], &[1.0], 7);
    let mixed = Ensemble::new(start.clone())
        .unwrap()
        .with_moves(vec![
            (Move::stretch(1.2).unwrap(), 0.3),
            (Move::stretch(4.0).unwrap(), 0.7),
        ])
        .unwrap()
        .with_seed(8)
        .run(&target, 20_000)
        .unwrap();
    assert_moments(&mixed, 1000, &[1.0], &[1.0], 0.03);
    let wide = Ensemble::new(start)
        .unwrap()
        .with_move(Move::stretch(4.0).unwrap())
        .run(&target, 2000)
        .unwrap();
    assert!(mixed.acceptance_fraction() > wide.acceptance_fraction());
    assert!(!McmcError::InvalidMove { name: "a" }.to_string().is_empty());
}
