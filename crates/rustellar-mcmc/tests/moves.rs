// The walk, differential-evolution, snooker and slice moves: Gaussians
// recovered (three correlated and badly scaled, ten), the snooker move
// free of scale, each move's chain the image of itself under an affine
// map, the same from a seed on any number of threads, the slice move's
// tuning and support; the moves' checks.

use std::num::NonZeroUsize;

use rustellar_mcmc::{Ensemble, McmcError, Move};

mod common;
use common::{assert_moments, badly_scaled, ball, gaussian};

fn all_moves() -> Vec<Move> {
    vec![
        Move::walk(None).unwrap(),
        Move::walk(Some(4)).unwrap(),
        Move::differential_evolution(None, 1e-5).unwrap(),
        Move::snooker(1.7).unwrap(),
        Move::slice(1.0, 200).unwrap(),
        Move::slice_gaussian(1.0, 200).unwrap(),
    ]
}

#[test]
fn each_move_recovers_a_badly_scaled_gaussian() {
    let (mu, cov, target) = badly_scaled();
    let start = ball(32, &mu, &[1e-3, 1.0, 1e3], 2);
    for (i, mv) in all_moves().into_iter().enumerate() {
        let chain = Ensemble::new(start.clone())
            .unwrap()
            .with_move(mv)
            .with_seed(20 + i as u64)
            .run(&target, 20_000)
            .unwrap();
        assert_moments(&chain, 2000, &mu, &cov, 0.06);
    }
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
    for mv in [
        Move::differential_evolution(None, 1e-5).unwrap(),
        Move::slice(1.0, 200).unwrap(),
        Move::slice_gaussian(1.0, 200).unwrap(),
    ] {
        let chain = Ensemble::new(start.clone())
            .unwrap()
            .with_move(mv)
            .with_seed(14)
            .run(&target, 10_000)
            .unwrap();
        assert_moments(&chain, 2000, &mu, &cov, 0.1);
    }
}

#[test]
fn snooker_free_of_scale() {
    // A five-dimensional normal at three scales: the variance comes out
    // the same, at every scale (emcee 3.1's snooker move does not).
    for scale in [0.1, 1.0, 10.0] {
        let target = move |x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>() / (scale * scale);
        let start = ball(32, &[0.0; 5], &[10.0 * scale; 5], 4);
        let chain = Ensemble::new(start)
            .unwrap()
            .with_move(Move::snooker(1.7).unwrap())
            .with_seed(4)
            .run(&target, 20_000)
            .unwrap();
        let cov = chain.covariance(2000);
        for i in 0..5 {
            let var = cov[i * 5 + i] / (scale * scale);
            assert!((var - 1.0).abs() < 0.1, "scale {scale}: {var}");
        }
    }
}

/// As the stretch move's test: y = A x + b, the chain in y the image of
/// the chain in x, for some steps (rounding grows exponentially, fastest
/// for the snooker move), every proposal accepted or rejected alike.
fn assert_invariant(mv: Move, a: [[f64; 3]; 3], steps: usize) {
    let b = [10.0, -4.0, 1.0];
    let map = |x: &[f64]| -> Vec<f64> {
        (0..3)
            .map(|i| a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2] + b[i])
            .collect()
    };
    let inv = common::invert3(&a.concat());
    let px = gaussian(
        vec![0.0; 3],
        vec![1.0, 0.3, 0.0, 0.3, 2.0, 0.1, 0.0, 0.1, 1.5],
    );
    let py = |y: &[f64]| {
        let r: Vec<f64> = y.iter().zip(b).map(|(v, c)| v - c).collect();
        let x: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| inv[i * 3 + j] * r[j]).sum())
            .collect();
        px(&x)
    };
    let start = ball(12, &[0.0; 3], &[1.0; 3], 5);
    let mapped: Vec<Vec<f64>> = start.iter().map(|x| map(x)).collect();
    {
        let cx = Ensemble::new(start.clone())
            .unwrap()
            .with_move(mv)
            .with_seed(6)
            .run(&px, steps)
            .unwrap();
        let cy = Ensemble::new(mapped.clone())
            .unwrap()
            .with_move(mv)
            .with_seed(6)
            .run(&py, steps)
            .unwrap();
        for s in 0..steps {
            for w in 0..12 {
                let want = map(cx.position(s, w));
                let have = cy.position(s, w);
                for i in 0..3 {
                    assert!(
                        (want[i] - have[i]).abs() < 1e-6 * want[i].abs().max(1.0),
                        "{mv:?} {s} {w}: {want:?} {have:?}"
                    );
                }
            }
        }
        assert_eq!(cx.acceptance_fractions(), cy.acceptance_fractions());
        assert_eq!(cx.evaluations(), cy.evaluations());
    }
}

#[test]
fn affine_invariance() {
    let shear = [[2.0, 0.5, 0.0], [-1.0, 3.0, 0.2], [0.0, 0.1, 0.5]];
    for mv in all_moves() {
        if !matches!(mv, Move::Snooker { .. }) {
            assert_invariant(mv, shear, 40);
        }
    }
    // The snooker move under a rotation and an even scaling.
    let (s, c) = (0.6, 0.8);
    let k = 3.0;
    let turn = [[k * c, -k * s, 0.0], [k * s, k * c, 0.0], [0.0, 0.0, k]];
    assert_invariant(Move::snooker(1.7).unwrap(), turn, 25);
}

#[test]
fn seeds_and_threads() {
    let target = gaussian(
        vec![0.0; 3],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.5, 3.0],
    );
    let start = ball(12, &[0.0; 3], &[1.0; 3], 7);
    let mixture: Vec<(Move, f64)> = all_moves()
        .into_iter()
        .zip([1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
        .collect();
    let ensemble = Ensemble::new(start)
        .unwrap()
        .with_seed(9)
        .with_moves(mixture)
        .unwrap();
    let one = ensemble.run(&target, 300).unwrap();
    for threads in [2, 5, 64] {
        let many = ensemble
            .clone()
            .with_threads(NonZeroUsize::new(threads).unwrap())
            .run(&target, 300)
            .unwrap();
        assert_eq!(one, many, "{threads} threads");
    }
    assert_ne!(one, ensemble.with_seed(10).run(&target, 300).unwrap());
}

#[test]
fn slice_always_moves_and_tunes() {
    let target = gaussian(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]);
    let start = ball(8, &[0.0, 0.0], &[1.0, 1.0], 8);
    let run = |mv: Move| {
        Ensemble::new(start.clone())
            .unwrap()
            .with_move(mv)
            .with_seed(3)
            .run(&target, 400)
            .unwrap()
    };
    let chain = run(Move::slice(1.0, 0).unwrap());
    assert!(chain.acceptance_fraction() > 0.999);
    assert!(chain.evaluations() > 8 + 8 * 400);
    // Far too wide a start: tuned over 100 steps, the last 300 cost far
    // fewer log-probabilities than without tuning.
    // Where the slice is a single point (a height that rounds to the
    // log-probability), the walkers stay.
    let point = |_: &[f64]| 1e20;
    let stuck = Ensemble::new(start.clone())
        .unwrap()
        .with_move(Move::slice(1.0, 0).unwrap())
        .run(&point, 3)
        .unwrap();
    assert_eq!(stuck.acceptance_fraction(), 0.0);
    assert_eq!(stuck.last(), start);
    let wide = run(Move::slice(1e4, 0).unwrap()).evaluations();
    let tuned = run(Move::slice(1e4, 100).unwrap()).evaluations();
    assert!((tuned as f64) < 0.5 * wide as f64, "{tuned} {wide}");
}

#[test]
fn slice_never_outside_the_support() {
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
        .with_move(Move::slice(1.0, 50).unwrap())
        .with_seed(1)
        .run(&target, 5000)
        .unwrap();
    assert!(chain.samples(0).all(|x| x[0] > 0.0 && x[0] <= 3.0));
    assert!((chain.mean(500)[0] - 0.7899).abs() < 0.03);
}

#[test]
fn checks() {
    let bad = |name| Err(McmcError::InvalidMove { name });
    assert_eq!(Move::walk(Some(1)), bad("subset"));
    assert_eq!(Move::walk(Some(2)), Ok(Move::Walk { subset: Some(2) }));
    for g in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(Move::differential_evolution(Some(g), 0.0), bad("gamma"));
        assert_eq!(Move::snooker(g), bad("gamma"));
        assert_eq!(Move::slice(g, 0), bad("mu"));
        assert_eq!(Move::slice_gaussian(g, 0), bad("mu"));
    }
    for s in [-1e-9, f64::NAN, f64::INFINITY] {
        assert_eq!(Move::differential_evolution(None, s), bad("sigma"));
    }
    assert_eq!(
        Move::differential_evolution(Some(0.5), 0.0),
        Ok(Move::DifferentialEvolution {
            gamma: Some(0.5),
            sigma: 0.0
        })
    );
    // Moves needing more walkers, and settings out of range set directly.
    let target = |x: &[f64]| -x[0] * x[0];
    let four = Ensemble::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let needs = |mv: Move| four.clone().with_move(mv).run(&target, 1).unwrap_err();
    assert_eq!(
        needs(Move::snooker(1.7).unwrap()),
        McmcError::TooFewWalkers { needed: 6 }
    );
    assert_eq!(
        needs(Move::walk(Some(3)).unwrap()),
        McmcError::TooFewWalkers { needed: 6 }
    );
    assert_eq!(
        needs(Move::Stretch { a: 0.5 }),
        McmcError::InvalidMove { name: "a" }
    );
    let two = Ensemble::new(vec![vec![0.0], vec![1.0]]).unwrap();
    for mv in [
        Move::differential_evolution(None, 0.0).unwrap(),
        Move::slice(1.0, 0).unwrap(),
    ] {
        assert_eq!(
            two.clone().with_move(mv).run(&target, 1).unwrap_err(),
            McmcError::TooFewWalkers { needed: 4 }
        );
    }
    assert!(
        four.with_move(Move::walk(None).unwrap())
            .run(&target, 3)
            .is_ok()
    );
}
