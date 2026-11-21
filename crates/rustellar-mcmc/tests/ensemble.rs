// Setting up an ensemble: the checks on its walkers, the builders and the
// starting log-probabilities.

use std::num::NonZeroUsize;

use rustellar_mcmc::{Ensemble, LogProbability, McmcError};

fn ball(walkers: usize, dimensions: usize) -> Vec<Vec<f64>> {
    (0..walkers)
        .map(|w| {
            (0..dimensions)
                .map(|d| (((w * 7 + d * 13) % 11) as f64 - 5.0) * 0.01)
                .collect()
        })
        .collect()
}

#[test]
fn defaults_and_builders() {
    let ensemble = Ensemble::new(ball(8, 3)).unwrap();
    assert_eq!(ensemble.walkers(), 8);
    assert_eq!(ensemble.dimensions(), 3);
    assert_eq!(ensemble.seed(), 0);
    assert_eq!(ensemble.threads().get(), 1);
    assert_eq!(ensemble.start(), &ball(8, 3)[..]);
    let ensemble = ensemble
        .with_seed(99)
        .with_threads(NonZeroUsize::new(4).unwrap());
    assert_eq!(ensemble.seed(), 99);
    assert_eq!(ensemble.threads().get(), 4);
}

#[test]
fn checks_on_the_walkers() {
    assert_eq!(Ensemble::new(Vec::new()), Err(McmcError::NoDimensions));
    assert_eq!(
        Ensemble::new(vec![Vec::new(); 4]),
        Err(McmcError::NoDimensions)
    );
    let mut uneven = ball(8, 3);
    uneven[5].pop();
    assert_eq!(
        Ensemble::new(uneven),
        Err(McmcError::DimensionMismatch { walker: 5 })
    );
    let mut nan = ball(8, 3);
    nan[2][1] = f64::NAN;
    assert_eq!(Ensemble::new(nan), Err(McmcError::NonFinite { walker: 2 }));
    let mut infinite = ball(8, 3);
    infinite[7][0] = f64::INFINITY;
    assert_eq!(
        Ensemble::new(infinite),
        Err(McmcError::NonFinite { walker: 7 })
    );
    assert_eq!(
        Ensemble::new(ball(5, 3)),
        Err(McmcError::TooFewWalkers { needed: 6 })
    );
    assert_eq!(Ensemble::new(ball(7, 3)), Err(McmcError::OddWalkers));
    // Every walker on one line through the space.
    let line: Vec<Vec<f64>> = (0..8)
        .map(|i| {
            let t = f64::from(i);
            vec![t, 2.0 * t, -t]
        })
        .collect();
    assert_eq!(Ensemble::new(line), Err(McmcError::Degenerate));
}

#[test]
fn the_start_inside_the_support() {
    // A unit normal restricted to positive values.
    let target = |x: &[f64]| {
        if x.iter().any(|&v| v < 0.0) {
            f64::NEG_INFINITY
        } else {
            -0.5 * x.iter().map(|v| v * v).sum::<f64>()
        }
    };
    let inside: Vec<Vec<f64>> = ball(6, 2)
        .into_iter()
        .map(|p| p.into_iter().map(|v| v + 1.0).collect())
        .collect();
    assert_eq!(Ensemble::new(inside).unwrap().check(&target), Ok(()));
    let ensemble = Ensemble::new(ball(6, 2)).unwrap();
    let first = ensemble
        .start()
        .iter()
        .position(|p| p.iter().any(|&v| v < 0.0))
        .unwrap();
    assert_eq!(
        ensemble.check(&target),
        Err(McmcError::OutsideSupport { walker: first })
    );
    // NaN counts as outside, through a trait object too.
    let nan = |_: &[f64]| f64::NAN;
    let dynamic: &dyn LogProbability = &nan;
    assert_eq!(
        ensemble.check(dynamic),
        Err(McmcError::OutsideSupport { walker: 0 })
    );
}

#[test]
fn messages() {
    for err in [
        McmcError::NoDimensions,
        McmcError::TooFewWalkers { needed: 4 },
        McmcError::OddWalkers,
        McmcError::DimensionMismatch { walker: 1 },
        McmcError::NonFinite { walker: 1 },
        McmcError::Degenerate,
        McmcError::OutsideSupport { walker: 1 },
    ] {
        assert!(!err.to_string().is_empty());
    }
}
