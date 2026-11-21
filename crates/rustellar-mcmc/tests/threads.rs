// The pool of threads: chains the same to the bit on any number of
// threads, for a transit posterior and every move; a start outside the
// support and a target that panics, with threads.

use std::num::NonZeroUsize;

use rustellar_fit::{
    Epoch, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio, ScaledSemiMajorAxis,
    TimeDelta, TransitFit, TransitParams,
};
use rustellar_mcmc::transit::TransitPosterior;
use rustellar_mcmc::{Ensemble, McmcError, Move};

fn threads(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

#[test]
fn transit_chains_the_same_on_any_threads() {
    let truth = TransitParams::new(
        Epoch::from_days(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap();
    let time: Vec<f64> = (0..300).map(|i| -0.15 + 0.001 * f64::from(i)).collect();
    let flux = time
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            truth.flux_at(Epoch::from_days(t).unwrap())
                + 1e-3 * ((i * 7919 % 13) as f64 / 6.0 - 1.0)
        })
        .collect();
    let lc = LightCurve::from_days(time, flux, vec![1e-3; 300]).unwrap();
    let setup = TransitFit::new(&lc, truth).with_fixed(Parameter::Period);
    let result = setup.run().unwrap();
    let posterior = TransitPosterior::new(&setup).unwrap();
    let start = posterior.walkers(&result, 20, 2).unwrap();
    let moves = vec![
        (Move::default(), 1.0),
        (Move::differential_evolution(None, 1e-5).unwrap(), 1.0),
        (Move::slice(1.0, 20).unwrap(), 1.0),
    ];
    let ensemble = Ensemble::new(start)
        .unwrap()
        .with_seed(4)
        .with_moves(moves)
        .unwrap();
    let one = ensemble.run(&posterior, 200).unwrap();
    for n in [2, 3, 8, 32] {
        let many = ensemble
            .clone()
            .with_threads(threads(n))
            .run(&posterior, 200)
            .unwrap();
        assert_eq!(one, many, "{n} threads");
    }
}

#[test]
fn a_start_outside_with_threads() {
    let target = |x: &[f64]| if x[0] > 0.5 { f64::NEG_INFINITY } else { 0.0 };
    let start = vec![vec![0.0], vec![0.2], vec![0.9], vec![0.3]];
    let err = Ensemble::new(start)
        .unwrap()
        .with_threads(threads(3))
        .run(&target, 10)
        .unwrap_err();
    assert_eq!(err, McmcError::OutsideSupport { walker: 2 });
}

#[test]
#[should_panic(expected = "the target failed")]
fn a_panic_in_the_target_reaches_the_caller() {
    // Every call panics once past the start, on whichever thread.
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let target = |x: &[f64]| {
        if calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 8 {
            panic!("the target failed");
        }
        -x[0] * x[0]
    };
    let start = (0..8).map(|i| vec![0.1 * f64::from(i)]).collect();
    let _ = Ensemble::new(start)
        .unwrap()
        .with_threads(threads(4))
        .run(&target, 5);
}
