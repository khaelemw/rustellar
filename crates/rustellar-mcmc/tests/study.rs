// The moves' autocorrelation times, for the README's table against emcee:
// a ten-dimensional normal (any Gaussian, for affine-invariant moves) with
// 40 walkers, and Goodman & Weare's Rosenbrock density with 20, three
// seeds each. Run with `cargo test --release -p rustellar-mcmc --test
// study -- --ignored --nocapture`.

use std::time::Instant;

use rustellar_mcmc::{Ensemble, Move};

mod common;
use common::ball;

type Target = fn(&[f64]) -> f64;

/// A target: its name, log-density, dimensions, walkers, steps, the steps
/// discarded and where the walkers start.
type Case = (&'static str, Target, usize, usize, usize, usize, Vec<f64>);

fn gauss(x: &[f64]) -> f64 {
    -0.5 * x.iter().map(|v| v * v).sum::<f64>()
}

fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (x[1] - x[0] * x[0], 1.0 - x[0]);
    -(100.0 * a * a + b * b) / 20.0
}

#[test]
#[ignore = "heavy: a study of the moves"]
fn moves_autocorrelation_times() {
    let targets: [Case; 2] = [
        ("gauss10", gauss, 10, 40, 20_000, 2000, vec![0.0; 10]),
        (
            "rosenbrock",
            rosenbrock,
            2,
            20,
            50_000,
            5000,
            vec![1.0, 1.0],
        ),
    ];
    let moves: Vec<(&str, Vec<(Move, f64)>)> = vec![
        ("stretch", vec![(Move::default(), 1.0)]),
        ("walk", vec![(Move::walk(None).unwrap(), 1.0)]),
        (
            "de",
            vec![(Move::differential_evolution(None, 1e-5).unwrap(), 1.0)],
        ),
        (
            "de+snooker",
            vec![
                (Move::differential_evolution(None, 1e-5).unwrap(), 0.8),
                (Move::snooker(1.7).unwrap(), 0.2),
            ],
        ),
        ("slice", vec![(Move::slice(1.0, 1000).unwrap(), 1.0)]),
        (
            "slice gaussian",
            vec![(Move::slice_gaussian(1.0, 1000).unwrap(), 1.0)],
        ),
    ];
    println!("target,move,tau,sd,evaluations per walker-step,per independent sample,seconds");
    for (name, target, d, walkers, steps, discard, centre) in &targets {
        for (mname, mv) in &moves {
            let (mut taus, mut per, mut secs) = (Vec::new(), 0.0, 0.0);
            for seed in 0..3u64 {
                let start = ball(*walkers, centre, &vec![2.0; *d], seed + 1);
                let ensemble = Ensemble::new(start)
                    .unwrap()
                    .with_seed(seed)
                    .with_moves(mv.clone())
                    .unwrap();
                let t = Instant::now();
                let chain = ensemble.run(target, *steps).unwrap();
                secs += t.elapsed().as_secs_f64();
                let tau = chain
                    .autocorrelation_times(*discard)
                    .iter()
                    .map(|t| t.tau)
                    .fold(0.0, f64::max);
                taus.push(tau);
                per += chain.evaluations() as f64 / (*walkers * *steps) as f64;
            }
            let mean = taus.iter().sum::<f64>() / 3.0;
            let sd = (taus.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / 3.0).sqrt();
            let per = per / 3.0;
            println!(
                "{name},{mname},{mean:.2},{sd:.2},{per:.2},{:.1},{:.3}",
                mean * per,
                secs / 3.0
            );
        }
    }
}

#[test]
#[ignore = "heavy: a study of the threads"]
fn threads_on_a_transit() {
    use std::num::NonZeroUsize;

    use rustellar_fit::{
        Epoch, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
        ScaledSemiMajorAxis, TimeDelta, TransitFit, TransitParams,
    };
    use rustellar_mcmc::transit::TransitPosterior;

    let truth = TransitParams::new(
        Epoch::from_days(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap();
    println!("points,threads,seconds per 1000 steps,speed-up");
    for n in [500u32, 5000] {
        let time: Vec<f64> = (0..n)
            .map(|i| 0.3 * (f64::from(i) / f64::from(n) - 0.5))
            .collect();
        let flux = time
            .iter()
            .map(|&t| truth.flux_at(Epoch::from_days(t).unwrap()))
            .collect();
        let lc = LightCurve::from_days(time, flux, vec![1e-3; n as usize]).unwrap();
        let setup = TransitFit::new(&lc, truth).with_fixed(Parameter::Period);
        let result = setup.run().unwrap();
        let posterior = TransitPosterior::new(&setup).unwrap();
        let start = posterior.walkers(&result, 40, 1).unwrap();
        let mut one = 0.0;
        for threads in [1, 2, 4, 8, 16] {
            let ensemble = Ensemble::new(start.clone())
                .unwrap()
                .with_threads(NonZeroUsize::new(threads).unwrap());
            let steps = 1000;
            let t = Instant::now();
            let _ = ensemble.run(&posterior, steps).unwrap();
            let secs = t.elapsed().as_secs_f64();
            if threads == 1 {
                one = secs;
            }
            println!("{n},{threads},{secs:.3},{:.2}", one / secs);
        }
    }
}

#[test]
#[ignore = "heavy: a study of the transit posterior's speed"]
fn transit_speed() {
    use std::num::NonZeroUsize;

    use rustellar_fit::{
        Epoch, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
        ScaledSemiMajorAxis, TimeDelta, TransitFit, TransitParams,
    };
    use rustellar_mcmc::transit::TransitPosterior;

    // The reference light curve, as emcee's study reads it.
    let text = include_str!("reference/transit_points.csv");
    let rows: Vec<Vec<f64>> = text
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    let column = |i: usize| rows.iter().map(|r| r[i]).collect::<Vec<f64>>();
    let lc = LightCurve::from_days(column(0), column(1), column(2)).unwrap();
    let start = TransitParams::new(
        Epoch::from_days(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap();
    let setup = TransitFit::new(&lc, start).with_fixed(Parameter::Period);
    let result = setup.run().unwrap();
    let posterior = TransitPosterior::new(&setup).unwrap();
    let walkers = posterior.walkers(&result, 40, 3).unwrap();
    let (steps, discard) = (30_000, 5_000);
    println!(
        "move,threads,seconds,tau,independent samples,log-probabilities each,ms per independent sample"
    );
    for (name, mv) in [
        ("stretch", Move::default()),
        ("de", Move::differential_evolution(None, 1e-5).unwrap()),
        ("slice", Move::slice(1.0, 2000).unwrap()),
    ] {
        for threads in [1, 4] {
            let ensemble = Ensemble::new(walkers.clone())
                .unwrap()
                .with_seed(9)
                .with_move(mv)
                .with_threads(NonZeroUsize::new(threads).unwrap());
            let t = Instant::now();
            let chain = ensemble.run(&posterior, steps).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let tau = chain
                .autocorrelation_times(discard)
                .iter()
                .map(|t| t.tau)
                .fold(0.0, f64::max);
            let ess = 40.0 * (steps - discard) as f64 / tau;
            let per = chain.evaluations() as f64 / ess;
            println!(
                "{name},{threads},{secs:.2},{tau:.0},{ess:.0},{per:.0},{:.3}",
                1000.0 * secs / ess
            );
        }
    }
}
