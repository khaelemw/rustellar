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
