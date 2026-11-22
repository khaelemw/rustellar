//! `rustellar-mcmc` against emcee (with batman and with PyTransit) and zeus
//! (with batman) on the transit posterior of rustellar-mcmc's reference
//! light curve: t0, k, a/R*, b, q1, q2 and the baseline free, uniform
//! priors, 40 walkers, 30,000 steps with the first 5,000 discarded. The
//! time is per independent sample: the run's seconds over 40 x 25,000
//! divided by the largest autocorrelation time.

use std::fmt::Write;
use std::num::NonZeroUsize;
use std::time::Instant;

use rustellar_fit::{
    Epoch, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio, ScaledSemiMajorAxis,
    TimeDelta, TransitFit, TransitParams,
};
use rustellar_mcmc::transit::TransitPosterior;
use rustellar_mcmc::{Ensemble, Move};

use super::{Reference, package, seconds, short};

const POINTS: &str = include_str!("../../../rustellar-mcmc/tests/reference/transit_points.csv");

/// Walkers, steps and the steps discarded, as the references.
pub const RUN: (usize, usize, usize) = (40, 30_000, 5_000);

/// The references' times, as the script that ran them recorded them: per
/// move, the seconds per independent sample, and in a row named
/// "MOVE tau" the largest autocorrelation time.
#[must_use]
pub fn reference() -> &'static str {
    include_str!("../../data/speed_mcmc.csv")
}

/// The moves timed.
pub const MOVES: [&str; 3] = ["stretch", "de", "slice"];

fn mv(name: &str) -> Move {
    match name {
        "stretch" => Move::default(),
        "de" => Move::differential_evolution(None, 1e-5).expect("valid"),
        _ => Move::slice(1.0, 2000).expect("valid"),
    }
}

/// The reference light curve.
///
/// # Panics
///
/// Never: the file is the reference test's.
#[must_use]
pub fn light_curve() -> LightCurve {
    let rows: Vec<Vec<f64>> = POINTS
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|v| v.parse().expect("a number")).collect())
        .collect();
    let column = |i: usize| rows.iter().map(|r| r[i]).collect::<Vec<f64>>();
    LightCurve::from_days(column(0), column(1), column(2)).expect("valid")
}

/// One time: the move, the threads, the largest autocorrelation time and
/// seconds per independent sample.
pub type Timing = (&'static str, usize, f64, f64);

/// Every move run on 1 and on `threads` threads (at most 4, as the
/// README), with `run`'s walkers, steps and steps discarded ([`RUN`]
/// for the references').
///
/// # Panics
///
/// Never: the light curve fits.
#[must_use]
pub fn run(lc: &LightCurve, threads: NonZeroUsize, run: (usize, usize, usize)) -> Vec<Timing> {
    let start = TransitParams::new(
        Epoch::from_days(0.0).expect("finite"),
        TimeDelta::from_days(3.0).expect("positive"),
        RadiusRatio::new(0.1).expect("valid"),
        ScaledSemiMajorAxis::new(10.0).expect("valid"),
        ImpactParameter::new(0.3).expect("valid"),
        LimbDarkening::quadratic(0.4, 0.25).expect("valid"),
    )
    .expect("valid");
    let setup = TransitFit::new(lc, start).with_fixed(Parameter::Period);
    let result = setup.run().expect("a fit");
    let posterior = TransitPosterior::new(&setup).expect("a posterior");
    let (walkers, steps, discard) = run;
    let start = posterior.walkers(&result, walkers, 3).expect("walkers");
    let mut counts = vec![1, threads.get().min(4)];
    counts.dedup();
    let mut out = Vec::new();
    for name in MOVES {
        for &n in &counts {
            let ensemble = Ensemble::new(start.clone())
                .expect("walkers")
                .with_seed(9)
                .with_move(mv(name))
                .with_threads(NonZeroUsize::new(n).expect("positive"));
            let t = Instant::now();
            let chain = ensemble.run(&posterior, steps).expect("a chain");
            let secs = t.elapsed().as_secs_f64();
            let tau = chain
                .autocorrelation_times(discard)
                .iter()
                .map(|t| t.tau)
                .fold(0.0, f64::max);
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            let independent = (walkers * (steps - discard)) as f64 / tau;
            out.push((name, n, tau, secs / independent));
        }
    }
    out
}

/// The README's table: each sampler's autocorrelation time and time per
/// independent sample.
#[must_use]
pub fn report(timings: &[Timing], references: &[Reference]) -> String {
    let mut out =
        String::from("| Sampler | Move | tau | Time per independent sample |\n|---|---|--:|--:|\n");
    for name in ["emcee+batman", "emcee+PyTransit", "zeus+batman"] {
        // "emcee+batman 3.1.6+2.5.1" as "emcee 3.1.6 + batman 2.5.1".
        let full = package(references, name);
        let versions = full.strip_prefix(name).unwrap_or("").trim().split('+');
        let label = name
            .split('+')
            .zip(versions.chain(std::iter::repeat("")))
            .map(|(n, v)| format!("{n} {v}").trim_end().to_owned())
            .collect::<Vec<String>>()
            .join(" + ");
        for m in MOVES {
            if let (Some(s), Some(tau)) = (
                seconds(references, name, m),
                seconds(references, name, &format!("{m} tau")),
            ) {
                writeln!(out, "| {label} | {m} | {tau:.0} | {} |", short(Some(s)))
                    .expect("a string");
            }
        }
    }
    for &(m, n, tau, s) in timings {
        let threads = if n == 1 {
            "1 thread".to_owned()
        } else {
            format!("{n} threads")
        };
        writeln!(
            out,
            "| rustellar-mcmc, {threads} | {m} | {tau:.0} | {} |",
            short(Some(s))
        )
        .expect("a string");
    }
    out
}
