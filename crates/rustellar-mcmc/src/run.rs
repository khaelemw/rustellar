//! A run: the ensemble moved in two halves, step after step.

use std::thread;

use crate::moves::Proposal;
use crate::rng::Rng;
use crate::{Chain, Ensemble, LogProbability, McmcError};

impl Ensemble {
    /// `steps` steps from the starting points, giving the [`Chain`].
    ///
    /// Each step moves the first half of the walkers using the second
    /// half's positions, then the second half using the first's
    /// (Foreman-Mackey et al. 2013), which keeps the target's distribution
    /// as each half's moves do. Every random number of a half is drawn
    /// before any log-probability is worked out, so the chain is the same
    /// for a seed on any number of threads. A NaN log-probability counts
    /// as minus infinity: never moved to.
    ///
    /// # Errors
    ///
    /// [`McmcError::OutsideSupport`] if a walker starts where `target` is
    /// not finite.
    pub fn run<T: LogProbability + Sync + ?Sized>(
        &self,
        target: &T,
        steps: usize,
    ) -> Result<Chain, McmcError> {
        let threads = self.threads().get();
        let walkers = self.walkers();
        let d = self.dimensions();
        let mut x: Vec<Vec<f64>> = self.start().to_vec();
        let mut lp = evaluate(target, &x, threads);
        if let Some(walker) = lp.iter().position(|v| !v.is_finite()) {
            return Err(McmcError::OutsideSupport { walker });
        }
        let mut rng = Rng::new(self.seed());
        let mut chain = Chain {
            walkers,
            dimensions: d,
            start: x.concat(),
            positions: Vec::with_capacity(steps * walkers * d),
            log_probabilities: Vec::with_capacity(steps * walkers),
            accepted: vec![0; walkers],
            evaluations: walkers,
        };
        let half = walkers / 2;
        for _ in 0..steps {
            for (active, other) in [(0..half, half..walkers), (half..walkers, 0..half)] {
                let mv = self.pick(&mut rng);
                let proposals: Vec<Proposal> = {
                    let others: Vec<&[f64]> = x[other].iter().map(Vec::as_slice).collect();
                    active
                        .clone()
                        .map(|k| mv.propose(&x[k], &others, &mut rng))
                        .collect()
                };
                let thresholds: Vec<f64> = active.clone().map(|_| rng.unit().ln()).collect();
                let to: Vec<Vec<f64>> = proposals.iter().map(|p| p.to.clone()).collect();
                let new = evaluate(target, &to, threads);
                chain.evaluations += to.len();
                for (((k, p), lp_new), threshold) in active.zip(proposals).zip(new).zip(thresholds)
                {
                    let gain = p.log_ratio + lp_new - lp[k];
                    if gain > threshold {
                        x[k] = p.to;
                        lp[k] = lp_new;
                        chain.accepted[k] += 1;
                    }
                }
            }
            for xi in &x {
                chain.positions.extend_from_slice(xi);
            }
            chain.log_probabilities.extend_from_slice(&lp);
        }
        Ok(chain)
    }
}

/// `target` at each point, NaN as minus infinity, over `threads` threads
/// in contiguous shares.
fn evaluate<T: LogProbability + Sync + ?Sized>(
    target: &T,
    points: &[Vec<f64>],
    threads: usize,
) -> Vec<f64> {
    let one = |x: &Vec<f64>| {
        let v = target.log_probability(x);
        if v.is_nan() { f64::NEG_INFINITY } else { v }
    };
    if threads <= 1 || points.len() <= 1 {
        return points.iter().map(one).collect();
    }
    let share = points.len().div_ceil(threads);
    let mut out = vec![0.0; points.len()];
    thread::scope(|scope| {
        for (part, chunk) in out.chunks_mut(share).zip(points.chunks(share)) {
            scope.spawn(move || {
                for (o, x) in part.iter_mut().zip(chunk) {
                    *o = one(x);
                }
            });
        }
    });
    out
}
