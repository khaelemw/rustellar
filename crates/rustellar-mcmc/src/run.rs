//! A run: the ensemble moved in two halves, step after step.

use std::thread;

use crate::moves::{Proposal, index, slice_direction};
use crate::rng::Rng;
use crate::slice::{self, SliceUpdate};
use crate::{Chain, Ensemble, LogProbability, McmcError, Move};

impl Ensemble {
    /// `steps` steps from the starting points, giving the [`Chain`].
    ///
    /// Each step splits the walkers in two halves at random, then moves the
    /// first half using the second half's positions and the second half
    /// using the first's (Foreman-Mackey et al. 2013), which keeps the
    /// target's distribution as each half's moves do. Every random number of a half is drawn
    /// before any log-probability is worked out, so the chain is the same
    /// for a seed on any number of threads. A NaN log-probability counts
    /// as minus infinity: never moved to.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] if a move's setting is out of range,
    /// [`McmcError::TooFewWalkers`] if a move needs more walkers (six for
    /// the snooker move, four for differential evolution and the slice
    /// move, twice the subset for the walk move), and
    /// [`McmcError::OutsideSupport`] if a walker starts where `target` is
    /// not finite.
    pub fn run<T: LogProbability + Sync + ?Sized>(
        &self,
        target: &T,
        steps: usize,
    ) -> Result<Chain, McmcError> {
        let threads = self.threads().get();
        let walkers = self.walkers();
        for &(mv, _) in self.moves() {
            let needed = mv.checked()?.walkers_needed();
            if walkers < needed {
                return Err(McmcError::TooFewWalkers { needed });
            }
        }
        // Each slice move's width, tuned over its first steps.
        let mut widths: Vec<f64> = self
            .moves()
            .iter()
            .map(|&(mv, _)| match mv {
                Move::Slice { mu, .. } => mu,
                _ => 0.0,
            })
            .collect();
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
        let mut order: Vec<usize> = (0..walkers).collect();
        for step in 0..steps {
            // A new split of the walkers in two halves each step, as emcee.
            for i in (1..walkers).rev() {
                order.swap(i, index(&mut rng, i + 1));
            }
            let (first, second) = order.split_at(half);
            for (active, other) in [(first, second), (second, first)] {
                let which = self.pick(&mut rng);
                let mv = self.moves()[which].0;
                if let Move::Slice {
                    tune, direction, ..
                } = mv
                {
                    let mu = widths[which];
                    // The directions and each walker's own random numbers,
                    // drawn before any log-probability.
                    let others: Vec<&[f64]> = other.iter().map(|&j| x[j].as_slice()).collect();
                    let jobs: Vec<(usize, Vec<f64>, Rng)> = active
                        .iter()
                        .map(|&k| {
                            let along = slice_direction(direction, mu, &others, &mut rng);
                            (k, along, Rng::new(rng.next_u64()))
                        })
                        .collect();
                    let updates: Vec<SliceUpdate> =
                        parallel_map(&jobs, threads, |(k, direction, r)| {
                            slice::update(target, &x[*k], lp[*k], direction, &mut r.clone())
                        });
                    let (mut expansions, mut contractions) = (0, 0);
                    for ((k, _, _), u) in jobs.iter().zip(updates) {
                        chain.evaluations += u.evaluations;
                        expansions += u.expansions;
                        contractions += u.contractions;
                        if u.to != x[*k] {
                            chain.accepted[*k] += 1;
                        }
                        x[*k] = u.to;
                        lp[*k] = u.log_probability;
                    }
                    if step < tune {
                        widths[which] = mu * tuning_factor(expansions, contractions);
                    }
                    continue;
                }
                let proposals: Vec<Proposal> = {
                    let others: Vec<&[f64]> = other.iter().map(|&j| x[j].as_slice()).collect();
                    active
                        .iter()
                        .map(|&k| mv.propose(&x[k], &others, &mut rng))
                        .collect()
                };
                let thresholds: Vec<f64> = active.iter().map(|_| rng.unit().ln()).collect();
                let to: Vec<Vec<f64>> = proposals.iter().map(|p| p.to.clone()).collect();
                let new = evaluate(target, &to, threads);
                chain.evaluations += to.len();
                for (((&k, p), lp_new), threshold) in
                    active.iter().zip(proposals).zip(new).zip(thresholds)
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

/// Karamanis & Beutler's (2021) update of the slice width,
/// `2 expansions / (expansions + contractions)`, kept within a factor of 2
/// so a step with no expansions cannot shrink the width to nothing.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn tuning_factor(expansions: usize, contractions: usize) -> f64 {
    let total = expansions + contractions;
    if total == 0 {
        return 1.0;
    }
    (2.0 * expansions as f64 / total as f64).clamp(0.5, 2.0)
}

/// `target` at each point, NaN as minus infinity, over `threads` threads.
fn evaluate<T: LogProbability + Sync + ?Sized>(
    target: &T,
    points: &[Vec<f64>],
    threads: usize,
) -> Vec<f64> {
    parallel_map(points, threads, |x| {
        let v = target.log_probability(x);
        if v.is_nan() { f64::NEG_INFINITY } else { v }
    })
}

/// `f` of each item, over `threads` threads in contiguous shares, in order.
fn parallel_map<I: Sync, O: Send>(
    items: &[I],
    threads: usize,
    f: impl Fn(&I) -> O + Sync,
) -> Vec<O> {
    if threads <= 1 || items.len() <= 1 {
        return items.iter().map(f).collect();
    }
    let share = items.len().div_ceil(threads);
    let mut out: Vec<Option<O>> = items.iter().map(|_| None).collect();
    let f = &f;
    thread::scope(|scope| {
        for (part, chunk) in out.chunks_mut(share).zip(items.chunks(share)) {
            scope.spawn(move || {
                for (o, item) in part.iter_mut().zip(chunk) {
                    *o = Some(f(item));
                }
            });
        }
    });
    out.into_iter().flatten().collect()
}
