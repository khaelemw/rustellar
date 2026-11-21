//! The threads a run's log-probabilities are worked out on: spawned once
//! per run and kept, each half-step's work shared out item by item, the
//! results placed by index so they never depend on which thread worked
//! them out.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex, PoisonError, RwLock};

use crate::LogProbability;
use crate::rng::Rng;
use crate::slice::{self, SliceUpdate};

/// One walker's slice update to make: where it is, its log-probability,
/// the direction and its own random numbers.
#[derive(Debug, Clone)]
pub(crate) struct SliceJob {
    pub(crate) x: Vec<f64>,
    pub(crate) lp: f64,
    pub(crate) direction: Vec<f64>,
    pub(crate) rng: Rng,
}

/// Work for the threads.
enum Job {
    /// Nothing yet.
    Idle,
    /// The log-probability at each point.
    Evaluate(Vec<Vec<f64>>),
    /// Each slice update.
    Slice(Vec<SliceJob>),
    /// The run is over.
    Stop,
}

/// How a run's parallel work is done: on the calling thread alone, or
/// shared with a pool.
pub(crate) trait Exec {
    /// The log-probability at each point, NaN as minus infinity.
    fn evaluate(&self, points: Vec<Vec<f64>>) -> Vec<f64>;
    /// Each slice update.
    fn slices(&self, jobs: Vec<SliceJob>) -> Vec<SliceUpdate>;
}

/// The target at `x`, NaN as minus infinity.
fn log_p<T: LogProbability + ?Sized>(target: &T, x: &[f64]) -> f64 {
    let v = target.log_probability(x);
    if v.is_nan() { f64::NEG_INFINITY } else { v }
}

fn slice_update<T: LogProbability + ?Sized>(target: &T, job: &SliceJob) -> SliceUpdate {
    slice::update(target, &job.x, job.lp, &job.direction, &mut job.rng.clone())
}

/// Everything on the calling thread.
pub(crate) struct Serial<'t, T: ?Sized>(pub(crate) &'t T);

impl<T: LogProbability + ?Sized> Exec for Serial<'_, T> {
    fn evaluate(&self, points: Vec<Vec<f64>>) -> Vec<f64> {
        points.iter().map(|x| log_p(self.0, x)).collect()
    }

    fn slices(&self, jobs: Vec<SliceJob>) -> Vec<SliceUpdate> {
        jobs.iter().map(|j| slice_update(self.0, j)).collect()
    }
}

/// A pool: the calling thread leads, the workers wait at a barrier for
/// each batch, every thread takes items from a shared counter, and a
/// second barrier ends the batch.
pub(crate) struct Pool<'t, T: ?Sized> {
    target: &'t T,
    job: RwLock<Job>,
    next: AtomicUsize,
    log_probabilities: Mutex<Vec<(usize, f64)>>,
    updates: Mutex<Vec<(usize, SliceUpdate)>>,
    panicked: Mutex<Option<Box<dyn Any + Send>>>,
    start: Barrier,
    end: Barrier,
}

impl<'t, T: LogProbability + Sync + ?Sized> Pool<'t, T> {
    /// A pool for `threads` threads in all, the caller among them.
    pub(crate) fn new(target: &'t T, threads: usize) -> Self {
        Self {
            target,
            job: RwLock::new(Job::Idle),
            next: AtomicUsize::new(0),
            log_probabilities: Mutex::new(Vec::new()),
            updates: Mutex::new(Vec::new()),
            panicked: Mutex::new(None),
            start: Barrier::new(threads),
            end: Barrier::new(threads),
        }
    }

    /// A worker's loop, until the run stops.
    pub(crate) fn worker(&self) {
        loop {
            self.start.wait();
            if matches!(
                *self.job.read().unwrap_or_else(PoisonError::into_inner),
                Job::Stop
            ) {
                return;
            }
            self.work();
            self.end.wait();
        }
    }

    /// Ends the workers' loops.
    pub(crate) fn stop(&self) {
        *self.job.write().unwrap_or_else(PoisonError::into_inner) = Job::Stop;
        self.start.wait();
    }

    /// This thread's share of the batch: items taken one at a time until
    /// none is left. A panic in the target is kept for the leader to
    /// raise, so no thread is left waiting at a barrier.
    fn work(&self) {
        let job = self.job.read().unwrap_or_else(PoisonError::into_inner);
        let (mut lps, mut updates) = (Vec::new(), Vec::new());
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            loop {
                let i = self.next.fetch_add(1, Ordering::Relaxed);
                match &*job {
                    Job::Evaluate(points) if i < points.len() => {
                        lps.push((i, log_p(self.target, &points[i])));
                    }
                    Job::Slice(jobs) if i < jobs.len() => {
                        updates.push((i, slice_update(self.target, &jobs[i])));
                    }
                    _ => break,
                }
            }
        }));
        if let Err(payload) = outcome {
            *lock(&self.panicked) = Some(payload);
        }
        lock(&self.log_probabilities).append(&mut lps);
        lock(&self.updates).append(&mut updates);
    }

    /// The batch worked out by every thread; a panic in the target raised
    /// once the workers are stopped.
    fn batch(&self, job: Job) {
        *self.job.write().unwrap_or_else(PoisonError::into_inner) = job;
        self.next.store(0, Ordering::Relaxed);
        self.start.wait();
        self.work();
        self.end.wait();
        if let Some(payload) = lock(&self.panicked).take() {
            self.stop();
            panic::resume_unwind(payload);
        }
    }
}

fn lock<V>(m: &Mutex<V>) -> std::sync::MutexGuard<'_, V> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The values in item order.
fn in_order<V>(mut items: Vec<(usize, V)>) -> Vec<V> {
    items.sort_unstable_by_key(|&(i, _)| i);
    items.into_iter().map(|(_, v)| v).collect()
}

impl<T: LogProbability + Sync + ?Sized> Exec for Pool<'_, T> {
    fn evaluate(&self, points: Vec<Vec<f64>>) -> Vec<f64> {
        self.batch(Job::Evaluate(points));
        in_order(std::mem::take(&mut *lock(&self.log_probabilities)))
    }

    fn slices(&self, jobs: Vec<SliceJob>) -> Vec<SliceUpdate> {
        self.batch(Job::Slice(jobs));
        in_order(std::mem::take(&mut *lock(&self.updates)))
    }
}
