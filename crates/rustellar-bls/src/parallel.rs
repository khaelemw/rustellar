//! Work split over threads, the outcomes in order, so a search gives the
//! same results to the bit on any number of threads.

use std::thread;

/// `work` on each of `jobs`, the first on this thread and the others on
/// threads of their own, the outcomes in the jobs' order. A panic on any
/// thread is raised again here once every thread has finished.
pub(crate) fn on_threads<J: Send, T: Send>(
    jobs: impl Iterator<Item = J>,
    work: &(impl Fn(J) -> T + Sync),
) -> Vec<T> {
    let jobs: Vec<J> = jobs.collect();
    let mut outcomes: Vec<Option<T>> = jobs.iter().map(|_| None).collect();
    thread::scope(|scope| {
        let mut pairs = jobs.into_iter().zip(outcomes.iter_mut());
        let (job, slot) = pairs.next().expect("at least one job");
        for (job, slot) in pairs {
            scope.spawn(move || *slot = Some(work(job)));
        }
        *slot = Some(work(job));
    });
    outcomes
        .into_iter()
        .map(|outcome| outcome.expect("every job ran"))
        .collect()
}

/// `work` on each item, in contiguous chunks on up to `threads` threads, the
/// outcomes in the items' order. Every item is worked on, whatever any
/// other's outcome.
pub(crate) fn map<I: Sync, T: Send>(
    items: &[I],
    threads: usize,
    work: &(impl Fn(&I) -> T + Sync),
) -> Vec<T> {
    if items.is_empty() {
        return Vec::new();
    }
    let chunk = items.len().div_ceil(threads.clamp(1, items.len()));
    on_threads(items.chunks(chunk), &|chunk: &[I]| {
        chunk.iter().map(work).collect::<Vec<T>>()
    })
    .into_iter()
    .flatten()
    .collect()
}
