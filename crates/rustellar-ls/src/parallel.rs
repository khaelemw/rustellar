//! Jobs run side by side: each but the last on a thread of its own, the
//! last on the calling thread, so a single job starts no thread.

use std::thread;

/// Runs every job and returns their results in order.
pub(crate) fn run_all<T, F>(jobs: impl IntoIterator<Item = F>) -> Vec<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    let mut jobs: Vec<F> = jobs.into_iter().collect();
    let last = jobs.pop();
    if jobs.is_empty() {
        return last.into_iter().map(|job| job()).collect();
    }
    thread::scope(|scope| {
        let handles: Vec<_> = jobs.into_iter().map(|job| scope.spawn(job)).collect();
        let last = last.map(|job| job());
        handles
            .into_iter()
            .map(|h| h.join().expect("no panic"))
            .chain(last)
            .collect()
    })
}

/// No more threads than `work / grain` (at least one), and at most
/// `threads`.
pub(crate) fn threads_for(threads: usize, work: usize, grain: usize) -> usize {
    threads.min(work / grain).max(1)
}
