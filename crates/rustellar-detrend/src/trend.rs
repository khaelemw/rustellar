//! The trend: each point's window of fluxes summarised by a location.

use std::ops::Range;

use crate::{
    biweight::{Scratch, biweight_sorted},
    exact_sum::ExactSum,
    location::{Location, midpoint},
    window::{WindowOptions, windows},
};

/// The trend at each point: `location` of the fluxes in its window, or
/// `None` where it has no window (the edge cutoff) or its window holds no
/// point (all masked). The points are shared among `threads` (at least 1)
/// threads, each a block of consecutive points; every result is the same
/// for any number of threads.
pub(crate) fn trend(
    times: &[f64],
    flux: &[f64],
    masked: Option<&[bool]>,
    options: &WindowOptions,
    location: Location,
    threads: usize,
) -> Vec<Option<f64>> {
    debug_assert_eq!(times.len(), flux.len());
    debug_assert!(threads > 0);
    let w = windows(times, masked, options);
    let kept: Vec<f64> = w.kept.iter().map(|&k| flux[k]).collect();
    let mut out = vec![None; w.windows.len()];
    let block = w.windows.len().div_ceil(threads).max(1);
    if block >= w.windows.len() {
        locate(&kept, &w.windows, location, &mut out);
        return out;
    }
    // A thread that panics makes the scope panic once all are joined.
    std::thread::scope(|scope| {
        for (windows, out) in w.windows.chunks(block).zip(out.chunks_mut(block)) {
            let kept = &kept;
            scope.spawn(move || locate(kept, windows, location, out));
        }
    });
    out
}

/// `location` of each window of `kept` into `out`. The windows only move
/// forward, so what they hold slides: the points that enter added, those
/// that leave taken away.
fn locate(
    kept: &[f64],
    windows: &[Option<Range<usize>>],
    location: Location,
    out: &mut [Option<f64>],
) {
    match location {
        Location::Mean => {
            let mut sum = Sliding::<ExactSum>::default();
            for (window, out) in windows.iter().zip(out) {
                *out = nonempty(window).map(|window| {
                    sum.slide(kept, &window);
                    sum.held.mean(window.len() as u64)
                });
            }
        }
        Location::Median | Location::Biweight(_) => {
            let mut sorted = Sliding::<Sorted>::default();
            let mut scratch = Scratch::default();
            for (window, out) in windows.iter().zip(out) {
                *out = nonempty(window).map(|window| {
                    sorted.slide(kept, &window);
                    let values = &sorted.held.0[..];
                    match location {
                        Location::Biweight(c) => biweight_sorted(values, &mut scratch, c),
                        _ => {
                            let n = values.len();
                            midpoint(values[(n - 1) / 2], values[n / 2])
                        }
                    }
                });
            }
        }
    }
}

fn nonempty(window: &Option<Range<usize>>) -> Option<Range<usize>> {
    window.clone().filter(|r| !r.is_empty())
}

/// What a sliding window holds: values can be added and taken away.
trait Holds: Default {
    fn clear(&mut self);
    fn add(&mut self, x: f64);
    fn remove(&mut self, x: f64);
}

impl Holds for ExactSum {
    fn clear(&mut self) {
        ExactSum::clear(self);
    }
    fn add(&mut self, x: f64) {
        ExactSum::add(self, x);
    }
    fn remove(&mut self, x: f64) {
        ExactSum::remove(self, x);
    }
}

/// The values, ascending by `total_cmp`.
#[derive(Default)]
struct Sorted(Vec<f64>);

impl Holds for Sorted {
    fn clear(&mut self) {
        self.0.clear();
    }
    fn add(&mut self, x: f64) {
        let at = self.0.partition_point(|v| v.total_cmp(&x).is_le());
        self.0.insert(at, x);
    }
    fn remove(&mut self, x: f64) {
        let at = self.0.partition_point(|v| v.total_cmp(&x).is_lt());
        debug_assert!(self.0[at].to_bits() == x.to_bits());
        self.0.remove(at);
    }
}

/// `kept[from..to]` held, moved to each window in turn.
#[derive(Default)]
struct Sliding<H> {
    held: H,
    from: usize,
    to: usize,
}

impl<H: Holds> Sliding<H> {
    fn slide(&mut self, kept: &[f64], window: &Range<usize>) {
        debug_assert!(window.start >= self.from && window.end >= self.to);
        if window.start >= self.to {
            // Nothing in common: start afresh.
            self.held.clear();
            (self.from, self.to) = (window.start, window.start);
        }
        for &x in &kept[self.from..window.start] {
            self.held.remove(x);
        }
        for &x in &kept[self.to..window.end] {
            self.held.add(x);
        }
        (self.from, self.to) = (window.start, window.end);
    }
}
