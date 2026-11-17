//! The trend: each point's window of fluxes summarised by a location.

use crate::{
    biweight::biweight,
    exact_sum::ExactSum,
    location::{Location, median},
    window::{WindowOptions, windows},
};

/// The trend at each point: `location` of the fluxes in its window, or
/// `None` where it has no window (the edge cutoff) or its window holds no
/// point (all masked).
pub(crate) fn trend(
    times: &[f64],
    flux: &[f64],
    masked: Option<&[bool]>,
    options: &WindowOptions,
    location: Location,
) -> Vec<Option<f64>> {
    debug_assert_eq!(times.len(), flux.len());
    let w = windows(times, masked, options);
    let kept: Vec<f64> = w.kept.iter().map(|&k| flux[k]).collect();
    match location {
        Location::Median => {
            let mut scratch = Vec::new();
            w.windows
                .iter()
                .map(|window| {
                    let window = window.clone().filter(|r| !r.is_empty())?;
                    scratch.clear();
                    scratch.extend_from_slice(&kept[window]);
                    Some(median(&mut scratch))
                })
                .collect()
        }
        Location::Biweight(c) => {
            let (mut scratch, mut deviations) = (Vec::new(), Vec::new());
            w.windows
                .iter()
                .map(|window| {
                    let window = window.clone().filter(|r| !r.is_empty())?;
                    scratch.clear();
                    scratch.extend_from_slice(&kept[window]);
                    Some(biweight(&mut scratch, &mut deviations, c))
                })
                .collect()
        }
        Location::Mean => {
            // The windows only move forward, so the sum slides: the points
            // that enter added, those that leave taken away, exactly.
            let mut sum = ExactSum::default();
            let (mut from, mut to) = (0, 0);
            w.windows
                .iter()
                .map(|window| {
                    let window = window.clone().filter(|r| !r.is_empty())?;
                    if window.start >= to {
                        sum.clear();
                        (from, to) = (window.start, window.start);
                    }
                    for &x in &kept[from..window.start] {
                        sum.remove(x);
                    }
                    for &x in &kept[to..window.end] {
                        sum.add(x);
                    }
                    (from, to) = (window.start, window.end);
                    Some(sum.mean(window.len() as u64))
                })
                .collect()
        }
    }
}
