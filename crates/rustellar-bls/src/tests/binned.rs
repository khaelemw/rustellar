// The binned search: equal to fitting each box point by point where every
// point sits on a bin edge, never above the exact search, and below it by at
// most half a bin. (Against astropy: the integration tests.)

use rustellar_core::{Epoch, LightCurve, TimeDelta};

use super::recipe::{Random, Recipe, Survey};
use crate::{
    BlsError, Objective, TransitBox,
    binned::{BinnedBox, Bounds, Scratch, WINDOW, best_binned, bin_count, box_at, point_totals},
    box_fit::{AtPeriod, Weighted},
    totals::{Orbit, Total},
};

// ==========================
// Helper Functions
// ==========================

fn binned(lc: &LightCurve, period: f64, durations: &[f64], oversample: usize) -> Option<BinnedBox> {
    best_binned(
        &Weighted::new(lc).unwrap(),
        period,
        durations,
        oversample,
        Objective::DeltaChi2,
        &mut Scratch::default(),
    )
    .unwrap()
}

fn exact(lc: &LightCurve, period: f64, duration: f64) -> Option<f64> {
    TransitBox::best(
        lc,
        TimeDelta::from_days(period).unwrap(),
        TimeDelta::from_days(duration).unwrap(),
    )
    .unwrap()
    .map(|b| b.fit.delta_chi2)
}

// The box reported, fitted point by point, holds the same points with the
// same fit, though points may sit exactly on its bins' edges.
fn holds_its_own_points(lc: &LightCurve, ours: &BinnedBox, period: f64) {
    let transit = TransitBox::new(
        TimeDelta::from_days(period).unwrap(),
        TimeDelta::from_days(ours.bins as f64 * ours.bin).unwrap(),
        lc.time()[0] + TimeDelta::from_days(ours.centre).unwrap(),
    )
    .unwrap();
    let fit = transit.fit(lc).unwrap();
    assert_eq!(fit.n_in, ours.fit.n_in, "P {period}: {ours:?}");
    assert!(
        (fit.delta_chi2 / ours.fit.delta_chi2 - 1.0).abs() < 1e-12,
        "P {period}: {fit:?} vs {ours:?}"
    );
}

// ==========================
// Point by point, where points sit on bin edges
// ==========================

// Times on a grid of 1/64 day, the bin width, and a period of 160 bins: every
// phase is a whole number of bins, exactly. A box of k bins from bin n then
// holds exactly the points a box of duration k/64 centred at bin n + (k-1)/2
// holds, so the binned search must find the best of those boxes, fitted
// point by point.
#[test]
fn equals_fitting_each_box_point_by_point() {
    let bin = 1.0 / 64.0;
    let period = 160.0 * bin;
    let durations = [4.0 * bin, 8.0 * bin, 16.0 * bin];
    let mut rng = Random(5);
    for _ in 0..40 {
        // About 300 of the grid times over 20 days, with a dip near a random
        // phase.
        let centre = rng.unit() * period;
        let (mut time, mut flux, mut err) = (vec![], vec![], vec![]);
        for j in 0..1280 {
            if rng.unit() < 0.25 {
                let t = 1325.0 + f64::from(j) * bin;
                let phase = (f64::from(j) * bin) % period;
                let dip = (phase - centre).abs() < 0.08;
                time.push(t);
                flux.push(1e-3 * (rng.unit() - 0.5) - if dip { 2e-3 } else { 0.0 });
                err.push(1e-3 * (0.5 + rng.unit()));
            }
        }
        let lc = LightCurve::from_btjd(time, flux, err).unwrap();
        let ours = binned(&lc, period, &durations, 4).unwrap();
        holds_its_own_points(&lc, &ours, period);

        let first = lc.time()[0].as_btjd();
        let mut best = 0.0_f64;
        for &d in &durations {
            let k = (d / bin).round();
            for n in 0..160 {
                let centre = first + (f64::from(n) + 0.5 * (k - 1.0)) * bin;
                let transit = TransitBox::new(
                    TimeDelta::from_days(period).unwrap(),
                    TimeDelta::from_days(d).unwrap(),
                    Epoch::from_btjd(centre).unwrap(),
                )
                .unwrap();
                if let Ok(fit) = transit.fit(&lc) {
                    if fit.depth > 0.0 {
                        best = best.max(fit.delta_chi2);
                    }
                }
            }
        }
        assert!(
            (ours.fit.delta_chi2 / best - 1.0).abs() < 1e-13,
            "{} vs {best}",
            ours.fit.delta_chi2
        );
    }
}

// Four bins of a quarter of a day: a box as long as the period would hold
// every point, so only the one-bin boxes are tried.
#[test]
fn a_box_of_every_bin_is_skipped() {
    let time: Vec<f64> = (0..40).map(|i| 0.1 * f64::from(i)).collect();
    let flux = time
        .iter()
        .map(|t| if t % 1.0 < 0.2 { 0.99 } else { 1.0 })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 40]).unwrap();
    let both = binned(&lc, 1.0, &[0.25, 1.0], 1).unwrap();
    assert_eq!(both, binned(&lc, 1.0, &[0.25], 1).unwrap());
    assert_eq!((both.start, both.bins, both.duration), (0, 1, 0));
}

// Weights of 1e300 and a dip of 1e5: the likelihood is about 1e310.
#[test]
fn statistics_that_overflow() {
    let lc =
        LightCurve::from_btjd(vec![0.0, 1.0, 2.0], vec![1e5, 0.0, 1e5], vec![1e-150; 3]).unwrap();
    assert_eq!(
        best_binned(
            &Weighted::new(&lc).unwrap(),
            10.0,
            &[0.5],
            1,
            Objective::DeltaChi2,
            &mut Scratch::default()
        ),
        Err(BlsError::OutOfRange)
    );
}

// ==========================
// The shared recipe
// ==========================

// ==========================
// Against astropy's fast search
// ==========================

// ==========================
// Where astropy's wrap-around goes wrong
// ==========================

// ==========================
// Bounded by the exact search
// ==========================

// The bins tile the orbit, so every box of k bins is k b long wherever it
// starts, and holds what a box of duration k b holds (slide it left by less
// than the gap before its first point): the binned search is never above the
// exact search at duration k b. 10,000 cases: 25 light curves, 100 periods,
// 4 durations each.
#[test]
fn never_above_the_exact_search() {
    let mut rng = Random(9);
    let mut cases = 0;
    for curve in 0..25 {
        let period = 0.8 + 3.0 * rng.unit();
        let lc = Recipe {
            seed: 1000 + curve,
            n: 300,
            t_start: 1325.0,
            cadence: 0.05,
            noise: 1e-3,
            period,
            tc: 1325.0 + period * rng.unit(),
            duration: 0.05 + 0.1 * rng.unit(),
            depth: 2e-3 * rng.unit(),
        }
        .light_curve();
        for _ in 0..100 {
            let trial = 0.5 + 4.0 * rng.unit();
            for d in [0.04, 0.07, 0.1, 0.15] {
                cases += 1;
                // One duration at a time: bins of at most d / 5, and boxes of
                // 5 bins, since d / b < 5 + d / P rounds to 5.
                let Some(ours) = binned(&lc, trial, &[d], 5) else {
                    continue;
                };
                assert_eq!(ours.bins, 5);
                let exact = exact(&lc, trial, 5.0 * ours.bin)
                    .expect("the exact search finds every set the binned one does");
                assert!(
                    ours.fit.delta_chi2 <= exact * (1.0 + 1e-12),
                    "P {trial}, D {d}: {} above {exact}",
                    ours.fit.delta_chi2
                );
            }
        }
    }
    assert_eq!(cases, 10_000);
}

// How far below the exact search: a noiseless transit of depth 0.01 and
// duration T, sampled every 1/1024 day with equal errors, searched with a box
// of length L = k b. Put a box of length L where it overlaps the transit by
// `o`: its depth is 0.01 (o / L - (T - o) / (P - L)), and its snr is that
// times a factor that depends on L alone. The exact search overlaps by
// min(T, L). The binned box's start is on a grid of b, so it can come within
// (b - |T - L|) / 2 of an edge where it overlaps fully, if that is positive:
// it overlaps by at least min(T, L) - max(0, b - |T - L|) / 2. Counting
// points rather than lengths moves each count by at most a point at each end
// in each transit, so the lengths are given two cadences of slack. For
// T = L that is an snr of about 1 - 1 / (2 k) of the exact search's: 95% at
// k = 10.
#[test]
fn below_the_exact_search_by_at_most_half_a_bin() {
    let cadence = 1.0 / 1024.0;
    let time: Vec<f64> = (0..10_240)
        .map(|i| 1325.0 + cadence * f64::from(i))
        .collect();
    let mut rng = Random(79);
    for oversample in [2, 3, 5, 10] {
        for _ in 0..30 {
            let period = 1.0 + 3.0 * rng.unit();
            let transit = 0.05 + 0.1 * rng.unit();
            let tc = 1325.0 + period * rng.unit();
            let flux = time
                .iter()
                .map(|t| {
                    let r = (t - tc) / period;
                    let f = r - r.floor();
                    if f.min(1.0 - f) * period < 0.5 * transit {
                        -1e-2
                    } else {
                        0.0
                    }
                })
                .collect();
            let lc = LightCurve::from_btjd(time.clone(), flux, vec![1e-3; time.len()]).unwrap();
            let duration = transit * (0.6 + 0.8 * rng.unit());
            let ours = binned(&lc, period, &[duration], oversample).unwrap();
            let length = ours.bins as f64 * ours.bin;
            let exact = exact(&lc, period, length).unwrap();

            let depth = |o: f64| o / length - (transit - o) / (period - length);
            let full = transit.min(length);
            let short = full - (ours.bin - (transit - length).abs()).max(0.0) / 2.0;
            let bound = depth(short - 2.0 * cadence) / depth(full + 2.0 * cadence);
            let ratio = (ours.fit.delta_chi2 / exact).sqrt();
            assert!(
                ratio >= bound && ratio <= 1.0 + 1e-12,
                "oversample {oversample}, P {period}: {ratio} vs {bound}"
            );
        }
    }
}

#[test]
fn flat_flux_has_no_dip() {
    let time: Vec<f64> = (0..50).map(|i| 0.1 * f64::from(i)).collect();
    let lc = LightCurve::from_btjd(time, vec![1.0; 50], vec![1e-3; 50]).unwrap();
    assert_eq!(binned(&lc, 1.3, &[0.2], 4), None);
}

// Most of the weight in one short stretch of phase, with the dip there: the
// box holds more than half the weight, so the weight out of it is taken from
// the running totals rather than from the orbit's weight less the box's. The
// fit is the one-box fit of the same points, and the exact search finds the
// same box.
#[test]
fn a_box_with_most_of_the_weight() {
    let mut time = vec![];
    let mut flux = vec![];
    let mut err = vec![];
    for i in 0..40 {
        // Ten points an orbit at phases 0.40 to 0.49 with small errors, and
        // two at 0.1 and 0.8 with large ones.
        let orbit = f64::from(i / 12);
        let k = i % 12;
        let (phase, f, e) = match k {
            0 => (0.1, 1.0, 1e-2),
            11 => (0.8, 1.0, 1e-2),
            _ => (0.39 + 0.01 * f64::from(k), 0.99, 1e-3),
        };
        time.push(1325.0 + orbit + phase);
        flux.push(f);
        err.push(e);
    }
    let lc = LightCurve::from_btjd(time, flux, err).unwrap();
    let ours = binned(&lc, 1.0, &[0.25], 5).unwrap();
    let total: f64 = lc.flux_err().iter().map(|e| 1.0 / (e * e)).sum();
    assert!(ours.fit.weight_in > 0.5 * total, "{ours:?}");
    assert!((ours.fit.depth - 0.01).abs() < 1e-12, "{ours:?}");
    let exact = exact(&lc, 1.0, 0.25).unwrap();
    assert!((ours.fit.delta_chi2 / exact - 1.0).abs() < 1e-13);
}

// One point of weight 1e300 in the box and one of weight 1 out of it, 1e5
// apart: delta_chi2 is about 1e10, but the likelihood, 0.5 W_in depth^2, is
// about 5e309. The search picks the box by delta_chi2; its fit overflows.
#[test]
fn best_box_whose_fit_overflows() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![0.0, 1e5], vec![1e-150, 1.0]).unwrap();
    assert_eq!(
        best_binned(
            &Weighted::new(&lc).unwrap(),
            10.0,
            &[0.5],
            1,
            Objective::DeltaChi2,
            &mut Scratch::default()
        ),
        Err(BlsError::OutOfRange)
    );
    assert_eq!(
        TransitBox::best(
            &lc,
            TimeDelta::from_days(10.0).unwrap(),
            TimeDelta::from_days(0.5).unwrap()
        ),
        Err(BlsError::OutOfRange)
    );
}

// ==========================
// Skipping blocks of boxes
// ==========================

// Running totals twice round over `m` bins with weights in 0.5..2 (a fifth
// of the bins empty), centred fluxes of either sign, and a dip.
fn random_totals(m: usize, seed: u64) -> Vec<Total> {
    let mut random = Random(seed);
    let bins: Vec<(f64, f64)> = (0..m)
        .map(|k| {
            if random.unit() < 0.2 {
                return (0.0, 0.0);
            }
            let w = 0.5 + 1.5 * random.unit();
            let dip = if (k as f64 / m as f64 - 0.3).abs() < 0.05 {
                -3.0
            } else {
                0.0
            };
            (w, w * (random.unit() - 0.5 + dip))
        })
        .collect();
    let mut total = Total::default();
    let mut totals = vec![total];
    for &(w, f) in bins.iter().chain(&bins) {
        total.weight.add(w);
        total.flux.add(f);
        totals.push(total);
    }
    totals
}

// The bound of every block, at every length and block size, is never below
// the score of a box in it; zero, finite and infinite bounds all happen.
#[test]
fn a_bound_is_never_below_a_score() {
    let mut seen = [0; 3];
    for objective in [
        Objective::DeltaChi2,
        Objective::Snr,
        Objective::AstropyLikelihood,
    ] {
        for seed in 0..4 {
            let m = 160;
            let totals = random_totals(m, seed);
            let orbit = Orbit::new(&totals[0], &totals[m], objective);
            let mut bounds = Bounds::default();
            bounds.prepare(&totals, &orbit);
            for length in [1, 2, 5, 31, 32, 33, 64, 100, 159] {
                for size in [1, 7, 32] {
                    for from in (0..m).step_by(size) {
                        let to = (from + size).min(m);
                        let bound = bounds.bound(from, to, length);
                        seen[usize::from(bound > 0.0) + usize::from(bound.is_infinite())] += 1;
                        for start in from..to {
                            let score = orbit
                                .statistic(
                                    &totals[start],
                                    &totals[start + length],
                                    &totals[start + m],
                                )
                                .unwrap();
                            assert!(
                                score <= bound,
                                "{objective:?} {seed}: {score} above {bound} ({length}, {from}..{to})"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        seen.iter().all(|&n| n > 0),
        "zero, finite and infinite: {seen:?}"
    );
}

// The best box by brute force: the points binned as the search bins them,
// every box scored on its own, the first of the highest over the durations
// and then the starts.
fn brute_force(
    lc: &LightCurve,
    period: f64,
    durations: &[f64],
    objective: Objective,
) -> Option<(usize, usize, usize)> {
    let weighted = Weighted::new(lc).unwrap();
    let m = (bin_count(period, durations, 10) as usize).max(1);
    let bin = period / m as f64;
    let at = AtPeriod::new(period);
    let mut bins = vec![(0.0, 0.0); m];
    let points = weighted
        .elapsed
        .iter()
        .zip(&weighted.weights)
        .zip(&weighted.centred);
    for ((&(s, e), &w), &c) in points {
        let mut phase = at.offset_after_start(s, e);
        if phase < 0.0 {
            phase += period;
        }
        let j = ((phase / bin) as u32 as usize).min(m - 1);
        bins[j].0 += w;
        bins[j].1 += c;
    }
    let mut total = Total::default();
    let mut totals = vec![total];
    for &(w, f) in bins.iter().chain(&bins) {
        total.weight.add(w);
        total.flux.add(f);
        totals.push(total);
    }
    let orbit = Orbit::new(&totals[0], &totals[m], objective);
    let mut best: Option<(f64, usize, usize, usize)> = None;
    for (duration, &d) in durations.iter().enumerate() {
        let length = (d / bin).round() as usize;
        if length >= m {
            continue;
        }
        for start in 0..m {
            let score = orbit
                .statistic(&totals[start], &totals[start + length], &totals[start + m])
                .unwrap();
            if score > best.map_or(0.0, |b| b.0) {
                best = Some((score, start, length, duration));
            }
        }
    }
    best.map(|(_, start, length, duration)| (start, length, duration))
}

// Skipping blocks finds the brute force's box, with one window of bins and
// with many, on a light curve with gaps, a transit and outliers.
#[test]
fn skipping_finds_the_brute_forces_box() {
    let mut random = Random(91);
    let mut time = Vec::new();
    for segment in 0..4 {
        let start = 1300.0 + 31.0 * f64::from(segment);
        for k in 0..1500 {
            time.push(start + f64::from(k) * 0.0139);
        }
    }
    let flux: Vec<f64> = time
        .iter()
        .map(|t| {
            let dip = if ((t - 1301.3) % 7.37) < 0.15 {
                2e-3
            } else {
                0.0
            };
            let outlier = if random.unit() < 0.002 { 5e-3 } else { 0.0 };
            1.0 - dip + outlier + 1e-3 * (random.unit() - 0.5)
        })
        .collect();
    let error: Vec<f64> = time
        .iter()
        .map(|_| 1e-3 * (0.8 + 0.4 * random.unit()))
        .collect();
    let lc = LightCurve::from_btjd(time, flux, error).unwrap();
    let durations = [0.01, 0.04, 0.08, 0.15, 0.3];
    let mut windows = [0, 0];
    for objective in [
        Objective::DeltaChi2,
        Objective::Snr,
        Objective::AstropyLikelihood,
    ] {
        for period in [0.7, 3.1, 7.37, 29.0, 61.0] {
            let m = bin_count(period, &durations, 10) as usize;
            windows[usize::from(m > WINDOW)] += 1;
            let ours = best_binned(
                &Weighted::new(&lc).unwrap(),
                period,
                &durations,
                10,
                objective,
                &mut Scratch::default(),
            )
            .unwrap()
            .map(|b| (b.start, b.bins, b.duration));
            assert_eq!(
                ours,
                brute_force(&lc, period, &durations, objective),
                "{objective:?} at {period}"
            );
        }
    }
    assert!(
        windows.iter().all(|&n| n > 0),
        "one window and many: {windows:?}"
    );
}

// A period so short that the light curve holds 5 x 10^7 orbits, past the
// 2^25 below which the bins are filled by the shorter path: the longer one
// finds the brute force's box too.
#[test]
fn bins_of_many_orbits_find_the_brute_forces_box() {
    let mut random = Random(17);
    let time: Vec<f64> = (0..2000).map(|k| 1300.0 + 0.05 * f64::from(k)).collect();
    let flux: Vec<f64> = time
        .iter()
        .map(|_| 1.0 + 1e-3 * (random.unit() - 0.5))
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 2000]).unwrap();
    let (period, durations) = (2e-6, [2e-7, 5e-7]);
    let weighted = Weighted::new(&lc).unwrap();
    let last = weighted.elapsed.last().unwrap().0;
    assert!(!AtPeriod::new(period).short_up_to(last));
    for objective in [
        Objective::DeltaChi2,
        Objective::Snr,
        Objective::AstropyLikelihood,
    ] {
        let ours = best_binned(
            &weighted,
            period,
            &durations,
            10,
            objective,
            &mut Scratch::default(),
        )
        .unwrap()
        .map(|b| (b.start, b.bins, b.duration));
        assert!(ours.is_some(), "{objective:?}");
        assert_eq!(
            ours,
            brute_force(&lc, period, &durations, objective),
            "{objective:?}"
        );
    }
}

// ==========================================
// The best box's points, found lazily
// ==========================================

// The best box's number of points and centre as the search computed them
// when every bin kept its count and phase range: every point into its bin,
// then the box's first and last points and its neighbours outside.
fn eager(weighted: &Weighted, period: f64, m: usize, b: &BinnedBox) -> (usize, f64) {
    let bin = period / m as f64;
    let at = AtPeriod::new(period);
    let mut bins = vec![(0_usize, f64::INFINITY, f64::NEG_INFINITY); m];
    for &(s, e) in &weighted.elapsed {
        let mut phase = at.offset_after_start(s, e);
        if phase < 0.0 {
            phase += period;
        }
        let slot = &mut bins[((phase / bin) as u32 as usize).min(m - 1)];
        slot.0 += 1;
        slot.1 = slot.1.min(phase);
        slot.2 = slot.2.max(phase);
    }
    let n_in = (b.start..b.start + b.bins).map(|k| bins[k % m].0).sum();
    let span = b.bins as f64 * bin;
    let round = |j: usize| (j / m) as f64 * period;
    let full = |j: &usize| bins[j % m].0 > 0;
    let low = |j: usize| bins[j % m].1 + round(j);
    let high = |j: usize| bins[j % m].2 + round(j);
    let start = b.start + m;
    let inside = start..start + b.bins;
    let first = low(inside.clone().find(full).unwrap());
    let last = high(inside.rev().find(full).unwrap());
    let next = low((start + b.bins..start + m).find(full).unwrap());
    let previous = high((start + b.bins - m..start).rev().find(full).unwrap());
    let lower = previous.max(last - span);
    let upper = first.min(next - span);
    (n_in, 0.5 * (lower + upper) + 0.5 * span - period)
}

// Light curves with gaps or at random times, a dip at the first point (so
// the best box runs past the orbit's end) or elsewhere; periods with one
// window of bins and with many, and with more orbits than the shorter fold
// allows: the count and centre are the eager ones to the bit.
#[test]
fn the_best_boxs_points_are_the_eager_ones() {
    let mut random = Random(5);
    let gappy: Vec<f64> = (0..4)
        .flat_map(|g| (0..1500).map(move |k| 1300.0 + 31.0 * f64::from(g) + 0.0139 * f64::from(k)))
        .collect();
    let mut scattered: Vec<f64> = (0..3000).map(|_| 1300.0 + 120.0 * random.unit()).collect();
    scattered.sort_by(f64::total_cmp);
    let (mut wraps, mut windows) = (0, [0, 0]);
    for time in [gappy, scattered] {
        for period in [0.37, 3.1, 7.37, 29.0, 61.0, 2e-6] {
            for (fractions, dip_at) in [
                ([0.002, 0.02, 0.06], 0.0),
                ([0.002, 0.02, 0.06], 0.4),
                ([0.01, 0.03, 0.08], 0.0),
                ([0.01, 0.03, 0.08], 0.4),
            ] {
                let durations = fractions.map(|f| f * period);
                let flux: Vec<f64> = time
                    .iter()
                    .map(|&t| {
                        let phase = (t - time[0] - dip_at * period).rem_euclid(period) / period;
                        let dip = if phase.min(1.0 - phase) < 0.01 {
                            3e-3
                        } else {
                            0.0
                        };
                        1.0 - dip + 1e-3 * (random.unit() - 0.5)
                    })
                    .collect();
                let error = vec![1e-3; time.len()];
                let lc = LightCurve::from_btjd(time.clone(), flux, error).unwrap();
                let weighted = Weighted::new(&lc).unwrap();
                let m = bin_count(period, &durations, 10) as usize;
                windows[usize::from(m > WINDOW)] += 1;
                for objective in [
                    Objective::DeltaChi2,
                    Objective::Snr,
                    Objective::AstropyLikelihood,
                ] {
                    let b = best_binned(
                        &weighted,
                        period,
                        &durations,
                        10,
                        objective,
                        &mut Scratch::default(),
                    )
                    .unwrap()
                    .unwrap();
                    wraps += usize::from(b.start + b.bins > m);
                    let (n_in, centre) = eager(&weighted, period, m, &b);
                    assert_eq!(b.fit.n_in, n_in, "{period} {dip_at} {objective:?}");
                    assert_eq!(
                        b.centre.to_bits(),
                        centre.to_bits(),
                        "{period} {dip_at} {objective:?}: {} against {centre}",
                        b.centre
                    );
                }
            }
        }
    }
    assert!(wraps > 0, "a box past the orbit's end");
    assert!(
        windows.iter().all(|&n| n > 0),
        "one window and many: {windows:?}"
    );
}

// A box built from the points it holds, at the binned search's own start
// and duration: the binned search's box, its points and middle to the bit
// and its fit to rounding. At other starts and durations: the points and
// middle of filling every bin.
#[test]
fn a_box_from_its_points_is_the_binned_box() {
    let dense = Survey {
        seed: 11,
        segments: vec![(1325.0, 6000), (1335.0, 6000)],
        cadence: 2.0 / 1440.0,
        noise: 1e-3,
        keep: 1.0,
        period: 1.3,
        tc: 1325.4,
        duration: 0.08,
        depth: 2e-3,
    }
    .light_curve();
    let sparse = Survey {
        seed: 12,
        segments: (0..6)
            .map(|q| (1325.0 + 90.0 * f64::from(q), 700))
            .collect(),
        cadence: 0.0204,
        noise: 3e-4,
        keep: 0.9,
        period: 23.3,
        tc: 1331.0,
        duration: 0.25,
        depth: 1e-3,
    }
    .light_curve();
    let durations = [0.02, 0.06, 0.12, 0.25];
    let mut random = Random(9);
    let mut others = 0;
    for lc in [dense, sparse] {
        let weighted = Weighted::new(&lc).unwrap();
        let points = point_totals(&weighted);
        let span = weighted.elapsed.last().unwrap().0;
        for k in 0..120 {
            let period = 0.6 * (span / 1.2).powf(f64::from(k) / 119.0);
            let m = bin_count(period, &durations, 10) as usize;
            let b = best_binned(
                &weighted,
                period,
                &durations,
                10,
                Objective::DeltaChi2,
                &mut Scratch::default(),
            )
            .unwrap()
            .unwrap();
            let c = box_at(
                &weighted, &points, period, &durations, 10, b.duration, b.start,
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                (c.start, c.bins, c.duration, c.fit.n_in, c.centre.to_bits()),
                (b.start, b.bins, b.duration, b.fit.n_in, b.centre.to_bits()),
                "{period}"
            );
            let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * y.abs();
            assert!(close(c.fit.delta_chi2, b.fit.delta_chi2), "{period}");
            assert!(close(c.fit.depth, b.fit.depth), "{period}");
            assert!(close(c.fit.weight_in, b.fit.weight_in), "{period}");

            let duration = (random.unit() * 4.0) as usize;
            let start = (random.unit() * m as f64) as usize;
            if let Some(c) =
                box_at(&weighted, &points, period, &durations, 10, duration, start).unwrap()
            {
                let (n_in, centre) = eager(&weighted, period, m, &c);
                assert_eq!(
                    (c.fit.n_in, c.centre.to_bits()),
                    (n_in, centre.to_bits()),
                    "{period} {start} {duration}"
                );
                others += 1;
            }
        }
    }
    assert!(others > 200, "{others} other boxes");
}
