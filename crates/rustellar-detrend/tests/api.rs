//! The public interface: settings, checks, and the trend and flattened light
//! curve against definitions written out plainly.

use std::num::NonZeroUsize;

use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_detrend::{
    BreakTolerance, Detrend, DetrendError, Edges, Method, WINDOW_PER_DURATION, transit_mask,
};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn curve(time: &[f64], flux: &[f64]) -> LightCurve {
    LightCurve::from_btjd(time.to_vec(), flux.to_vec(), vec![0.5; time.len()]).unwrap()
}

/// xorshift64*: portable, so the data are the same everywhere.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Times on a grid of 1/64 day (exact), with gaps; small integer fluxes, so
/// sums and midpoints are exact.
fn grid(random: &mut Random, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut time = Vec::with_capacity(n);
    let mut t = 1000.0;
    for _ in 0..n {
        t += (1 + if random.below(40) == 0 {
            random.below(80)
        } else {
            random.below(3)
        }) as f64
            / 64.0;
        time.push(t);
    }
    let flux = (0..n).map(|_| 100.0 + random.below(20) as f64).collect();
    (time, flux)
}

/// The windows by definition: for each point, the unmasked fluxes within
/// half a window, in its segment, or `None` within the cutoff.
fn windows(
    time: &[f64],
    flux: &[f64],
    masked: &[bool],
    window: f64,
    tolerance: f64,
    cutoff: f64,
    edges: Edges,
) -> Vec<Option<Vec<f64>>> {
    let n = time.len();
    let mut segment = vec![0; n];
    for k in 1..n {
        segment[k] = segment[k - 1] + usize::from(time[k] - time[k - 1] > tolerance);
    }
    let half = window / 2.0;
    let cutoff = cutoff.min(half);
    (0..n)
        .map(|i| {
            let same: Vec<usize> = (0..n).filter(|&k| segment[k] == segment[i]).collect();
            let (first, last) = (time[same[0]], time[*same.last().unwrap()]);
            if time[i] - first < cutoff || last - time[i] < cutoff {
                return None;
            }
            Some(
                same.into_iter()
                    .filter(|&k| !masked[k])
                    .filter(|&k| {
                        let d = time[k] - time[i];
                        d.abs() < half || (d == -half) || (d == half && edges == Edges::Symmetric)
                    })
                    .map(|k| flux[k])
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn the_defaults() {
    let d = Detrend::default();
    assert_eq!(d.method, Method::Biweight { c: 5.0 });
    assert_eq!(d.window.as_days(), 0.5);
    assert_eq!(d.break_tolerance, BreakTolerance::HalfWindow);
    assert_eq!(d.edge_cutoff.as_days(), 0.0);
    assert_eq!(d.edges, Edges::Symmetric);
    assert!(d.threads.get() >= 1);
}

#[test]
fn the_checks_in_order() {
    let lc = curve(&[1.0, 2.0], &[1.0, 2.0]);
    let check = |d: Detrend, mask: Option<&[bool]>| d.trend(&lc, mask).err();
    let tuned: [fn(f64) -> Method; 4] = [
        |c| Method::Biweight { c },
        |c| Method::Welsch { c },
        |c| Method::AndrewsSineWave { c },
        |c| Method::Huber { c },
    ];
    for method in tuned {
        let tuning = |c| Detrend::default().with_method(method(c));
        for c in [f64::NAN, f64::INFINITY, -1.0, 0.0, 9e-31, 1.1e30] {
            assert_eq!(
                check(tuning(c), None),
                Some(DetrendError::TuningOutOfRange),
                "{c}"
            );
        }
        for c in [1e-30, 1e30] {
            assert_eq!(check(tuning(c), None), None);
        }
    }
    let proportioned: [fn(f64) -> Method; 2] = [
        |proportion| Method::TrimmedMean { proportion },
        |proportion| Method::WinsorizedMean { proportion },
    ];
    for method in proportioned {
        let cut = |p| Detrend::default().with_method(method(p));
        for p in [f64::NAN, f64::INFINITY, -1e-300, 0.5, 1.0] {
            assert_eq!(
                check(cut(p), None),
                Some(DetrendError::ProportionOutOfRange),
                "{p}"
            );
        }
        for p in [0.0, 0.1, 0.5 - f64::EPSILON / 4.0] {
            assert_eq!(check(cut(p), None), None);
        }
    }
    for w in [0.0, -1.0] {
        let d = Detrend::default().with_window(days(w));
        assert_eq!(
            check(d, None),
            Some(DetrendError::NotPositive { name: "window" })
        );
    }
    for b in [0.0, -1.0] {
        let d = Detrend::default().with_break_tolerance(BreakTolerance::Fixed(days(b)));
        let e = DetrendError::NotPositive {
            name: "break_tolerance",
        };
        assert_eq!(check(d, None), Some(e));
    }
    let d = Detrend::default().with_edge_cutoff(days(-1e-9));
    assert_eq!(check(d, None), Some(DetrendError::NegativeEdgeCutoff));
    for mask in [&[false][..], &[false; 3]] {
        assert_eq!(
            check(Detrend::default(), Some(mask)),
            Some(DetrendError::MaskLengthMismatch)
        );
    }

    // The first failing check wins.
    let all = Detrend::default()
        .with_method(Method::Biweight { c: 0.0 })
        .with_window(days(0.0))
        .with_break_tolerance(BreakTolerance::Fixed(days(0.0)))
        .with_edge_cutoff(days(-1.0));
    let mask: &[bool] = &[true];
    assert_eq!(check(all, Some(mask)), Some(DetrendError::TuningOutOfRange));
    let all = all.with_method(Method::TrimmedMean { proportion: 0.5 });
    assert_eq!(
        check(all, Some(mask)),
        Some(DetrendError::ProportionOutOfRange)
    );
    let all = all.with_method(Method::Median);
    assert_eq!(
        check(all, Some(mask)),
        Some(DetrendError::NotPositive { name: "window" })
    );
    let all = all.with_window(days(1.0));
    let e = DetrendError::NotPositive {
        name: "break_tolerance",
    };
    assert_eq!(check(all, Some(mask)), Some(e));
    let all = all.with_break_tolerance(BreakTolerance::Never);
    assert_eq!(
        check(all, Some(mask)),
        Some(DetrendError::NegativeEdgeCutoff)
    );
    let all = all.with_edge_cutoff(days(0.0));
    assert_eq!(
        check(all, Some(mask)),
        Some(DetrendError::MaskLengthMismatch)
    );
    // flatten checks the same, first.
    assert_eq!(
        all.flatten(&lc, Some(mask)),
        Err(DetrendError::MaskLengthMismatch)
    );
}

#[test]
fn the_messages() {
    let cases = [
        (
            DetrendError::NotPositive { name: "window" },
            "window must be greater than zero",
        ),
        (
            DetrendError::NegativeEdgeCutoff,
            "edge_cutoff must not be negative",
        ),
        (
            DetrendError::TuningOutOfRange,
            "the tuning constant c must be between 1e-30 and 1e30",
        ),
        (
            DetrendError::ProportionOutOfRange,
            "the proportion must be at least 0 and below 0.5",
        ),
        (
            DetrendError::MaskLengthMismatch,
            "the mask must have one entry per point of the light curve",
        ),
        (
            DetrendError::NoPointsLeft,
            "no point has a trend to flatten by",
        ),
    ];
    for (error, message) in cases {
        assert_eq!(error.to_string(), message);
        let _: &dyn std::error::Error = &error;
    }
}

#[test]
fn the_median_and_mean_by_definition() {
    let mut random = Random(0x5eed_0501);
    for case in 0..60 {
        let n = 1 + random.below(300) as usize;
        let (time, flux) = grid(&mut random, n);
        let masked: Vec<bool> = (0..time.len()).map(|_| random.below(5) == 0).collect();
        let window = (1 + random.below(64)) as f64 / 32.0;
        let (tolerance, setting) = match case % 3 {
            0 => (window / 2.0, BreakTolerance::HalfWindow),
            1 => (f64::INFINITY, BreakTolerance::Never),
            _ => {
                let b = (1 + random.below(64)) as f64 / 64.0;
                (b, BreakTolerance::Fixed(days(b)))
            }
        };
        let cutoff = random.below(48) as f64 / 64.0;
        let edges = if case % 2 == 0 {
            Edges::Symmetric
        } else {
            Edges::HalfOpen
        };
        let expected = windows(&time, &flux, &masked, window, tolerance, cutoff, edges);
        let lc = curve(&time, &flux);
        let mask = if case % 4 == 0 {
            None
        } else {
            Some(&masked[..])
        };
        let expected = if mask.is_none() {
            windows(
                &time,
                &flux,
                &vec![false; time.len()],
                window,
                tolerance,
                cutoff,
                edges,
            )
        } else {
            expected
        };
        let base = Detrend::default()
            .with_window(days(window))
            .with_break_tolerance(setting)
            .with_edge_cutoff(days(cutoff))
            .with_edges(edges);
        let median = base.with_method(Method::Median).trend(&lc, mask).unwrap();
        let mean = base.with_method(Method::Mean).trend(&lc, mask).unwrap();
        for i in 0..time.len() {
            let want = expected[i].as_ref().filter(|v| !v.is_empty()).map(|v| {
                let mut v = v.clone();
                v.sort_by(f64::total_cmp);
                let n = v.len();
                (
                    (v[(n - 1) / 2] + v[n / 2]) / 2.0,
                    v.iter().sum::<f64>() / n as f64,
                )
            });
            assert_eq!(median[i], want.map(|w| w.0), "case {case}, point {i}");
            assert_eq!(mean[i], want.map(|w| w.1), "case {case}, point {i}");
        }
    }
}

// Every method but the median and mean (checked by definition above).
const ESTIMATORS: [Method; 9] = [
    Method::Biweight { c: 5.0 },
    Method::Biweight { c: 3.0 },
    Method::Welsch { c: 2.11 },
    Method::AndrewsSineWave { c: 1.339 },
    Method::Huber { c: 1.28 },
    Method::HodgesLehmann,
    Method::TrimmedMean { proportion: 0.1 },
    Method::WinsorizedMean { proportion: 0.1 },
    Method::WinsorizedMean { proportion: 0.0 },
];

#[test]
fn each_window_alone() {
    // A window's location is the same whether found sliding or with the
    // window as the whole light curve.
    let mut random = Random(0x5eed_0502);
    let (time, flux) = grid(&mut random, 400);
    let flux: Vec<f64> = flux
        .iter()
        .map(|f| f * (1.0 + 1e-9 * random.below(1000) as f64))
        .collect();
    let expected = windows(
        &time,
        &flux,
        &vec![false; 400],
        0.75,
        0.375,
        0.0,
        Edges::Symmetric,
    );
    for method in ESTIMATORS {
        let d = Detrend::default()
            .with_window(days(0.75))
            .with_method(method);
        let trend = d.trend(&curve(&time, &flux), None).unwrap();
        let whole = d
            .with_window(days(1e6))
            .with_break_tolerance(BreakTolerance::Never);
        for (i, values) in expected.iter().enumerate() {
            let values = values.as_ref().unwrap();
            let time: Vec<f64> = (0..values.len()).map(|k| k as f64).collect();
            let alone = whole.trend(&curve(&time, values), None).unwrap();
            assert_eq!(
                trend[i].unwrap().to_bits(),
                alone[0].unwrap().to_bits(),
                "{method:?}, point {i}"
            );
        }
    }
}

#[test]
fn any_number_of_threads() {
    let mut random = Random(0x5eed_0503);
    let (time, flux) = grid(&mut random, 9000);
    let flux: Vec<f64> = flux
        .iter()
        .map(|f| f * (1.0 + 1e-7 * random.below(1000) as f64))
        .collect();
    let masked: Vec<bool> = (0..time.len()).map(|_| random.below(7) == 0).collect();
    let lc = curve(&time, &flux);
    for method in [Method::Median, Method::Mean].into_iter().chain(ESTIMATORS) {
        let d = Detrend::default().with_method(method);
        let one = d
            .with_threads(NonZeroUsize::MIN)
            .trend(&lc, Some(&masked))
            .unwrap();
        for threads in [2, 3, 8, 64] {
            let d = d.with_threads(NonZeroUsize::new(threads).unwrap());
            assert_eq!(
                d.trend(&lc, Some(&masked)).unwrap(),
                one,
                "{method:?}, {threads}"
            );
        }
    }
}

#[test]
fn the_break_tolerance_settings() {
    let mut random = Random(0x5eed_0504);
    let (time, flux) = grid(&mut random, 500);
    let lc = curve(&time, &flux);
    let d = Detrend::default().with_window(days(0.5));
    let trend = |b| d.with_break_tolerance(b).trend(&lc, None).unwrap();
    assert_eq!(
        trend(BreakTolerance::HalfWindow),
        trend(BreakTolerance::Fixed(days(0.25)))
    );
    assert_eq!(
        trend(BreakTolerance::Never),
        trend(BreakTolerance::Fixed(days(1e9)))
    );
    // A gap of 1/8 day, within half the window: split only by a smaller
    // tolerance. (Across a gap longer than half the window no point is in
    // another's window, so there splitting changes only the edge cutoff.)
    let time: Vec<f64> = (0..128)
        .map(|k| k as f64 / 64.0 + if k >= 64 { 0.125 } else { 0.0 })
        .collect();
    let flux: Vec<f64> = (0..128).map(|k| if k >= 64 { 2.0 } else { 1.0 }).collect();
    let lc = curve(&time, &flux);
    // The mean: the biweight of two flat levels is the nearer one's anyway.
    let d = d.with_method(Method::Mean);
    let trend = |b| d.with_break_tolerance(b).trend(&lc, None).unwrap();
    let apart = trend(BreakTolerance::Fixed(days(0.1)));
    assert!(apart.iter().zip(&flux).all(|(t, f)| *t == Some(*f)));
    assert_ne!(trend(BreakTolerance::Never), apart);
    assert_eq!(
        trend(BreakTolerance::Never),
        trend(BreakTolerance::HalfWindow)
    );
}

#[test]
fn flattening() {
    let time = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
    let flux = [3.0, -3.0, 0.0, 0.0, 6.0, 6.0, 6.0];
    let lc = LightCurve::from_btjd(time.to_vec(), flux.to_vec(), vec![0.3; 7]).unwrap();
    // Each point with its two neighbours (one at the ends).
    let d = Detrend::default()
        .with_method(Method::Mean)
        .with_window(days(0.25));
    let flat = d.flatten(&lc, None).unwrap();
    assert_eq!(
        flat.trend(),
        [
            Some(0.0),
            Some(0.0),
            Some(-1.0),
            Some(2.0),
            Some(4.0),
            Some(6.0),
            Some(6.0)
        ]
    );
    // A zero trend leaves its points out; a negative one divides the error
    // by its magnitude.
    assert_eq!(flat.indices(), [2, 3, 4, 5, 6]);
    let lc = flat.light_curve();
    assert_eq!(lc.time()[0].as_btjd(), 0.25);
    assert_eq!(lc.flux(), [-0.0, 0.0, 1.5, 1.0, 1.0]);
    assert_eq!(
        lc.flux_err(),
        [0.3, 0.3 / 2.0, 0.3 / 4.0, 0.3 / 6.0, 0.3 / 6.0]
    );
    assert_eq!(flat.clone().into_light_curve(), *flat.light_curve());

    // A flux far above its trend overflows: left out.
    let lc = curve(&[0.0, 0.125, 0.25, 0.375], &[1e300, 1e-10, 1e-10, 1e-10]);
    let d = Detrend::default().with_method(Method::Median);
    let flat = d.flatten(&lc, None).unwrap();
    assert_eq!(flat.trend()[0], Some(1e-10));
    assert_eq!(flat.indices(), [1, 2, 3]);

    // An error far below its trend underflows: left out.
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1e300, 1e300], vec![1e-300, 1.0]).unwrap();
    let flat = Detrend::default().flatten(&lc, None).unwrap();
    assert_eq!(flat.indices(), [1]);

    // Nothing left: every point masked, or every trend zero.
    let lc = curve(&[0.0, 1.0], &[0.0, 0.0]);
    assert_eq!(
        Detrend::default().flatten(&lc, None),
        Err(DetrendError::NoPointsLeft)
    );
    let flat = Detrend::default().flatten(&lc, Some(&[true, true]));
    assert_eq!(flat, Err(DetrendError::NoPointsLeft));
    assert_eq!(
        Detrend::default().trend(&lc, Some(&[true, true])),
        Ok(vec![None, None])
    );
}

#[test]
fn a_masked_transit_is_bridged() {
    // A dip masked out: the trend passes under it, and the flattened dip
    // keeps its full depth.
    let time: Vec<f64> = (0..720).map(|i| i as f64 / 360.0).collect();
    let dip = |t: f64| (0.9..1.0).contains(&t);
    let flux: Vec<f64> = time
        .iter()
        .map(|&t| if dip(t) { 0.99 } else { 1.0 })
        .collect();
    let masked: Vec<bool> = time.iter().map(|&t| dip(t)).collect();
    let lc = curve(&time, &flux);
    let flat = Detrend::default().flatten(&lc, Some(&masked)).unwrap();
    assert!(flat.trend().iter().all(|&t| t == Some(1.0)));
    assert_eq!(flat.light_curve().flux(), &flux[..]);
}

#[test]
fn a_window_for_a_transit() {
    let detrend = Detrend::default().with_window_for_transit(days(0.2));
    assert_eq!(detrend.window, days(0.2) * WINDOW_PER_DURATION);
    assert_eq!(WINDOW_PER_DURATION, 2.5);
    // Checked when used, as any window.
    let lc = curve(&[0.0, 1.0], &[1.0, 1.0]);
    assert_eq!(
        Detrend::default()
            .with_window_for_transit(days(-1.0))
            .flatten(&lc, None),
        Err(DetrendError::NotPositive { name: "window" })
    );
}

#[test]
fn the_transit_mask_by_definition() {
    // Times either side of t0, at awkward steps: masked exactly when some
    // whole n puts t0 + n period within (not at) half a duration.
    let mut random = Random(0x6d61_736b);
    for _ in 0..200 {
        let period = 0.3 + (random.below(1000) as f64) / 97.0;
        let duration = period * (1 + random.below(900)) as f64 / 1000.0;
        let t0 = (random.below(4000) as f64) / 37.0 - 50.0;
        let time: Vec<f64> = (0..300)
            .map(|i| t0 - 3.1 * period + i as f64 * period / 41.3)
            .collect();
        let lc = curve(&time, &vec![1.0; time.len()]);
        let mask = transit_mask(
            &lc,
            days(period),
            Epoch::from_btjd(t0).unwrap(),
            days(duration),
        )
        .unwrap();
        for (&t, &m) in time.iter().zip(&mask) {
            let n = ((t - t0) / period).round();
            let near = (-1..=1)
                .map(|k| (t - (t0 + (n + f64::from(k)) * period)).abs())
                .fold(f64::INFINITY, f64::min);
            // Away from the edge, where rounding could decide either way.
            if (near - duration / 2.0).abs() > 1e-9 {
                assert_eq!(m, near < duration / 2.0, "t {t}, period {period}");
            }
        }
    }
}

#[test]
fn the_transit_mask_edges_and_checks() {
    // Every 1/16 day (exact in binary, so is every step below); transits
    // every 8 steps from step 5, 4 steps long: the points 2 steps away are
    // not masked, those just inside are.
    let time: Vec<f64> = (0..30).map(|i| f64::from(i) / 16.0).collect();
    let lc = curve(&time, &[1.0; 30]);
    let steps = |n: f64| days(n / 16.0);
    let t0 = Epoch::from_btjd(5.0 / 16.0).unwrap();
    let mask = transit_mask(&lc, steps(8.0), t0, steps(4.0)).unwrap();
    let masked: Vec<usize> = (0..30).filter(|&i| mask[i]).collect();
    assert_eq!(masked, [4, 5, 6, 12, 13, 14, 20, 21, 22, 28, 29]);
    let mask = transit_mask(&lc, steps(8.0), t0, steps(4.0 + 1e-9)).unwrap();
    assert_eq!(mask.iter().filter(|&&m| m).count(), 11 + 7);

    let t0 = Epoch::from_btjd(0.0).unwrap();
    for (period, duration, name) in [
        (0.0, 1.0, "period"),
        (-1.0, 1.0, "period"),
        (1.0, 0.0, "duration"),
        (1.0, -1.0, "duration"),
    ] {
        assert_eq!(
            transit_mask(&lc, days(period), t0, days(duration)),
            Err(DetrendError::NotPositive { name })
        );
    }
}
