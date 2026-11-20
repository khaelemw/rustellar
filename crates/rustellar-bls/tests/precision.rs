// Precision on long light curves (13 and 56 TESS sectors, up to a million
// points; flux near 1 and in counts near 1e5): at periods from 0.75 to 400
// days, the box astropy's fast search reports was found by replaying its
// arithmetic bit for bit, and that box's statistics computed in exact
// rationals. Our binned search reports the same box; its depth, snr and
// delta chi-squared are checked against the exact values within a bound
// derived from how they are computed, and astropy's own error is recorded.

#[expect(dead_code, reason = "the precision tests use only some of the recipes")]
#[path = "../src/tests/recipe.rs"]
mod recipe;

use recipe::{Survey, check_samples};
use rustellar_bls::{Bls, BoxFit, Durations, Objective, Periods, SearchOptions};
use rustellar_core::LightCurve;

// ==========================
// Helper Functions
// ==========================

const CURVES: &str = include_str!("reference/precision_curves.csv");
const BOXES: &str = include_str!("reference/precision_boxes.csv");
// Bins of 1/1024 d: every period searched is a whole number of them, so
// astropy's bins and ours are the same bins.
const DURATIONS: [f64; 3] = [0.0625, 0.125, 0.25];
const OVERSAMPLE: usize = 64;
const BIN: f64 = 1.0 / 1024.0;
const U: f64 = f64::EPSILON / 2.0;

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

// 2-minute TESS sectors back to back, each two orbits of 9,000 points a day
// apart, the flux moved by `offset`.
fn curve(v: &[&str]) -> LightCurve {
    let f = |i: usize| v[i].parse::<f64>().unwrap();
    let sectors: u32 = v[3].parse().unwrap();
    let cadence = 2.0 / 1440.0;
    let segments = (0..sectors)
        .flat_map(|k| {
            let start = 1325.0 + 27.5 * f64::from(k);
            [(start, 9000), (start + 9000.0 * cadence + 1.0, 9000)]
        })
        .collect();
    let lc = Survey {
        seed: v[2].parse().unwrap(),
        segments,
        cadence,
        noise: f(4),
        keep: 1.0,
        period: f(6),
        tc: f(7),
        duration: f(8),
        depth: f(9),
    }
    .light_curve();
    let offset = f(5);
    let flux = lc.flux().iter().map(|y| y + offset).collect();
    let time = lc.time().iter().map(|t| t.as_days()).collect();
    let lc = LightCurve::from_days(time, flux, lc.flux_err().to_vec()).unwrap();
    assert_eq!(lc.len(), v[10].parse::<usize>().unwrap());
    check_samples(&lc, v[11]);
    lc
}

// The exact statistics of a box: depth, snr, delta chi-squared, flux in and
// out of transit.
#[derive(Debug, Clone, Copy)]
struct Exact([f64; 5]);

fn values(fit: &BoxFit) -> [f64; 5] {
    [
        fit.depth,
        fit.snr,
        fit.delta_chi2,
        fit.flux_in,
        fit.flux_out,
    ]
}

// How far each of our values may be from the exact ones, for the points
// `inside` with each plain sum over at most `plain` terms.
//
// With u = 2^-53 and g = plain u / (1 - plain u):
// - the weighted mean m rounds, but y - m is exact (Sterbenz: every flux is
//   within a factor of 2 of m, checked), so each centred term c = w (y - m)
//   rounds once, by u |c|;
// - sums of up to `plain` terms (inside a bin) err by g times their
//   magnitudes; the running totals over them are compensated, leaving 3u of
//   the result and a second-order (n u)^2 of the magnitudes, n the number of
//   totals;
// - so S_in errs by (u + g) A_in + 3u |S_in| + (n u)^2 A, with A_in the sum of
//   |c| in transit and A over all, and W_in by (g + 3u) W_in + (n u)^2 W; the
//   same out of transit;
// - a level l = S / W then errs by
//   ((u + g) A_in + (n u)^2 (A + |l| W)) / W_in + (g + 7u) |l|, the depth
//   l_out - l_in by both and u |depth|; snr adds g + 4u relative,
//   delta chi-squared twice the depth's relative error plus 2g + 6u, and
//   flux m + l adds u of itself.
fn bounds(lc: &LightCurve, inside: &[bool], plain: usize, totals: usize) -> [f64; 5] {
    let w: Vec<f64> = lc.flux_err().iter().map(|e| 1.0 / (e * e)).collect();
    let weight: f64 = w.iter().sum();
    let mean = w.iter().zip(lc.flux()).map(|(w, y)| w * y).sum::<f64>() / weight;
    assert!(
        lc.flux()
            .iter()
            .all(|&y| 0.5 * mean <= y && y <= 2.0 * mean)
    );
    let (mut w_in, mut w_out, mut s_in, mut s_out, mut a_in, mut a_out) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for ((&w, &y), &inside) in w.iter().zip(lc.flux()).zip(inside) {
        let c = w * (y - mean);
        if inside {
            (w_in, s_in, a_in) = (w_in + w, s_in + c, a_in + c.abs());
        } else {
            (w_out, s_out, a_out) = (w_out + w, s_out + c, a_out + c.abs());
        }
    }
    let plain = plain as f64 * U;
    let g = plain / (1.0 - plain);
    let second = (totals as f64 * U).powi(2);
    let (l_in, l_out) = (s_in / w_in, s_out / w_out);
    let level = |w: f64, l: f64, a: f64| {
        ((U + g) * a + second * ((a_in + a_out) + l.abs() * (w_in + w_out))) / w
            + (g + 7.0 * U) * l.abs()
    };
    let (e_in, e_out) = (level(w_in, l_in, a_in), level(w_out, l_out, a_out));
    let depth = l_out - l_in;
    let e_depth = e_in + e_out + U * depth.abs();
    let r_depth = e_depth / depth.abs();
    [
        e_depth,
        (r_depth + g + 4.0 * U) * (depth.abs() / (1.0 / w_in + 1.0 / w_out).sqrt()),
        (2.0 * r_depth + 2.0 * g + 6.0 * U) * (w_in * (w_out / weight) * depth * depth),
        e_in + U * (mean + l_in).abs(),
        e_out + U * (mean + l_out).abs(),
    ]
}

// The most points in one of the `m` bins at `period`, our way: phases from
// the first time.
fn most_in_a_bin(lc: &LightCurve, period: f64, m: usize) -> usize {
    let t0 = lc.time()[0].as_days();
    let mut counts = vec![0_usize; m];
    for t in lc.time() {
        let phase = (t.as_days() - t0).rem_euclid(period) / period;
        counts[((phase * m as f64) as usize).min(m - 1)] += 1;
    }
    counts.into_iter().max().unwrap()
}

// The largest relative error of our depth, snr and delta chi-squared.
fn largest(ours: &[f64; 5], exact: Exact) -> f64 {
    (0..3)
        .map(|j| (ours[j] / exact.0[j] - 1.0).abs())
        .fold(0.0, f64::max)
}

// ==========================
// The study
// ==========================

#[test]
fn exact_statistics_on_long_light_curves() {
    let curves: Vec<LightCurve> = rows(CURVES).map(|v| curve(&v)).collect();
    let durations = Durations::from_days(&DURATIONS).unwrap();
    let mut worst = [0.0_f64; 2];
    for v in rows(BOXES) {
        let f = |i: usize| v[i].parse::<f64>().unwrap();
        let lc = &curves[v[0].parse::<usize>().unwrap()];
        let period = f(1);
        let (start, bins, m, n_in): (usize, usize, usize, usize) = (
            v[4].parse().unwrap(),
            v[5].parse().unwrap(),
            v[6].parse().unwrap(),
            v[7].parse().unwrap(),
        );
        let exact = Exact([f(8), f(9), f(10), f(11), f(12)]);
        let result = Bls::new(lc)
            .unwrap()
            .search(
                &Periods::from_days(&[period]).unwrap(),
                &durations,
                &SearchOptions::default()
                    .with_objective(Objective::Snr)
                    .with_oversample(OVERSAMPLE.try_into().unwrap())
                    .with_refine_peaks(0),
            )
            .unwrap();

        // The same box as astropy's: the same bins, so the same points.
        let binned = result.boxes()[0].unwrap();
        assert_eq!(binned.fit.n_in, n_in, "period {period}");
        assert_eq!(binned.transit.duration().as_days(), bins as f64 * BIN);
        let first = lc.time()[0].as_days();
        let centre = (start as f64 + 0.5 * bins as f64) * BIN;
        let offset = (binned.transit.transit_time().as_days() - first - centre).rem_euclid(period);
        assert!(
            offset.min(period - offset) <= 0.5 * BIN,
            "period {period}: {offset}"
        );
        let inside: Vec<bool> = lc
            .time()
            .iter()
            .map(|&t| binned.transit.contains(t))
            .collect();
        assert_eq!(inside.iter().filter(|&&x| x).count(), n_in);

        // Ours within the bound, binned and for the one box.
        let plain = most_in_a_bin(lc, period, m);
        let binned_bound = bounds(lc, &inside, plain, m + bins);
        let one = binned.transit.fit(lc).unwrap();
        assert_eq!(one.n_in, n_in);
        let one_bound = bounds(lc, &inside, 1, lc.len());
        for (fit, bound, path) in [
            (&binned.fit, binned_bound, "binned"),
            (&one, one_bound, "one box"),
        ] {
            for ((ours, exact), bound) in values(fit).into_iter().zip(exact.0).zip(bound) {
                assert!(
                    (ours - exact).abs() <= bound,
                    "{path}, period {period}: {ours} vs {exact}, bound {bound:e}"
                );
            }
        }

        let ours_error = largest(&values(&binned.fit), exact);
        let astropy_error = ((f(2) / exact.0[1] - 1.0).abs()).max((f(3) / exact.0[0] - 1.0).abs());
        println!(
            "period {period:>12}: in transit {n_in:>6}, most in a bin {plain:>5}; relative error, ours {ours_error:.1e} (bound {:.1e}), astropy {astropy_error:.1e}",
            (0..2)
                .map(|j| binned_bound[j] / exact.0[j].abs())
                .fold(0.0, f64::max)
        );
        // Few points in transit against many out: astropy's in-transit sums
        // are small differences of large plain running totals, and its error
        // is above even our worst-case bound (it would pass if astropy kept
        // compensated totals; then the docs change).
        if period >= 150.0 {
            let ours_bound = (0..2)
                .map(|j| binned_bound[j] / exact.0[j].abs())
                .fold(0.0, f64::max);
            assert!(astropy_error > ours_bound, "period {period}");
        }
        worst = [worst[0].max(ours_error), worst[1].max(astropy_error)];
    }
    println!(
        "largest relative error: ours {:.1e}, astropy {:.1e}",
        worst[0], worst[1]
    );
}
