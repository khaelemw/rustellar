// The least-squares solve: the box's closed form, an independent solve on
// random data, noiseless recovery, the errors against the scatter of many
// noisy fits, and the phase.

use crate::VetError;
use crate::fit::{phase, solve};

// A small deterministic generator (SplitMix64), so the tests need no crate.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    // A standard normal, by Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u, v) = (1.0 - self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[test]
fn the_phase() {
    assert_eq!(phase(0.0, 3.0), 0.0);
    assert_eq!(phase(3.25, 3.0), 0.25);
    assert_eq!(phase(-3.25, 3.0), -0.25);
    assert_eq!(phase(2.75, 3.0), -0.25);
    assert_eq!(phase(1.5, 3.0), -1.5);
    assert_eq!(phase(-1.5, 3.0), -1.5);
    // Within half a period, unrounded: even a tiny step beside a huge
    // period.
    assert_eq!(phase(1e-3, 1e305), 1e-3);
    assert_eq!(phase(0.1, 3.0), 0.1);
}

// For a box the fit is the weighted means in and out: depth 1 - in / out,
// its error from 1 / W_in + 1 / W_out on the difference.
#[test]
fn the_box_is_the_means() {
    let mut r = Random(7);
    let n = 500;
    let shapes: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 9 == 0))).collect();
    let errs: Vec<f64> = (0..n).map(|_| 1e-3 * (1.0 + r.unit())).collect();
    let flux: Vec<f64> = shapes
        .iter()
        .zip(&errs)
        .map(|(s, e)| 1.0 - 2e-3 * s + e * r.normal())
        .collect();
    let fit = solve(&flux, &errs, &shapes).unwrap();
    let mean = |inside: bool| {
        let (mut w, mut wy) = (0.0, 0.0);
        for ((s, e), y) in shapes.iter().zip(&errs).zip(&flux) {
            if (*s > 0.0) == inside {
                w += 1.0 / (e * e);
                wy += y / (e * e);
            }
        }
        (wy / w, w)
    };
    let ((y_in, w_in), (y_out, w_out)) = (mean(true), mean(false));
    assert!((fit.baseline - y_out).abs() < 1e-15);
    assert!((fit.depth - (1.0 - y_in / y_out)).abs() < 1e-14);
    assert!((fit.baseline_err - w_out.powf(-0.5)).abs() < 1e-15);
    // The means in and out are independent, so for 1 - in / out the delta
    // method gives var(in) / out^2 + in^2 var(out) / out^4.
    let err = (1.0 / w_in / y_out.powi(2) + y_in.powi(2) / w_out / y_out.powi(4)).sqrt();
    assert!(
        (fit.depth_err / err - 1.0).abs() < 1e-12,
        "{} {err}",
        fit.depth_err
    );
    assert_eq!(fit.points_in_transit, 56);
}

// The same fit by the normal equations solved directly, on random shapes,
// errors and fluxes: the two agree to rounding.
#[test]
fn an_independent_solve() {
    let mut r = Random(11);
    for case in 0..200 {
        let n = 20 + (r.next() % 300) as usize;
        let shapes: Vec<f64> = (0..n)
            .map(|_| if r.unit() < 0.3 { r.unit() } else { 0.0 })
            .collect();
        let errs: Vec<f64> = (0..n).map(|_| 10f64.powf(-4.0 + 2.0 * r.unit())).collect();
        let flux: Vec<f64> = (0..n).map(|_| 100.0 * (1.0 + 0.01 * r.normal())).collect();
        let Ok(fit) = solve(&flux, &errs, &shapes) else {
            continue;
        };
        let (mut s0, mut s1, mut s2, mut t0, mut t1, mut yy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for ((s, e), y) in shapes.iter().zip(&errs).zip(&flux) {
            let w = 1.0 / (e * e);
            s0 += w;
            s1 += w * s;
            s2 += w * s * s;
            t0 += w * y;
            t1 += w * s * y;
            yy += w * y * y;
        }
        let det = s0 * s2 - s1 * s1;
        let alpha = (s2 * t0 - s1 * t1) / det;
        let beta = (s0 * t1 - s1 * t0) / det;
        let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * b.abs().max(1e-300);
        assert!(close(fit.baseline, alpha, 1e-10), "{case}");
        assert!(close(fit.depth, -beta / alpha, 1e-7), "{case}");
        assert!(close(fit.baseline_err, (s2 / det).sqrt(), 1e-8), "{case}");
        let chi2 = yy - alpha * t0 - beta * t1;
        assert!((fit.chi2 - chi2).abs() <= 1e-6 * yy, "{case}");
    }
}

#[test]
fn noiseless_data_give_the_depth_back() {
    let shapes: Vec<f64> = (0..200)
        .map(|i| (1.0 - (f64::from(i) - 100.0).powi(2) / 400.0).max(0.0))
        .collect();
    let flux: Vec<f64> = shapes.iter().map(|s| 42.0 * (1.0 - 3e-3 * s)).collect();
    let fit = solve(&flux, &[0.5; 200], &shapes).unwrap();
    assert!((fit.depth - 3e-3).abs() < 1e-15, "{}", fit.depth);
    assert!((fit.baseline - 42.0).abs() < 1e-13);
    assert!(fit.chi2 < 1e-20);
    assert_eq!(fit.points_in_transit, 39);
}

// Over many noise draws, the depths scatter as the errors say.
#[test]
fn the_errors_match_the_scatter() {
    let mut r = Random(3);
    let shapes: Vec<f64> = (0..400)
        .map(|i| (1.0 - (f64::from(i % 100) - 50.0).powi(2) / 64.0).max(0.0))
        .collect();
    let errs: Vec<f64> = (0..400).map(|i| 1e-3 * (1.0 + f64::from(i % 3))).collect();
    let (mut depths, mut stated) = (vec![], 0.0);
    for _ in 0..4000 {
        let flux: Vec<f64> = shapes
            .iter()
            .zip(&errs)
            .map(|(s, e)| 1.0 - 1e-3 * s + e * r.normal())
            .collect();
        let fit = solve(&flux, &errs, &shapes).unwrap();
        depths.push(fit.depth);
        stated += fit.depth_err / 4000.0;
        // Delta chi-squared is the depth's significance squared, near enough.
        assert!((fit.delta_chi2 / fit.snr().powi(2) - 1.0).abs() < 0.01);
    }
    let mean = depths.iter().sum::<f64>() / 4000.0;
    let spread = (depths.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / 3999.0).sqrt();
    assert!((mean - 1e-3).abs() < 3.0 * stated / 4000f64.sqrt());
    // 4,000 draws pin the spread to about 1.1%.
    assert!((spread / stated - 1.0).abs() < 0.04, "{spread} {stated}");
}

#[test]
fn what_cannot_be_fitted() {
    assert_eq!(
        solve(&[1.0, 1.0], &[1.0, 1.0], &[0.0, 0.0]).unwrap_err(),
        VetError::NoPointsInTransit
    );
    assert_eq!(
        solve(&[1.0, 1.0], &[1.0, 1.0], &[1.0, 0.5]).unwrap_err(),
        VetError::NoPointsOutOfTransit
    );
    assert_eq!(
        solve(&[-1.0, -1.0], &[1.0, 1.0], &[1.0, 0.0]).unwrap_err(),
        VetError::BaselineNotPositive
    );
    assert_eq!(
        solve(&[0.0, 1.0], &[1.0, 1.0], &[1.0, 0.0])
            .unwrap()
            .baseline,
        1.0
    );
}
