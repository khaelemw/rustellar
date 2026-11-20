//! `Method::RobustSpline` against the same clipping loop run exactly (60
//! digits: fits, residuals, medians and comparisons) and with scipy in
//! doubles, on light curves with flares, dips and single outliers; then by
//! definition on small cases.

use rustellar_core::{LightCurve, TimeDelta};
use rustellar_detrend::{BreakTolerance, Detrend, DetrendError, Method};

const CASES: &str = include_str!("reference/robust_spline_cases.csv");
const POINTS: &str = include_str!("reference/robust_spline_points.csv");

fn rows(text: &'static str) -> impl Iterator<Item = Vec<&'static str>> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| l.split(',').collect())
}

fn optional(s: &str) -> Option<f64> {
    (!s.is_empty()).then(|| s.parse().unwrap())
}

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

#[derive(Default)]
struct Case {
    window: f64,
    tolerance: f64,
    sigma: f64,
    time: Vec<f64>,
    flux: Vec<f64>,
    masked: Vec<bool>,
    scipy: Vec<Option<f64>>,
    exact: Vec<Option<f64>>,
}

fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = rows(CASES)
        .map(|r| Case {
            window: r[1].parse().unwrap(),
            tolerance: r[2].parse().unwrap(),
            sigma: r[3].parse().unwrap(),
            ..Case::default()
        })
        .collect();
    for r in rows(POINTS) {
        let c = &mut cases[r[0].parse::<usize>().unwrap()];
        c.time.push(r[1].parse().unwrap());
        c.flux.push(r[2].parse().unwrap());
        c.masked.push(r[3] == "1");
        c.scipy.push(optional(r[5]));
        c.exact.push(optional(r[6]));
    }
    cases
}

fn detrend(window: f64, tolerance: BreakTolerance, sigma: f64) -> Detrend {
    Detrend::default()
        .with_method(Method::RobustSpline { sigma })
        .with_window(days(window))
        .with_break_tolerance(tolerance)
}

fn trend(c: &Case) -> Vec<Option<f64>> {
    let n = c.time.len();
    let lc = LightCurve::from_days(c.time.clone(), c.flux.clone(), vec![1.0; n]).unwrap();
    detrend(c.window, BreakTolerance::Fixed(days(c.tolerance)), c.sigma)
        .trend(&lc, Some(&c.masked))
        .unwrap()
}

// The largest error from the exact loop's trend, in units of 2^-52 times
// the largest flux.
fn error(got: &[Option<f64>], exact: &[Option<f64>], flux: &[f64]) -> f64 {
    let ulp = flux.iter().fold(0.0f64, |m, x| m.max(x.abs())) * f64::EPSILON;
    got.iter()
        .zip(exact)
        .map(|(g, e)| match (g, e) {
            (Some(g), Some(e)) => (g - e).abs() / ulp,
            (None, None) => 0.0,
            _ => f64::INFINITY,
        })
        .fold(0.0, f64::max)
}

#[test]
fn against_the_exact_loop_and_scipy() {
    let (mut ours, mut scipy) = (vec![], vec![]);
    for c in &cases() {
        ours.push(error(&trend(c), &c.exact, &c.flux));
        scipy.push(error(&c.scipy, &c.exact, &c.flux));
    }
    let count = |v: &[f64], limit: f64| v.iter().filter(|&&e| e < limit).count();
    eprintln!(
        "within 4 ulps of the exact loop: ours {} of {}, scipy {}",
        count(&ours, 4.0),
        ours.len(),
        count(&scipy, 4.0),
    );
    // The same points clipped as the exact loop, and the trend within a
    // few ulps of its, in every case; never further than scipy's loop.
    for (o, s) in ours.iter().zip(&scipy) {
        assert!(*o < 4.0 && o <= s, "{ours:?} {scipy:?}");
    }
}

// A flare and a dip on a slow wave: clipped out, the spline follows the
// wave; the plain spline is pulled by both.
#[test]
fn flares_and_dips_drop_out() {
    let time: Vec<f64> = (0..2160).map(|i| f64::from(i) / 720.0).collect();
    let star = |t: f64| 1.0 + 3e-3 * (t * 2.1).sin();
    let flux: Vec<f64> = time
        .iter()
        .map(|&t| {
            let flare = if t >= 1.0 {
                8e-3 * (-(t - 1.0) / 0.02).exp()
            } else {
                0.0
            };
            let dip = if (t - 2.2).abs() < 0.05 { 4e-3 } else { 0.0 };
            star(t) + flare - dip
        })
        .collect();
    let lc = LightCurve::from_days(time.clone(), flux, vec![1e-3; 2160]).unwrap();
    let robust = detrend(0.5, BreakTolerance::Never, 3.0)
        .trend(&lc, None)
        .unwrap();
    let plain = Detrend::default()
        .with_method(Method::Spline)
        .with_window(days(0.5))
        .trend(&lc, None)
        .unwrap();
    let worst = |trend: &[Option<f64>]| {
        trend
            .iter()
            .zip(&time)
            .map(|(g, &t)| (g.unwrap() - star(t)).abs())
            .fold(0.0, f64::max)
    };
    // The flare's faint tail stays in; the rest is gone.
    let (robust, plain) = (worst(&robust), worst(&plain));
    assert!(robust < 2e-5 && plain > 10.0 * robust, "{robust} {plain}");
}

// Four points and a cubic through them: no residual, no scatter, so the
// loop stops at once with the plain spline.
#[test]
fn no_scatter_stops_the_loop() {
    let lc = LightCurve::from_days(
        vec![0.0, 0.3, 0.5, 1.0],
        vec![1.0, 3.0, -2.0, 5.0],
        vec![1.0; 4],
    )
    .unwrap();
    let robust = detrend(10.0, BreakTolerance::Never, 3.0)
        .trend(&lc, None)
        .unwrap();
    let plain = Detrend::default()
        .with_method(Method::Spline)
        .with_window(days(10.0))
        .with_break_tolerance(BreakTolerance::Never)
        .trend(&lc, None)
        .unwrap();
    assert_eq!(robust, plain);
}

#[test]
fn sigma_is_checked() {
    let lc = LightCurve::from_days(vec![0.0, 1.0], vec![1.0; 2], vec![1.0; 2]).unwrap();
    for sigma in [0.0, -1.0, f64::NAN, 1e31] {
        assert_eq!(
            detrend(1.0, BreakTolerance::Never, sigma).trend(&lc, None),
            Err(DetrendError::TuningOutOfRange)
        );
    }
}

// Pure white noise: clipping both sides leaves the trend unbiased (wotan's
// rspline, clipping only below, sits about 0.09 standard deviations high).
#[test]
fn no_bias_on_noise() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut unit = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut sum, mut n) = (0.0, 0.0);
    for _ in 0..20 {
        let time: Vec<f64> = (0..7200).map(|i| f64::from(i) / 720.0).collect();
        let flux: Vec<f64> = (0..7200)
            .map(|_| {
                let (u, v) = (1.0 - unit(), unit());
                1.0 + 1e-3 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
            })
            .collect();
        let lc = LightCurve::from_days(time, flux, vec![1e-3; 7200]).unwrap();
        for t in detrend(0.5, BreakTolerance::Never, 3.0)
            .trend(&lc, None)
            .unwrap()
        {
            sum += t.unwrap() - 1.0;
            n += 1.0;
        }
    }
    let bias = sum / n / 1e-3;
    assert!(bias.abs() < 0.01, "{bias}");
}
