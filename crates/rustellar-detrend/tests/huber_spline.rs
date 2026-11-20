//! `Method::HuberSpline` against its fixed point solved at 50 digits and
//! against statsmodels' `RLM(HuberT)` on the same B-splines (at its default
//! tolerance and at 1e-14), on light curves with flares, dips and single
//! outliers; then by definition on small cases.

use rustellar_core::{LightCurve, TimeDelta};
use rustellar_detrend::{BreakTolerance, Detrend, DetrendError, Method};

const CASES: &str = include_str!("reference/huber_spline_cases.csv");
const POINTS: &str = include_str!("reference/huber_spline_points.csv");

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
    c: f64,
    time: Vec<f64>,
    flux: Vec<f64>,
    masked: Vec<bool>,
    statsmodels: Vec<Option<f64>>,
    tight: Vec<Option<f64>>,
    exact: Vec<Option<f64>>,
}

fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = rows(CASES)
        .map(|r| Case {
            window: r[1].parse().unwrap(),
            tolerance: r[2].parse().unwrap(),
            c: r[3].parse().unwrap(),
            ..Case::default()
        })
        .collect();
    for r in rows(POINTS) {
        let c = &mut cases[r[0].parse::<usize>().unwrap()];
        c.time.push(r[1].parse().unwrap());
        c.flux.push(r[2].parse().unwrap());
        c.masked.push(r[3] == "1");
        c.statsmodels.push(optional(r[4]));
        c.tight.push(optional(r[5]));
        c.exact.push(optional(r[6]));
    }
    cases
}

fn detrend(window: f64, tolerance: BreakTolerance, c: f64) -> Detrend {
    Detrend::default()
        .with_method(Method::HuberSpline { c })
        .with_window(days(window))
        .with_break_tolerance(tolerance)
}

fn trend(c: &Case) -> Vec<Option<f64>> {
    let n = c.time.len();
    let lc = LightCurve::from_btjd(c.time.clone(), c.flux.clone(), vec![1.0; n]).unwrap();
    detrend(c.window, BreakTolerance::Fixed(days(c.tolerance)), c.c)
        .trend(&lc, Some(&c.masked))
        .unwrap()
}

// The largest error from the fixed point, in units of 2^-52 times the
// largest flux.
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
fn against_the_fixed_point_and_statsmodels() {
    let (mut ours, mut loose, mut tight) = (vec![], vec![], vec![]);
    for c in &cases() {
        ours.push(error(&trend(c), &c.exact, &c.flux));
        loose.push(error(&c.statsmodels, &c.exact, &c.flux));
        tight.push(error(&c.tight, &c.exact, &c.flux));
    }
    eprintln!("ours {ours:.1?}");
    eprintln!("statsmodels {loose:.1?}");
    eprintln!("statsmodels at 1e-14 {tight:.1?}");
    for ((o, l), t) in ours.iter().zip(&loose).zip(&tight) {
        // A few ulps; where the knots leave the fit badly conditioned, a
        // hundredth of statsmodels' error at its default tolerance or less
        // (at 1e-14 it can come nearer there: its design matrix is the
        // B-splines correctly rounded, this crate's computed).
        assert!(o <= l, "{ours:?}");
        assert!(*o < 4.0 || *o * 100.0 < *l, "{ours:?}");
        assert!(*o < 4.0 || *o < 2.0 * t, "{ours:?}");
    }
}

// A flare and a dip on a slow wave: Huber's weights pull the spline far
// less than plain least squares.
#[test]
fn flares_and_dips_pull_less() {
    let time: Vec<f64> = (0..2160).map(|i| f64::from(i) / 720.0).collect();
    let star = |t: f64| 1.0 + 3e-3 * (t * 2.1).sin();
    let mut state = 7u64;
    let flux: Vec<f64> = time
        .iter()
        .map(|&t| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = 2e-4 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            let flare = if t >= 1.0 {
                8e-3 * (-(t - 1.0) / 0.02).exp()
            } else {
                0.0
            };
            let dip = if (t - 2.2).abs() < 0.05 { 4e-3 } else { 0.0 };
            star(t) + flare - dip + noise
        })
        .collect();
    let lc = LightCurve::from_btjd(time.clone(), flux, vec![1e-3; 2160]).unwrap();
    let worst = |method| {
        Detrend::default()
            .with_method(method)
            .with_window(days(0.5))
            .with_break_tolerance(BreakTolerance::Never)
            .trend(&lc, None)
            .unwrap()
            .iter()
            .zip(&time)
            .map(|(g, &t)| (g.unwrap() - star(t)).abs())
            .fold(0.0, f64::max)
    };
    let (huber, plain) = (
        worst(Method::HuberSpline { c: 1.345 }),
        worst(Method::Spline),
    );
    assert!(plain > 3.0 * huber, "{huber} {plain}");
}

// Four points and a cubic through them: residuals of rounding only, so
// the plain spline to rounding.
#[test]
fn no_scatter_stops_the_loop() {
    let lc = LightCurve::from_btjd(
        vec![0.0, 0.3, 0.5, 1.0],
        vec![1.0, 3.0, -2.0, 5.0],
        vec![1.0; 4],
    )
    .unwrap();
    let huber = detrend(10.0, BreakTolerance::Never, 1.345)
        .trend(&lc, None)
        .unwrap();
    let plain = Detrend::default()
        .with_method(Method::Spline)
        .with_window(days(10.0))
        .with_break_tolerance(BreakTolerance::Never)
        .trend(&lc, None)
        .unwrap();
    for (h, p) in huber.iter().zip(&plain) {
        assert!(
            (h.unwrap() - p.unwrap()).abs() < 1e-12,
            "{huber:?} {plain:?}"
        );
    }
}

#[test]
fn c_is_checked() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0; 2], vec![1.0; 2]).unwrap();
    for c in [0.0, -1.0, f64::NAN, 1e31] {
        assert_eq!(
            detrend(1.0, BreakTolerance::Never, c).trend(&lc, None),
            Err(DetrendError::TuningOutOfRange)
        );
    }
}

// Zero flux: no residual, no scale, the plain spline (zero). A tiny c:
// no point within c s, so reweighting takes over; the trend still follows
// the star.
#[test]
fn no_scale_and_a_tiny_c() {
    let time: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.01).collect();
    let zero = LightCurve::from_btjd(time.clone(), vec![0.0; 200], vec![1.0; 200]).unwrap();
    let got = detrend(0.5, BreakTolerance::Never, 1.345)
        .trend(&zero, None)
        .unwrap();
    assert!(got.iter().all(|g| *g == Some(0.0)), "{got:?}");
    let star = |t: f64| 1.0 + 1e-3 * t;
    let flux: Vec<f64> = time
        .iter()
        .enumerate()
        .map(|(i, &t)| star(t) + if i % 2 == 0 { 1e-4 } else { -1e-4 })
        .collect();
    let lc = LightCurve::from_btjd(time.clone(), flux, vec![1.0; 200]).unwrap();
    let got = detrend(0.5, BreakTolerance::Never, 1e-30)
        .trend(&lc, None)
        .unwrap();
    for (g, &t) in got.iter().zip(&time) {
        assert!((g.unwrap() - star(t)).abs() < 2e-4, "{t}");
    }
}
