//! `Method::Spline` against scipy's `LSQUnivariateSpline` on the same knots
//! and an exact least-squares spline (60 digits, rounded once), on light
//! curves of normalised flux, counts, values about zero and values about
//! 1e200, with gaps, masks and edge cutoffs.

use rustellar_core::{LightCurve, TimeDelta};
use rustellar_detrend::{BreakTolerance, Detrend, Method};

const CASES: &str = include_str!("reference/spline_cases.csv");
const POINTS: &str = include_str!("reference/spline_points.csv");

fn rows(text: &'static str) -> impl Iterator<Item = Vec<&'static str>> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| l.split(',').collect())
}

fn optional(s: &str) -> Option<f64> {
    (!s.is_empty()).then(|| s.parse().unwrap())
}

struct Case {
    window: f64,
    tolerance: f64,
    cutoff: f64,
    time: Vec<f64>,
    flux: Vec<f64>,
    masked: Vec<bool>,
    scipy: Vec<Option<f64>>,
    exact: Vec<Option<f64>>,
}

fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = rows(CASES)
        .map(|r| Case {
            window: r[2].parse().unwrap(),
            tolerance: r[3].parse().unwrap(),
            cutoff: r[4].parse().unwrap(),
            time: vec![],
            flux: vec![],
            masked: vec![],
            scipy: vec![],
            exact: vec![],
        })
        .collect();
    for r in rows(POINTS) {
        let c = &mut cases[r[0].parse::<usize>().unwrap()];
        c.time.push(r[1].parse().unwrap());
        c.flux.push(r[2].parse().unwrap());
        c.masked.push(r[3] == "1");
        c.scipy.push(optional(r[4]));
        c.exact.push(optional(r[5]));
    }
    cases
}

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn trend(c: &Case) -> Vec<Option<f64>> {
    let n = c.time.len();
    let lc = LightCurve::from_btjd(c.time.clone(), c.flux.clone(), vec![1.0; n]).unwrap();
    let tolerance = if c.tolerance.is_infinite() {
        BreakTolerance::Never
    } else {
        BreakTolerance::Fixed(days(c.tolerance))
    };
    Detrend::default()
        .with_method(Method::Spline)
        .with_window(days(c.window))
        .with_break_tolerance(tolerance)
        .with_edge_cutoff(days(c.cutoff))
        .trend(&lc, Some(&c.masked))
        .unwrap()
}

// The error of `got` from the exact spline, in units of the largest flux's
// magnitude in the case times 2^-52 (one ulp at the data's scale).
fn error(got: &[Option<f64>], exact: &[Option<f64>], flux: &[f64]) -> f64 {
    let scale = flux.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let ulp = scale * f64::EPSILON;
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
fn against_scipy_and_the_exact_spline() {
    let (mut ours, mut scipy) = (vec![], vec![]);
    for c in &cases() {
        let got = trend(c);
        ours.push(error(&got, &c.exact, &c.flux));
        scipy.push(error(&c.scipy, &c.exact, &c.flux));
    }
    let worst = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    let median = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    eprintln!(
        "ours: median {:.2}, worst {:.2}; scipy: median {:.2}, worst {:.2} (ulps of the scale)",
        median(&ours),
        worst(&ours),
        median(&scipy),
        worst(&scipy)
    );
    // About an ulp of the data's scale for most; never worse than scipy,
    // and a tenth of its error or less where the knots leave the fit badly
    // conditioned.
    assert!(median(&ours) < 2.0, "{ours:?}");
    for (o, s) in ours.iter().zip(&scipy) {
        assert!(*o <= (s / 10.0).max(4.0) && o <= s, "{ours:?} {scipy:?}");
    }
}

fn spline(time: &[f64], flux: &[f64], window: f64, mask: Option<&[bool]>) -> Vec<Option<f64>> {
    let lc = LightCurve::from_btjd(time.to_vec(), flux.to_vec(), vec![1.0; time.len()]).unwrap();
    Detrend::default()
        .with_method(Method::Spline)
        .with_window(days(window))
        .with_break_tolerance(BreakTolerance::Never)
        .trend(&lc, mask)
        .unwrap()
}

fn close(got: &[Option<f64>], want: impl Fn(f64) -> f64, time: &[f64], tol: f64) {
    for (g, &t) in got.iter().zip(time) {
        let g = g.expect("a trend");
        assert!((g - want(t)).abs() < tol, "{t} {g} {}", want(t));
    }
}

// A cubic is a spline on any knots: it comes back, masked points included.
#[test]
fn a_cubic_comes_back() {
    let time: Vec<f64> = (0..500).map(|i| 1500.0 + f64::from(i) * 0.01).collect();
    let cubic = |t: f64| {
        let x = t - 1502.0;
        1.0 + 1e-3 * x - 2e-4 * x * x + 5e-5 * x * x * x
    };
    let flux: Vec<f64> = time.iter().map(|&t| cubic(t)).collect();
    let masked: Vec<bool> = time.iter().map(|&t| (t - 1501.5).abs() < 0.05).collect();
    for window in [0.3, 1.0, 10.0] {
        close(
            &spline(&time, &flux, window, Some(&masked)),
            cubic,
            &time,
            1e-14,
        );
    }
}

// A transit masked: the spline passes under it, as the star does.
#[test]
fn a_masked_transit_is_bridged() {
    let time: Vec<f64> = (0..1440).map(|i| f64::from(i) / 360.0).collect();
    let star = |t: f64| 1.0 + 2e-3 * (t / 1.3).sin();
    let flux: Vec<f64> = time
        .iter()
        .map(|&t| star(t) - if (t - 2.0).abs() < 0.06 { 5e-3 } else { 0.0 })
        .collect();
    let masked: Vec<bool> = time.iter().map(|&t| (t - 2.0).abs() < 0.1).collect();
    close(&spline(&time, &flux, 0.5, Some(&masked)), star, &time, 2e-6);
    // Unmasked, the dip pulls the spline down.
    let pulled = spline(&time, &flux, 0.5, None)[720].unwrap();
    assert!(pulled < star(2.0) - 1e-4, "{pulled}");
}

// A gap longer than the knot spacing (in one segment): the coefficients
// there are set by smoothness, so a line stays a line across it.
#[test]
fn a_gap_is_bridged_smoothly() {
    let time: Vec<f64> = (0..400)
        .map(|i| f64::from(i) * 0.01)
        .filter(|&t| !(1.0..3.0).contains(&t))
        .collect();
    let line = |t: f64| 2.0 - 0.1 * t;
    let flux: Vec<f64> = time.iter().map(|&t| line(t)).collect();
    let got = spline(&time, &flux, 0.25, None);
    close(&got, line, &time, 1e-9);
}

// Too few points for the knots: one gives a constant, two a line, three a
// quadratic; every other point masked.
#[test]
fn few_points() {
    let time: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.1).collect();
    fn quadratic(t: f64) -> f64 {
        1.0 + 0.2 * t - 0.03 * t * t
    }
    let flux: Vec<f64> = time.iter().map(|&t| quadratic(t)).collect();
    for (kept, want) in [
        (vec![10], (|_| quadratic(1.0)) as fn(f64) -> f64),
        (vec![10, 30], |t| {
            let (a, b) = (quadratic(1.0), quadratic(3.0));
            a + (b - a) * (t - 1.0) / 2.0
        }),
        (vec![10, 30, 45], quadratic),
    ] {
        let masked: Vec<bool> = (0..50).map(|i| !kept.contains(&i)).collect();
        close(&spline(&time, &flux, 0.5, Some(&masked)), want, &time, 1e-6);
    }
}

// Segments with every point masked get no trend; one point alone is its
// own trend; the edge cutoff leaves out the ends; a window far below the
// spacing gives no more intervals than points.
#[test]
fn segments_and_edges() {
    let time = [0.0, 0.1, 0.2, 0.3, 5.0, 9.0, 9.1, 9.2];
    let flux = [1.0, 2.0, 3.0, 4.0, 7.0, 1.0, 1.5, 2.0];
    let lc = LightCurve::from_btjd(time.to_vec(), flux.to_vec(), vec![1.0; 8]).unwrap();
    let masked = [false, false, false, false, false, true, true, true];
    let d = Detrend::default()
        .with_method(Method::Spline)
        .with_window(days(1.0))
        .with_break_tolerance(BreakTolerance::Fixed(days(1.0)));
    let got = d.trend(&lc, Some(&masked)).unwrap();
    for (g, f) in got[..5].iter().zip(&flux) {
        assert!((g.unwrap() - f).abs() < 1e-12, "{got:?}");
    }
    assert_eq!(got[5..], [None, None, None]);
    let cut = d.with_edge_cutoff(days(0.1)).trend(&lc, None).unwrap();
    assert_eq!([cut[0], cut[4], cut[5]], [None, None, None]);
    assert!(cut[1].is_some());
    let tiny = d.with_window(days(1e-9)).trend(&lc, None).unwrap();
    for (g, f) in tiny.iter().zip(&flux) {
        assert!((g.unwrap() - f).abs() < 1e-9, "{tiny:?}");
    }
}

#[test]
fn the_window_is_checked() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0; 2], vec![1.0; 2]).unwrap();
    let d = Detrend::default()
        .with_method(Method::Spline)
        .with_window(days(-1.0));
    assert_eq!(
        d.trend(&lc, None),
        Err(rustellar_detrend::DetrendError::NotPositive { name: "window" })
    );
}
