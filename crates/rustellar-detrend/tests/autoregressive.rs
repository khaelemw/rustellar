//! `Autoregressive` against statsmodels 0.15's `AutoReg` and
//! `ar_select_order` (trend "c") and an exact least-squares fit (60 digits,
//! rounded once), on autoregressive series about 1, with a leftover wave,
//! in counts, and close to a unit root; then against its definition on
//! light curves with gaps.

use rustellar_core::LightCurve;
use rustellar_detrend::{Autoregressive, DetrendError, MAX_ORDER, Order};

const CASES: &str = include_str!("reference/ar_cases.csv");
const POINTS: &str = include_str!("reference/ar_points.csv");

struct Case {
    kind: &'static str,
    fixed: usize,
    statsmodels: Vec<f64>,
    exact: Vec<f64>,
    sigma2: f64,
    most: usize,
    chosen: usize,
    statsmodels_chosen: Vec<f64>,
    exact_chosen: Vec<f64>,
    flux: Vec<f64>,
}

fn numbers(s: &str) -> Vec<f64> {
    s.split(' ').map(|v| v.parse().unwrap()).collect()
}

fn cases() -> Vec<Case> {
    let rows = |text: &'static str| {
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .skip(1)
            .map(|l| l.split(',').collect::<Vec<_>>())
    };
    let mut cases: Vec<Case> = rows(CASES)
        .map(|r| Case {
            kind: r[1],
            fixed: r[2].parse().unwrap(),
            statsmodels: numbers(r[3]),
            exact: numbers(r[4]),
            sigma2: r[5].parse().unwrap(),
            most: r[6].parse().unwrap(),
            chosen: r[7].parse().unwrap(),
            statsmodels_chosen: numbers(r[8]),
            exact_chosen: numbers(r[9]),
            flux: vec![],
        })
        .collect();
    for r in rows(POINTS) {
        cases[r[0].parse::<usize>().unwrap()]
            .flux
            .push(r[1].parse().unwrap());
    }
    cases
}

/// Every 2 minutes, no gaps.
fn regular(flux: &[f64]) -> LightCurve {
    let n = flux.len();
    #[expect(clippy::cast_precision_loss, reason = "small")]
    let time = (0..n).map(|i| i as f64 / 720.0).collect();
    LightCurve::from_btjd(time, flux.to_vec(), vec![1e-3; n]).unwrap()
}

/// The largest error of `got` from `exact`, each coefficient in ulps of
/// its exact value.
fn ulps(got: &[f64], exact: &[f64]) -> f64 {
    assert_eq!(got.len(), exact.len());
    got.iter()
        .zip(exact)
        .map(|(g, e)| (g - e).abs() / (e.abs() * f64::EPSILON))
        .fold(0.0, f64::max)
}

fn params(white: &rustellar_detrend::Whitened) -> Vec<f64> {
    let mut p = vec![white.constant()];
    p.extend_from_slice(white.coefficients());
    p
}

#[test]
fn fixed_order_against_statsmodels_and_exact() {
    let (mut ours, mut theirs) = (vec![], vec![]);
    for (i, c) in cases().iter().enumerate() {
        let white = Autoregressive::default()
            .with_order(Order::Fixed(c.fixed))
            .whiten(&regular(&c.flux))
            .unwrap();
        assert_eq!(white.order(), c.fixed);
        let (us, them) = (
            ulps(&params(&white), &c.exact),
            ulps(&c.statsmodels, &c.exact),
        );
        // A few ulps; the unit-root cases' constant cancels to near zero.
        let bound = if c.kind == "unit root" { 64.0 } else { 8.0 };
        assert!(
            us <= bound,
            "case {i} ({}): {us} ulps (statsmodels {them})",
            c.kind
        );
        assert!(
            ((white.variance() - c.sigma2) / c.sigma2).abs() < 1e-9,
            "case {i}: {} {}",
            white.variance(),
            c.sigma2
        );
        ours.push(us);
        theirs.push(them);
    }
    ours.sort_by(f64::total_cmp);
    theirs.sort_by(f64::total_cmp);
    let mid = ours.len() / 2;
    // Ahead of statsmodels' least squares at the median and at the worst.
    assert!(ours[mid] < theirs[mid], "{} {}", ours[mid], theirs[mid]);
    assert!(ours[ours.len() - 1] < theirs[theirs.len() - 1]);
}

#[test]
fn aic_order_against_statsmodels() {
    for (i, c) in cases().iter().enumerate() {
        let white = Autoregressive::default()
            .with_order(Order::Aic(c.most))
            .whiten(&regular(&c.flux))
            .unwrap();
        assert_eq!(white.order(), c.chosen, "case {i}");
        let us = ulps(&params(&white), &c.exact_chosen);
        let bound = if c.kind == "unit root" { 64.0 } else { 8.0 };
        assert!(us <= bound, "case {i}: {us}");
        // statsmodels' refit is the same least squares.
        assert!(ulps(&c.statsmodels_chosen, &c.exact_chosen) < 1e6);
    }
}

#[test]
fn residuals_by_definition_with_gaps() {
    // Three runs: 2-minute cadence, then a 5-cadence gap, then one missing
    // cadence, then the rest.
    let mut time = vec![];
    let mut flux = vec![];
    let mut state = 7_u64;
    for i in 0..300_u32 {
        if (100..105).contains(&i) || i == 200 {
            continue;
        }
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[expect(clippy::cast_precision_loss, reason = "a fixed generator")]
        let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
        time.push(f64::from(i) / 720.0);
        flux.push(1.0 + 1e-3 * noise + 5e-4 * (f64::from(i) / 9.0).sin());
    }
    let n = time.len();
    let err: Vec<f64> = (0..n)
        .map(|i| 1e-3 + 1e-6 * f64::from(u32::try_from(i).unwrap()))
        .collect();
    let lc = LightCurve::from_btjd(time, flux.clone(), err.clone()).unwrap();
    let white = Autoregressive::default()
        .with_order(Order::Fixed(2))
        .whiten(&lc)
        .unwrap();
    // The first two of each run cannot be predicted.
    let starts = [0, 100, 195];
    let expected: Vec<usize> = (0..n)
        .filter(|i| starts.iter().all(|s| !(*s..s + 2).contains(i)))
        .collect();
    assert_eq!(white.indices(), expected.as_slice());
    let (c, phi) = (white.constant(), white.coefficients());
    for (k, &i) in white.indices().iter().enumerate() {
        let e = flux[i] - c - phi[0] * flux[i - 1] - phi[1] * flux[i - 2];
        let got = white.light_curve().flux()[k] - 1.0;
        assert!((got - e).abs() < 1e-15, "{i}: {got} {e}");
        assert_eq!(white.light_curve().flux_err()[k], err[i]);
        assert_eq!(white.light_curve().time()[k], lc.time()[i]);
    }
    assert!((white.gain() - (1.0 - phi[0] - phi[1])).abs() < 1e-15);
    #[expect(clippy::cast_precision_loss, reason = "small")]
    let mean_square = white
        .light_curve()
        .flux()
        .iter()
        .map(|f| (f - 1.0) * (f - 1.0))
        .sum::<f64>()
        / white.indices().len() as f64;
    assert!((white.variance() / mean_square - 1.0).abs() < 1e-12);
    // The fit, by definition: the residuals are orthogonal to each column
    // (to the rounding of 1 + e, an ulp of 1 per point).
    for lag in 0..=2 {
        let dot: f64 = white
            .indices()
            .iter()
            .zip(white.light_curve().flux())
            .map(|(&i, f)| (f - 1.0) * if lag == 0 { 1.0 } else { flux[i - lag] })
            .sum();
        assert!(dot.abs() < 1e-13, "{lag}: {dot}");
    }
    let lc2 = white.clone().into_light_curve();
    assert_eq!(&lc2, white.light_curve());
}

#[test]
fn settings_and_errors() {
    assert_eq!(Autoregressive::default().order(), Order::Aic(MAX_ORDER));
    let a = Autoregressive::default().with_order(Order::Fixed(3));
    assert_eq!(a.order(), Order::Fixed(3));
    // Order 3: four coefficients, so five predictable points at least.
    let flux = [1.0, 1.1, 0.9, 1.2, 1.0, 0.95, 1.05, 1.0];
    assert_eq!(
        a.whiten(&regular(&flux[..7])).unwrap_err(),
        DetrendError::TooFewPoints
    );
    assert_eq!(
        a.whiten(&regular(&flux)).unwrap().indices(),
        &[3, 4, 5, 6, 7]
    );
    let aic = Autoregressive::default().with_order(Order::Aic(3));
    assert_eq!(
        aic.whiten(&regular(&flux[..7])).unwrap_err(),
        DetrendError::TooFewPoints
    );
    assert_eq!(
        DetrendError::TooFewPoints.to_string(),
        "too few points to fit the autoregressive model"
    );
    // One point: no steps, one run.
    assert_eq!(
        Autoregressive::default()
            .with_order(Order::Fixed(0))
            .whiten(&regular(&[1.0]))
            .unwrap_err(),
        DetrendError::TooFewPoints
    );
}

#[test]
fn constant_and_dependent_columns() {
    // A constant: every lag equals the constant column, so they are left
    // at zero and the constant takes it all, whatever order AIC picks
    // among sums of squares that are all rounding.
    let flat = vec![1.25; 100];
    let white = Autoregressive::default()
        .with_order(Order::Fixed(2))
        .whiten(&regular(&flat))
        .unwrap();
    assert_eq!(white.constant(), 1.25);
    assert_eq!(white.coefficients(), &[0.0, 0.0]);
    assert!(white.light_curve().flux().iter().all(|&f| f == 1.0));
    assert_eq!(white.variance(), 0.0);
    let white = Autoregressive::default().whiten(&regular(&flat)).unwrap();
    assert!(white.coefficients().iter().all(|&phi| phi == 0.0));
    assert!(white.light_curve().flux().iter().all(|&f| f == 1.0));
    // Alternating: x_t = 2 - x_(t-1) exactly.
    let zigzag: Vec<f64> = (0..60)
        .map(|i| if i % 2 == 0 { 0.5 } else { 1.5 })
        .collect();
    let white = Autoregressive::default()
        .with_order(Order::Fixed(1))
        .whiten(&regular(&zigzag))
        .unwrap();
    assert!((white.constant() - 2.0).abs() < 1e-14);
    assert!((white.coefficients()[0] + 1.0).abs() < 1e-14);
    assert_eq!(white.gain(), 2.0);
}
