use rstest::rstest;
use rustellar_core::{Epoch, LightCurve, LightCurveError, TimeDelta};

// ==========================
// Helper Functions
// ==========================

fn ok() -> Vec<f64> {
    vec![1.0, 2.0, 3.0]
}

fn with(value: f64) -> Vec<f64> {
    vec![1.0, value, 3.0]
}

fn empty() -> Vec<f64> {
    vec![]
}

fn days(x: f64) -> TimeDelta {
    TimeDelta::from_days(x).unwrap()
}

fn btjd(x: f64) -> Epoch {
    Epoch::from_btjd(x).unwrap()
}

fn lc_from_times(time: Vec<f64>) -> LightCurve {
    let len = time.len();
    LightCurve::from_btjd(time, vec![1.0; len], vec![0.1; len]).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert!(actual.len() == expected.len());

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: got {a}, expected {e} (diff {}, tol {tol})",
            (a - e).abs()
        );
    }
}

// ==========================
// Tests
// ==========================

#[rstest]
#[case::typical(ok(), ok(), ok())]
#[case::single_point(vec![1.0], vec![1.0], vec![0.1])]
#[case::negative_times(vec![-3.0, -2.0, -1.0], ok(), ok())]
#[case::bjd_scale_times(vec![2460000.1, 2460000.2, 2460000.3], ok(), ok())]
#[case::zero_flux(ok(), with(0.0), ok())]
#[case::negative_flux(ok(), with(-0.5), ok())]
#[case::tiny_error(ok(), ok(), with(f64::MIN_POSITIVE))]
#[case::huge_values(ok(), with(1e300), with(1e300))]
fn from_btjd_accepts_valid_input(
    #[case] time: Vec<f64>,
    #[case] flux: Vec<f64>,
    #[case] flux_err: Vec<f64>,
) {
    let lc = LightCurve::from_btjd(time.clone(), flux.clone(), flux_err.clone()).unwrap();
    let time_back: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
    assert_eq!(time_back, time);
    assert_eq!(lc.flux(), flux);
    assert_eq!(lc.flux_err(), flux_err)
}

#[test]
fn new_accepts_epochs() {
    let time = vec![btjd(1.0), btjd(2.0), btjd(3.0)];
    let lc = LightCurve::new(time.clone(), ok(), ok()).unwrap();
    assert_eq!(lc.time(), time);
}

#[test]
fn from_btjd_matches_new() {
    let time = vec![btjd(1.0), btjd(2.0), btjd(3.0)];
    assert_eq!(
        LightCurve::from_btjd(ok(), ok(), ok()),
        LightCurve::new(time, ok(), ok())
    );
}

#[test]
fn new_returns_error_when_empty() {
    assert_eq!(
        LightCurve::new(vec![], vec![], vec![]),
        Err(LightCurveError::Empty)
    );
}

#[rstest]
#[case::time_empty(empty(), ok(), ok())]
#[case::flux_empty(ok(), empty(), ok())]
#[case::flux_err_empty(ok(), ok(), empty())]
#[case::time_and_flux_empty(empty(), empty(), ok())]
#[case::time_and_flux_err_empty(empty(), ok(), empty())]
#[case::flux_and_flux_err_empty(ok(), empty(), empty())]
fn new_returns_length_mismatch_error(
    #[case] time: Vec<f64>,
    #[case] flux: Vec<f64>,
    #[case] flux_err: Vec<f64>,
) {
    assert_eq!(
        LightCurve::from_btjd(time, flux, flux_err),
        Err(LightCurveError::LengthMismatch)
    );
}

#[rstest]
#[case::infinite_time(with(f64::INFINITY), ok(), ok())]
#[case::infinite_flux(ok(), with(f64::INFINITY), ok())]
#[case::infinite_flux_err(ok(), ok(), with(f64::INFINITY))]
#[case::negative_infinite_time(with(f64::NEG_INFINITY), ok(), ok())]
#[case::negative_infinite_flux(ok(), with(f64::NEG_INFINITY), ok())]
#[case::negative_infinite_flux_err(ok(), ok(), with(f64::NEG_INFINITY))]
#[case::nan_time(with(f64::NAN), ok(), ok())]
#[case::nan_flux(ok(), with(f64::NAN), ok())]
#[case::nan_flux_error(ok(), ok(), with(f64::NAN))]
fn new_returns_non_finite_error(
    #[case] time: Vec<f64>,
    #[case] flux: Vec<f64>,
    #[case] flux_err: Vec<f64>,
) {
    assert_eq!(
        LightCurve::from_btjd(time, flux, flux_err),
        Err(LightCurveError::NonFinite)
    );
}

#[test]
fn from_btjd_checks_time_before_lengths() {
    // A NaN time is reported even though the lengths also differ, as the docs
    // on `from_btjd` say.
    assert_eq!(
        LightCurve::from_btjd(vec![f64::NAN], ok(), ok()),
        Err(LightCurveError::NonFinite)
    );
}

#[rstest]
#[case::negative_flux_err(ok(), ok(), with(-1.0))]
#[case::zero_flux_err(ok(), ok(), with(0.0))]
fn new_returns_non_positive_error(
    #[case] time: Vec<f64>,
    #[case] flux: Vec<f64>,
    #[case] flux_err: Vec<f64>,
) {
    assert_eq!(
        LightCurve::from_btjd(time, flux, flux_err),
        Err(LightCurveError::NonPositiveError)
    );
}

#[rstest]
#[case::non_ascending_time(with(0.0), ok(), ok())]
#[case::duplicate_times(with(1.0), ok(), ok())]
fn new_returns_not_sorted_error(
    #[case] time: Vec<f64>,
    #[case] flux: Vec<f64>,
    #[case] flux_err: Vec<f64>,
) {
    assert_eq!(
        LightCurve::from_btjd(time, flux, flux_err),
        Err(LightCurveError::NotSorted)
    );
}

#[rstest]
#[case::one_point(vec![1.0], vec![1.0], vec![0.1], 1)]
#[case::three_points(ok(), ok(), ok(), 3)]
#[case::many_points((0..1000).map(f64::from).collect(), vec![1.0; 1000], vec![0.1; 1000], 1000)]
fn len_returns_number_of_points(
    #[case] time: Vec<f64>,
    #[case] flux: Vec<f64>,
    #[case] flux_err: Vec<f64>,
    #[case] expected: usize,
) {
    let lc = LightCurve::from_btjd(time, flux, flux_err).unwrap();
    assert_eq!(lc.len(), expected);
}

#[rstest]
#[case::zero(0.0)]
#[case::negative_zero(-0.0)]
#[case::negative(-1.0)]
fn fold_rejects_invalid_period(#[case] period: f64) {
    let lc = lc_from_times(ok());
    assert_eq!(
        lc.fold(days(period), btjd(0.0)),
        Err(LightCurveError::InvalidPeriod)
    )
}

#[rstest]
#[case::basic(vec![-0.5, 0.0, 0.5, 1.0, 1.5], 2.0, 0.0, vec![-0.5, -0.25, -0.25, 0.0, 0.25])]
#[case::shifted_t0(vec![0.3, 0.8, 1.3], 2.0, 0.3, vec![-0.5, 0.0, 0.25])]
#[case::before_t0(vec![-3.5], 2.0, 0.0, vec![0.25])]
#[case::many_orbits(vec![2000.5], 2.0, 0.0, vec![0.25])]
#[case::bjd_scale(vec![2460000.5, 2460001.0], 2.0, 2459999.0, vec![-0.25, 0.0])]
fn fold_gives_expected_phases(
    #[case] time: Vec<f64>,
    #[case] period: f64,
    #[case] t0: f64,
    #[case] expected: Vec<f64>,
) {
    let lc = lc_from_times(time);
    let folded = lc.fold(days(period), btjd(t0)).unwrap();
    assert_close(folded.phase(), &expected, 1e-9)
}

#[test]
fn fold_period_in_hours_matches_days() {
    let lc = lc_from_times(ok());
    let in_days = lc.fold(days(0.75), btjd(0.0)).unwrap();
    let in_hours = lc
        .fold(TimeDelta::from_hours(18.0).unwrap(), btjd(0.0))
        .unwrap();
    assert_eq!(in_days, in_hours);
}

#[test]
fn fold_t0_in_bjd_matches_btjd() {
    let lc = lc_from_times(vec![3000.5, 3001.0, 3001.5]);
    let from_bjd = lc
        .fold(days(2.0), Epoch::from_bjd(2_460_000.0).unwrap())
        .unwrap();
    let from_btjd = lc.fold(days(2.0), btjd(3000.0)).unwrap();
    assert_eq!(from_bjd, from_btjd);
}

#[test]
fn fold_phases_stay_in_range() {
    let time = vec![
        -1.0000000000000002,
        -1.0,
        -0.9999999999999999,
        0.9999999999999999,
        1.0,
        1.0000000000000002,
        2.9999999999999996,
        3.0,
        3.0000000000000004,
    ];
    let folded = lc_from_times(time).fold(days(2.0), btjd(0.0)).unwrap();

    for (i, &p) in folded.phase().iter().enumerate() {
        assert!(
            (-0.5..0.5).contains(&p),
            "index {i}: phase {p} is outside [-0.5, 0.5)"
        );
    }
}

#[test]
fn fold_keeps_rows_together() {
    // Phases before sorting are [0.0, 0.25, -0.5, -0.25], so the sorted row
    // order is the original rows 2, 3, 0, 1.
    let lc = LightCurve::from_btjd(
        vec![0.0, 0.5, 1.0, 1.5],
        vec![1.0, 0.98, 0.99, 1.01],
        vec![0.1, 0.2, 0.3, 0.4],
    )
    .unwrap();

    let folded = lc.fold(days(2.0), btjd(0.0)).unwrap();

    assert_eq!(folded.phase(), &[-0.5, -0.25, 0.0, 0.25]);
    assert_eq!(folded.flux(), &[0.99, 1.01, 1.0, 0.98]);
    assert_eq!(folded.flux_err(), &[0.3, 0.4, 0.1, 0.2]);
}

#[test]
fn fold_sorts_by_phase() {
    let time: Vec<f64> = (0..60).map(|i| f64::from(i) * 0.37).collect();
    let folded = lc_from_times(time).fold(days(2.0), btjd(0.0)).unwrap();

    assert_eq!(folded.phase().len(), 60);
    for (i, w) in folded.phase().windows(2).enumerate() {
        assert!(
            w[0] <= w[1],
            "index {i}: phase {} comes before {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn fold_equal_phases_keep_time_order() {
    // t = 0.5 and t = 2.5 are one period apart, so both fold to phase 0.25.
    let lc = LightCurve::from_btjd(vec![0.5, 2.5], vec![1.0, 2.0], vec![0.1, 0.1]).unwrap();

    let folded = lc.fold(days(2.0), btjd(0.0)).unwrap();

    assert_eq!(folded.phase(), &[0.25, 0.25]);
    assert_eq!(folded.flux(), &[1.0, 2.0]);
}
