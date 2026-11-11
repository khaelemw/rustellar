use rstest::rstest;
use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, LimbDarkeningError, ParamError, RadiusRatio,
    ScaledSemiMajorAxis, TransitParams,
};

// ==========================
// Helper Functions
// ==========================

// Each constructor, made to return the plain number, so one test can cover
// all three types.
type Constructor = fn(f64) -> Result<f64, ParamError>;

fn k(x: f64) -> Result<f64, ParamError> {
    RadiusRatio::new(x).map(RadiusRatio::get)
}

fn b(x: f64) -> Result<f64, ParamError> {
    ImpactParameter::new(x).map(ImpactParameter::get)
}

fn a_rs(x: f64) -> Result<f64, ParamError> {
    ScaledSemiMajorAxis::new(x).map(ScaledSemiMajorAxis::get)
}

// Transit parameters with example B values (guide 3.4) apart from the three
// that `TransitParams::new` checks against each other.
fn transit(period_days: f64, a_rs: f64, b: f64) -> Result<TransitParams, ParamError> {
    TransitParams::new(
        Epoch::from_btjd(0.0).unwrap(),
        TimeDelta::from_days(period_days).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(a_rs).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(0.3, 0.25).unwrap(),
    )
}

// The same, but with the orbit given by its inclination.
fn transit_from_inc(
    period_days: f64,
    a_rs: f64,
    inc_deg: f64,
) -> Result<TransitParams, ParamError> {
    TransitParams::from_inclination_deg(
        Epoch::from_btjd(0.0).unwrap(),
        TimeDelta::from_days(period_days).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(a_rs).unwrap(),
        inc_deg,
        LimbDarkening::quadratic(0.3, 0.25).unwrap(),
    )
}

// ==========================
// Tests
// ==========================

#[rstest]
#[case::k_tiny(k, 1e-6)]
#[case::k_typical(k, 0.1)]
#[case::k_larger_than_star(k, 1.5)]
#[case::b_central(b, 0.0)]
#[case::b_typical(b, 0.5)]
#[case::b_missing_star(b, 2.0)]
#[case::a_rs_just_outside_star(a_rs, 1.000_000_1)]
#[case::a_rs_typical(a_rs, 10.0)]
#[case::a_rs_earth(a_rs, 215.0)]
fn params_accept_valid_value(#[case] constructor: Constructor, #[case] value: f64) {
    assert_eq!(constructor(value), Ok(value));
}

#[rstest]
#[case::k_zero(k, 0.0, ParamError::RadiusRatioNotPositive)]
#[case::k_negative_zero(k, -0.0, ParamError::RadiusRatioNotPositive)]
#[case::k_negative(k, -0.1, ParamError::RadiusRatioNotPositive)]
#[case::b_just_negative(b, -1e-12, ParamError::ImpactParameterNegative)]
#[case::b_negative(b, -0.5, ParamError::ImpactParameterNegative)]
#[case::a_rs_at_surface(a_rs, 1.0, ParamError::SemiMajorAxisTooSmall)]
#[case::a_rs_inside_star(a_rs, 0.5, ParamError::SemiMajorAxisTooSmall)]
#[case::a_rs_negative(a_rs, -10.0, ParamError::SemiMajorAxisTooSmall)]
fn params_reject_out_of_range(
    #[case] constructor: Constructor,
    #[case] value: f64,
    #[case] expected: ParamError,
) {
    assert_eq!(constructor(value), Err(expected));
}

#[rstest]
#[case::k(k, "k")]
#[case::b(b, "b")]
#[case::a_rs(a_rs, "a_rs")]
fn params_reject_non_finite(
    #[case] constructor: Constructor,
    #[case] name: &'static str,
    #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] value: f64,
) {
    assert_eq!(constructor(value), Err(ParamError::NonFinite { name }));
}

#[test]
fn limb_darkening_coefficients() {
    assert_eq!(
        LimbDarkening::uniform().quadratic_coefficients(),
        (0.0, 0.0)
    );
    assert_eq!(
        LimbDarkening::linear(0.6).unwrap().quadratic_coefficients(),
        (0.6, 0.0)
    );
    assert_eq!(
        LimbDarkening::quadratic(0.3, 0.25)
            .unwrap()
            .quadratic_coefficients(),
        (0.3, 0.25)
    );
}

#[rstest]
#[case::uniform_as_quadratic(0.0, 0.0)]
#[case::sun_like(0.3, 0.25)]
#[case::edge_exactly_dark(1.0, 0.0)]
#[case::limb_brightening(-0.2, 0.0)]
#[case::lowest_point_inside_but_positive(0.5, -0.5)]
#[case::lowest_point_past_edge(1.0, -0.3)]
#[case::lowest_point_before_centre(-0.5, -0.5)]
fn quadratic_accepts_non_negative_intensity(#[case] u1: f64, #[case] u2: f64) {
    assert!(LimbDarkening::quadratic(u1, u2).is_ok());
}

#[rstest]
#[case::negative_at_edge(0.8, 0.3)]
// Positive at the centre and at the edge, but negative in between.
#[case::negative_inside(4.5, -4.0)]
fn quadratic_rejects_negative_intensity(#[case] u1: f64, #[case] u2: f64) {
    assert_eq!(
        LimbDarkening::quadratic(u1, u2),
        Err(LimbDarkeningError::NegativeIntensity)
    );
}

#[rstest]
#[case::edge_exactly_dark(1.0, true)]
#[case::limb_brightening(-0.5, true)]
#[case::negative_at_edge(1.1, false)]
fn linear_accepts_only_non_negative_intensity(#[case] u1: f64, #[case] accepted: bool) {
    assert_eq!(LimbDarkening::linear(u1).is_ok(), accepted);
}

#[rstest]
fn limb_darkening_rejects_non_finite(
    #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] bad: f64,
) {
    let expected = Err(LimbDarkeningError::NonFinite);
    assert_eq!(LimbDarkening::linear(bad), expected);
    assert_eq!(LimbDarkening::quadratic(bad, 0.0), expected);
    assert_eq!(LimbDarkening::quadratic(0.0, bad), expected);
}

#[test]
fn transit_params_keep_inputs() {
    let t0 = Epoch::from_btjd(1234.5).unwrap();
    let period = TimeDelta::from_days(3.0).unwrap();
    let k = RadiusRatio::new(0.1).unwrap();
    let a_rs = ScaledSemiMajorAxis::new(10.0).unwrap();
    let b = ImpactParameter::new(0.3).unwrap();
    let ld = LimbDarkening::quadratic(0.3, 0.25).unwrap();

    let params = TransitParams::new(t0, period, k, a_rs, b, ld).unwrap();

    assert_eq!(params.t0(), t0);
    assert_eq!(params.period(), period);
    assert_eq!(params.k(), k);
    assert_eq!(params.a_rs(), a_rs);
    assert_eq!(params.b(), b);
    assert_eq!(params.limb_darkening(), ld);
}

#[rstest]
#[case::zero(0.0)]
#[case::negative_zero(-0.0)]
#[case::negative(-1.0)]
fn transit_params_reject_non_positive_period(#[case] period_days: f64) {
    assert_eq!(
        transit(period_days, 10.0, 0.3),
        Err(ParamError::PeriodNotPositive)
    );
}

#[rstest]
#[case::central(0.0, 10.0, true)]
#[case::grazing(1.05, 10.0, true)]
#[case::missing_star(5.0, 10.0, true)]
#[case::just_below_a_rs(1.999_999, 2.0, true)]
#[case::equal_to_a_rs(2.0, 2.0, false)]
#[case::above_a_rs(3.0, 2.0, false)]
fn transit_params_need_b_below_a_rs(#[case] b: f64, #[case] a_rs: f64, #[case] accepted: bool) {
    let result = transit(3.0, a_rs, b);

    if accepted {
        assert!(result.is_ok(), "b = {b}, a_rs = {a_rs}: {result:?}");
    } else {
        assert_eq!(result, Err(ParamError::ImpactParameterTooLarge));
    }
}

#[test]
fn transit_params_check_period_before_b() {
    assert_eq!(transit(0.0, 2.0, 3.0), Err(ParamError::PeriodNotPositive));
}

#[test]
fn inclination_gives_known_b() {
    // 10 * cos(89 degrees). Passing degrees straight to `cos` would give 0.21.
    let params = transit_from_inc(3.0, 10.0, 89.0).unwrap();
    assert!((params.b().get() - 0.174_524_064_372_836).abs() < 1e-12);
}

#[test]
fn inclination_of_90_gives_central_transit() {
    let params = transit_from_inc(3.0, 10.0, 90.0).unwrap();
    assert!(params.b().get() < 1e-15, "b = {}", params.b().get());
}

#[rstest]
#[case::central(0.0, 10.0)]
#[case::typical(0.3, 10.0)]
#[case::near_edge(0.95, 10.0)]
#[case::missing_star(1.2, 10.0)]
#[case::close_orbit(0.5, 3.0)]
#[case::earth_like(5.0, 215.0)]
fn b_to_inclination_to_b_round_trips(#[case] b: f64, #[case] a_rs: f64) {
    let inc = transit(3.0, a_rs, b).unwrap().inclination_deg();
    let back = transit_from_inc(3.0, a_rs, inc).unwrap().b().get();
    assert!((back - b).abs() < 1e-12, "b = {b} came back as {back}");
}

#[rstest]
#[case::grazing(85.0)]
#[case::typical(87.5)]
#[case::near_central(89.9)]
#[case::central(90.0)]
fn inclination_to_b_to_inclination_round_trips(#[case] inc: f64) {
    let back = transit_from_inc(3.0, 10.0, inc).unwrap().inclination_deg();
    assert!(
        (back - inc).abs() < 1e-9,
        "{inc} degrees came back as {back}"
    );
}

#[rstest]
#[case::zero(0.0)]
#[case::negative(-1.0)]
#[case::just_above_90(90.000_001)]
#[case::retrograde(180.0)]
fn inclination_rejects_out_of_range(#[case] inc: f64) {
    assert_eq!(
        transit_from_inc(3.0, 10.0, inc),
        Err(ParamError::InclinationOutOfRange)
    );
}

#[rstest]
fn inclination_rejects_non_finite(#[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] inc: f64) {
    assert_eq!(
        transit_from_inc(3.0, 10.0, inc),
        Err(ParamError::NonFinite {
            name: "inclination"
        })
    );
}

#[test]
fn inclination_still_checks_period() {
    assert_eq!(
        transit_from_inc(0.0, 10.0, 89.0),
        Err(ParamError::PeriodNotPositive)
    );
}

#[test]
fn tiny_inclination_gives_b_equal_to_a_rs() {
    // cos of a tiny angle rounds to exactly 1, so b = a_rs, which `new` rejects.
    assert_eq!(
        transit_from_inc(3.0, 10.0, 1e-9),
        Err(ParamError::ImpactParameterTooLarge)
    );
}
