use rstest::rstest;
use rustellar_model::{ImpactParameter, ParamError, RadiusRatio, ScaledSemiMajorAxis};

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
