use rstest::rstest;
use rustellar_model::{LimbDarkeningError, ParamError};

#[rstest]
#[case::non_finite(ParamError::NonFinite { name: "k" }, "k must be finite")]
#[case::radius_ratio_not_positive(
    ParamError::RadiusRatioNotPositive,
    "radius ratio k must be greater than zero"
)]
#[case::impact_parameter_negative(
    ParamError::ImpactParameterNegative,
    "impact parameter b must not be negative"
)]
#[case::semi_major_axis_too_small(
    ParamError::SemiMajorAxisTooSmall,
    "scaled semi-major axis a_rs must be greater than 1"
)]
#[case::period_not_positive(ParamError::PeriodNotPositive, "period must be greater than zero")]
#[case::impact_parameter_too_large(
    ParamError::ImpactParameterTooLarge,
    "impact parameter b must be less than a_rs"
)]
#[case::inclination_out_of_range(
    ParamError::InclinationOutOfRange,
    "inclination must be greater than 0 and at most 90 degrees"
)]
fn param_error_display_message(#[case] error: ParamError, #[case] expected: &str) {
    assert_eq!(error.to_string(), expected);
}

#[rstest]
#[case::non_finite(
    LimbDarkeningError::NonFinite,
    "limb-darkening coefficients must be finite"
)]
#[case::negative_intensity(
    LimbDarkeningError::NegativeIntensity,
    "limb-darkening coefficients give negative intensity on the stellar disk"
)]
fn limb_darkening_error_display_message(#[case] error: LimbDarkeningError, #[case] expected: &str) {
    assert_eq!(error.to_string(), expected);
}
