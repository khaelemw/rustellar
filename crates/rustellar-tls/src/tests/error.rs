// Every error says what is wrong.

use crate::TlsError;

#[test]
fn each_error_says_what_is_wrong() {
    let cases = [
        (
            TlsError::NonFinite { name: "min_depth" },
            "min_depth must be finite",
        ),
        (
            TlsError::NotPositive {
                name: "stars.mass_min",
            },
            "stars.mass_min must be greater than zero",
        ),
        (
            TlsError::NoBaseline,
            "the light curve must span more than one time",
        ),
        (
            TlsError::OutOfRange,
            "a weight is out of the range an f64 holds precisely",
        ),
        (
            TlsError::DurationStepNotAboveOne,
            "duration_step must be above 1",
        ),
        (
            TlsError::StarRangeEmpty,
            "a range of stars has its least above its greatest",
        ),
        (
            TlsError::TemplateNotTransiting,
            "the template's planet does not transit",
        ),
        (TlsError::GpuUnavailable, "the GPU asked for cannot be used"),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
        let boxed: Box<dyn std::error::Error> = Box::new(error);
        assert!(boxed.source().is_none());
    }
}
