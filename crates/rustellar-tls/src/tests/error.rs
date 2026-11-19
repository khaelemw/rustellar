// Every error says what is wrong.

use crate::{GridError, Periods, TlsError};

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

// A period list's or grid's error, wrapped: `?` works on them, and the
// grid's error is the source.
#[test]
fn grid_errors_are_wrapped() {
    fn build() -> Result<Periods, TlsError> {
        Ok(Periods::from_days(&[])?)
    }
    let error = build().unwrap_err();
    assert_eq!(error, TlsError::Grid(GridError::NoPeriods));
    assert_eq!(
        error.to_string(),
        "the periods: at least one period is needed"
    );
    let source = std::error::Error::source(&error).unwrap();
    assert_eq!(source.to_string(), "at least one period is needed");
    assert!(std::error::Error::source(&TlsError::NoBaseline).is_none());
}
