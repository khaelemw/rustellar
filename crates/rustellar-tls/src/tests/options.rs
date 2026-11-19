// The public options: TLS's defaults, and each setting checked.

use std::num::NonZeroUsize;

use rustellar_model::{LimbDarkening, RadiusRatio};

use crate::durations;
use crate::template;
use crate::{Device, Method, SearchOptions, Stars, Template, TlsError};

#[test]
fn the_defaults_are_tlss() {
    let options = SearchOptions::default();
    assert_eq!(options.template, Template::Default);
    assert_eq!(options.stars.build(), Ok(durations::Stars::TLS));
    assert_eq!(options.min_depth, 10e-6);
    assert_eq!(options.duration_step, 1.1);
    assert_eq!(options.steps.get(), 50);
    assert_eq!(options.oversampling.get(), 3);
    assert_eq!(options.refine_peaks, 5);
    assert_eq!(options.method, Method::Cells);
    assert_eq!(options.device, Device::Auto);
    assert_eq!(options.check(), Ok(()));
}

#[test]
fn each_template_is_built() {
    assert_eq!(
        Template::Default.build(),
        Ok(template::Template::tls_default())
    );
    assert_eq!(
        Template::Grazing.build(),
        Ok(template::Template::tls_grazing())
    );
    assert_eq!(Template::Box.build(), Ok(template::Template::tls_box()));
    let k = RadiusRatio::new(0.1).unwrap();
    let ld = LimbDarkening::quadratic(0.4, 0.2).unwrap();
    assert_eq!(
        Template::Planet { k, b: 0.5, ld }.build(),
        Ok(template::Template::new(k, 0.5, ld).unwrap())
    );
    // Beyond the star's edge, or a negative impact parameter.
    for b in [1.1, -0.1, f64::NAN] {
        assert_eq!(
            Template::Planet { k, b, ld }.build(),
            Err(TlsError::TemplateNotTransiting)
        );
    }
}

#[test]
fn each_star_range_is_checked() {
    let tls = Stars::default();
    let bad = [
        (
            Stars {
                radius_min: f64::NAN,
                ..tls
            },
            TlsError::NonFinite {
                name: "stars.radius_min",
            },
        ),
        (
            Stars {
                radius_max: f64::INFINITY,
                ..tls
            },
            TlsError::NonFinite {
                name: "stars.radius_max",
            },
        ),
        (
            Stars {
                mass_min: 0.0,
                ..tls
            },
            TlsError::NotPositive {
                name: "stars.mass_min",
            },
        ),
        (
            Stars {
                mass_max: -1.0,
                ..tls
            },
            TlsError::NotPositive {
                name: "stars.mass_max",
            },
        ),
        (
            Stars {
                radius_min: 4.0,
                ..tls
            },
            TlsError::StarRangeEmpty,
        ),
        (
            Stars {
                mass_min: 2.0,
                ..tls
            },
            TlsError::StarRangeEmpty,
        ),
    ];
    for (stars, error) in bad {
        assert_eq!(stars.build(), Err(error), "{stars:?}");
    }
    // One star.
    let sun = Stars {
        radius_min: 1.0,
        radius_max: 1.0,
        mass_min: 1.0,
        mass_max: 1.0,
    };
    assert!(sun.build().is_ok());
}

#[test]
fn each_setting_is_checked() {
    let options = SearchOptions::default();
    assert_eq!(
        options.with_min_depth(f64::NAN).check(),
        Err(TlsError::NonFinite { name: "min_depth" })
    );
    assert_eq!(
        options.with_duration_step(f64::INFINITY).check(),
        Err(TlsError::NonFinite {
            name: "duration_step"
        })
    );
    for step in [1.0, 0.5] {
        assert_eq!(
            options.with_duration_step(step).check(),
            Err(TlsError::DurationStepNotAboveOne)
        );
    }
    // A negative least depth is allowed: every fit that dims counts.
    assert_eq!(options.with_min_depth(-1.0).check(), Ok(()));
}

#[test]
fn each_builder_sets_its_setting() {
    let ten = NonZeroUsize::new(10).unwrap();
    let options = SearchOptions::default()
        .with_steps(ten)
        .with_oversampling(ten)
        .with_refine_peaks(10)
        .with_template(Template::Box)
        .with_stars(Stars::default())
        .with_min_depth(1e-4)
        .with_duration_step(1.2)
        .with_method(Method::Fold)
        .with_threads(ten)
        .with_device(Device::Gpu(1));
    assert_eq!(
        (options.steps, options.oversampling, options.refine_peaks),
        (ten, ten, 10)
    );
    assert_eq!(
        (options.template, options.method),
        (Template::Box, Method::Fold)
    );
    assert_eq!((options.min_depth, options.duration_step), (1e-4, 1.2));
    assert_eq!((options.threads, options.device), (ten, Device::Gpu(1)));
}
