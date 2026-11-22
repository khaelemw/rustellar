// The registry of suites: lookups, the settings and a cheap run of each
// kind.

use std::num::NonZeroUsize;

use crate::suites::{self, Measure, SUITES, Settings};

#[test]
fn the_registry() {
    assert_eq!(SUITES.len(), 28);
    // Names are unique within each measure.
    for s in SUITES {
        assert_eq!(
            SUITES
                .iter()
                .filter(|t| t.name == s.name && t.measure == s.measure)
                .count(),
            1,
            "{}",
            s.name
        );
        assert!(s.minutes >= s.quick_minutes && s.quick_minutes > 0.0);
    }
    assert_eq!(suites::of("tls", Measure::Accuracy).len(), 14);
    assert_eq!(suites::of("detrend", Measure::Accuracy).len(), 4);
    assert_eq!(suites::of("vet", Measure::Speed)[0].name, "vet");
    assert!(suites::of("model", Measure::Accuracy).is_empty());
    assert_eq!(
        suites::find("limb", Measure::Accuracy).map(|s| s.crate_name),
        Some("bls")
    );
    assert!(suites::find("limb", Measure::Speed).is_none());
}

#[test]
fn the_settings() {
    let s = Settings::default();
    assert_eq!(s.limit, usize::MAX);
    assert_eq!(s.step, NonZeroUsize::MIN);
    let q = s.clone().quick();
    assert_eq!((q.limit, q.step.get()), (20, 125));
    // Quick never widens what was asked.
    let narrow = Settings {
        limit: 3,
        step: NonZeroUsize::new(200).unwrap(),
        ..Settings::default()
    };
    let q = narrow.quick();
    assert_eq!((q.limit, q.step.get()), (3, 200));
    let _ = Settings::default().cpu();
}

#[test]
fn a_run_of_each_kind() {
    let settings = Settings {
        limit: 0,
        ..Settings::default().cpu()
    };
    for name in ["many", "methods", "masking", "whitening", "window"] {
        let suite = suites::find(name, Measure::Accuracy).unwrap();
        let out = suites::run(suite, &settings);
        assert!(
            out.contains("(0 curves") || out.contains("### many (0 curves)"),
            "{out}"
        );
    }
    let out = suites::run(
        suites::find("aliases", Measure::Accuracy).unwrap(),
        &settings,
    );
    assert!(out.contains("(0 candidates,"), "{out}");
    let out = suites::run(suites::find("fit", Measure::Speed).unwrap(), &settings);
    assert!(out.starts_with("## rustellar-fit speed"), "{out}");
}
