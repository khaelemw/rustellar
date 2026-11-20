// The alias study on a few curves: a found planet given at every ratio and
// pointed back each time, a curve too faint for TLS left out, and TLS's
// own alias kept as it is.

use std::num::NonZeroUsize;

use rustellar_vet::Ratio;

use crate::Kind;
use crate::vet::{Source, cases, median_text, report};

fn two() -> NonZeroUsize {
    NonZeroUsize::new(2).unwrap()
}

#[test]
fn found_planets_at_every_ratio() {
    // The first three: two planets TLS finds, and one too faint (a false
    // peak), which gives no case.
    let found = cases(Kind::Many, 3, two());
    assert_eq!(found.len(), 14);
    for c in &found {
        assert_eq!(c.kind, Kind::Many);
        assert!(c.aligned);
        assert_eq!(c.by_chi2, Some(c.truth()), "{c:?}");
        assert!(c.truth_snr.unwrap() > 7.0);
        let one = Ratio::new(1, 1).unwrap();
        assert_eq!(c.source == Source::Tls, c.given == one);
    }
    let table = report(&found);
    assert!(
        table.contains("| true | TLS 1.32 | 2 | 2 | 2 (100.0%) |"),
        "{table}"
    );
    assert!(
        table.contains("| 1/3 × true | moved | 2 | 2 | 2 (100.0%) |"),
        "{table}"
    );
    assert!(table.contains("| SNR at the true period |"), "{table}");
}

#[test]
fn an_alias_tls_found() {
    // The third curve of varying: TLS found half the true period.
    let found = cases(Kind::Varying, 3, two());
    let own: Vec<_> = found
        .iter()
        .filter(|c| c.source == Source::Tls && c.given == Ratio::new(1, 2).unwrap())
        .collect();
    assert_eq!(own.len(), 1);
    assert_eq!(own[0].truth(), Ratio::new(2, 1).unwrap());
    let table = report(&found);
    assert!(table.contains("| 1/2 × true | TLS 1.32 | 1 |"), "{table}");
}

#[test]
fn no_cases() {
    assert!(cases(Kind::Many, 0, two()).is_empty());
    let table = report(&[]);
    assert_eq!(table.lines().count(), 5, "{table}");
    assert_eq!(median_text(&[]), "-");
    assert_eq!(median_text(&[1.0, 2.0, 4.0]), "2.0");
}
