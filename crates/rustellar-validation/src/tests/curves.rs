// The reference curves built again: every file read, every set's times,
// and the noise drawn as the generators draw it.

use crate::curves::Kind;
use crate::random::Random;
use crate::{light_curve, rows};

#[test]
fn every_set_and_its_file() {
    let counts = [300, 200, 200, 40, 24];
    for (kind, count) in Kind::ALL.into_iter().zip(counts) {
        assert_eq!(Kind::from_name(kind.name()), Some(kind));
        let rows = rows(kind.reference());
        assert_eq!(rows.len(), count, "{kind:?}");
        let lc = light_curve(kind, &rows[0]);
        let n = lc.time().len();
        let expected = match kind {
            // A sparse curve keeps about three quarters.
            Kind::Many if rows[0].sparse => n,
            Kind::Many | Kind::Varying => 1296,
            Kind::Mixed => 648 + 1872,
            Kind::Short => 27 * 720,
            Kind::Sectors => 52 * 720,
        };
        assert_eq!(n, expected, "{kind:?}");
    }
    assert_eq!(Kind::from_name("none"), None);
}

// A sparse curve drops about a quarter of its points.
#[test]
fn sparse_curves_drop_a_quarter() {
    let rows = rows(Kind::Many.reference());
    let sparse = rows.iter().find(|r| r.sparse).unwrap();
    let n = light_curve(Kind::Many, sparse).time().len();
    assert!((900..1050).contains(&n), "{n}");
    assert!(rows.iter().any(|r| !r.sparse));
}

// SplitMix64's first outputs from seed 0, as published with it, and noise
// of unit variance.
#[test]
fn the_generators_random_numbers() {
    let mut random = Random::new(0);
    assert_eq!(
        random.uniform(),
        (0xE220_A839_7B1D_CDAF_u64 >> 11) as f64 / 2f64.powi(53)
    );
    let n = 100_000;
    let variance: f64 = (0..n).map(|_| random.noise().powi(2)).sum::<f64>() / f64::from(n);
    assert!((variance - 1.0).abs() < 0.02, "{variance}");
}

#[test]
#[should_panic(expected = "no column seed")]
fn a_missing_column() {
    let _ = rows("k,period\n0.1,3.0");
}
