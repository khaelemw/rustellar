// The reference curves built again: every file read, every set's times,
// and the noise drawn as the reference data were drawn.

use crate::curves::{Kind, raw};
use crate::random::Random;
use crate::{light_curve, rows};

#[test]
fn every_set_and_its_file() {
    let counts = [
        300, 200, 200, 40, 24, 100, 100, 100, 100, 100, 40, 40, 100, 100,
    ];
    for (kind, count) in Kind::ALL.into_iter().zip(counts) {
        assert_eq!(Kind::from_name(kind.name()), Some(kind));
        let rows = rows(kind.reference());
        assert_eq!(rows.len(), count, "{kind:?}");
        let lc = light_curve(kind, &rows[0]);
        let n = lc.time().len();
        let expected = match kind {
            // A sparse curve keeps about three quarters.
            Kind::Many if rows[0].sparse => n,
            // Flagged points and NaNs are dropped, and a detrended curve
            // may lose a few at its ends.
            Kind::Flagged | Kind::Variable => n,
            Kind::Many
            | Kind::Varying
            | Kind::Long
            | Kind::Deep
            | Kind::Grazing
            | Kind::Eccentric => 1296,
            Kind::Ultrashort => 1440,
            Kind::Fast => 25_920,
            Kind::Gaps => 3 * (1296 - 48),
            Kind::Mixed => 648 + 1872,
            Kind::Short => 27 * 720,
            Kind::Sectors => 52 * 720,
        };
        assert_eq!(n, expected, "{kind:?}");
    }
    assert_eq!(Kind::from_name("none"), None);
}

// Each regime's curves as the reference files' generator built them: the
// same finite points, and their flux summed to within what batman and
// rustellar-model differ by (a few parts in 10^8 a point), where noise
// drawn out of turn would be off by 10^-3 or more.
#[test]
fn the_regimes_built_as_the_references() {
    for kind in &Kind::ALL[5..] {
        let csv = kind.reference();
        let header: Vec<&str> = csv.lines().next().unwrap().split(',').collect();
        let at = |name: &str| header.iter().position(|h| *h == name).unwrap();
        let (points, sum) = (at("points"), at("flux_sum"));
        for (row, line) in rows(csv).iter().zip(csv.lines().skip(1)) {
            let v: Vec<f64> = line.split(',').map(|x| x.parse().unwrap()).collect();
            let lc = raw(*kind, row);
            assert_eq!(lc.len() as f64, v[points], "{kind:?} {}", row.seed);
            let ours: f64 = lc.flux().iter().sum();
            assert!(
                (ours - v[sum]).abs() < 1e-7 * v[points],
                "{kind:?} {}: {ours} against {}",
                row.seed,
                v[sum]
            );
        }
    }
    // The flagged curves lose about 3% of their points.
    let flagged = rows(Kind::Flagged.reference());
    let n = light_curve(Kind::Flagged, &flagged[0]).len();
    assert!((1200..1290).contains(&n), "{n}");
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
fn the_reference_random_numbers() {
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
