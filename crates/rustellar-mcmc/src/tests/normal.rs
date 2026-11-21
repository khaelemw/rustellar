// The normal quantile against 50-digit values (mpmath), across both
// branches and the far tail, and at its edges.

#![allow(clippy::excessive_precision)]

use crate::normal::quantile;

#[test]
fn against_exact_values() {
    let exact = [
        (1e-300, -37.047_096_299_361_199_236_547),
        (1e-100, -21.273_453_560_965_324_294_18),
        (1e-20, -9.262_340_089_798_407_579_572),
        (1e-10, -6.361_340_902_404_056_199_1),
        (1e-5, -4.264_890_793_922_824_610_234),
        (0.001, -3.090_232_306_167_813_535_358),
        (0.02, -2.053_748_910_631_823_044_339),
        (0.0749, -1.440_238_267_527_963_663_112),
        (0.075, -1.439_531_470_938_455_934_95),
        (0.075_001, -1.439_524_406_603_521_482_707),
        (0.2, -0.841_621_233_572_914_165_522_5),
        (0.37, -0.331_853_346_436_816_589_992_5),
        (0.500_000_000_001, 2.506_572_823_701_860_466_856e-12),
        (0.8, 0.841_621_233_572_914_363_803_6),
        (0.925, 1.439_531_470_938_456_229_063),
        (0.95, 1.644_853_626_951_472_284_276),
        (0.999, 3.090_232_306_167_813_277_758),
        (0.999_999_999_9, 6.361_340_889_697_421_864_155),
        (1.0 - 2f64.powi(-53), 8.209_536_151_601_386_855_631),
    ];
    for (p, want) in exact {
        let got = quantile(p);
        assert!(((got - want) / want).abs() < 1e-15, "{p}: {got} vs {want}");
    }
    assert_eq!(quantile(0.5), 0.0);
}

#[test]
fn edges() {
    assert_eq!(quantile(0.0), f64::NEG_INFINITY);
    assert_eq!(quantile(1.0), f64::INFINITY);
    assert!(quantile(-0.1).is_nan());
    assert!(quantile(1.5).is_nan());
    assert!(quantile(f64::NAN).is_nan());
    // Symmetric about 1/2 where p - 1/2 and 1 - p are exact.
    for p in [0.25, 0.375, 0.0625, 0.031_25, 2f64.powi(-20)] {
        assert_eq!(quantile(p), -quantile(1.0 - p));
    }
}
