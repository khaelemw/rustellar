use crate::fap::gamma_ratio;

#[test]
#[expect(
    clippy::excessive_precision,
    reason = "40-digit references, rounded once"
)]
fn gamma_ratio_to_a_few_roundings() {
    // G(x + 1/2) / G(x) at 40 digits (mpmath), rounded.
    for (x, want) in [
        (0.5, 0.564_189_583_547_756_286_948_079_5),
        (1.0, 0.886_226_925_452_758_013_649_083_7),
        (2.5, 1.504_505_556_127_350_098_528_212),
        (7.0, 2.598_964_313_608_039_370_105_663),
        (19.5, 4.387_667_120_628_882_278_244_735),
        (20.0, 4.444_275_161_239_915_144_577_549),
        (33.3, 5.748_995_165_713_226_005_899_667),
        (1e3, 31.618_824_001_815_912_821_167_01),
        (5e5, 707.106_604_409_874_324_878_496_7),
    ] {
        let got = gamma_ratio(x);
        assert!(
            (got - want).abs() <= 8.0 * f64::EPSILON * want,
            "{x}: {got} {want}"
        );
    }
}
