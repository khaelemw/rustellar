// T14 and the duration grid: the formula, its cap, its scaling with period,
// star and planet, and the grid's spacing and ends.

use std::f64::consts::PI;

use super::Random;
use crate::durations::{Constants, FRACTION_MAX, Planet, Stars, grid, t14};

const IAU: Constants = Constants::IAU;

#[test]
fn the_sun_and_earth() {
    // A year around the Sun: the orbit's radius from Kepler's third law,
    // a = (G M P^2 / 4 pi^2)^(1/3), 1.496e11 m; a central transit of a
    // small planet crosses 2 R at speed 2 pi a / P, for 12.98 hours.
    let seconds = 365.25 * 86_400.0;
    let a = (1.327_124_4e20 * seconds * seconds / (4.0 * PI * PI)).cbrt();
    assert!((a / 1.496e11 - 1.0).abs() < 1e-3, "{a}");
    let expected = 2.0 * 6.957e8 / (2.0 * PI * a);
    let fraction = t14(IAU, 1.0, 1.0, 365.25, Planet::Small);
    assert!(
        (fraction / expected - 1.0).abs() < 1e-15,
        "{fraction} {expected}"
    );
    let hours = fraction * 365.25 * 24.0;
    assert!((hours - 12.98).abs() < 0.01, "{hours}");
}

#[test]
fn scales_as_period_to_minus_two_thirds() {
    let mut random = Random::new(3);
    for _ in 0..1000 {
        let r = random.between(0.1, 3.0);
        let m = random.between(0.1, 2.0);
        let p = random.between(5.0, 300.0);
        let f = random.between(1.1, 8.0);
        let ratio = t14(IAU, r, m, p * f, Planet::Small) / t14(IAU, r, m, p, Planet::Small);
        assert!((ratio / f.powf(-2.0 / 3.0) - 1.0).abs() < 1e-14, "{ratio}");
        // And as the radius over the cube root of the mass.
        let ratio = t14(IAU, 2.0 * r, 8.0 * m, p, Planet::Small) / t14(IAU, r, m, p, Planet::Small);
        assert!((ratio - 1.0).abs() < 1e-15, "{ratio}");
    }
}

#[test]
fn a_large_planet_adds_two_jupiter_radii() {
    let small = t14(IAU, 1.0, 1.0, 50.0, Planet::Small);
    let large = t14(IAU, 1.0, 1.0, 50.0, Planet::Large);
    let expected = (6.957e8 + 2.0 * 6.9911e7) / 6.957e8;
    assert!((large / small - expected).abs() < 1e-15);
}

#[test]
fn capped_at_twelve_percent() {
    assert_eq!(t14(IAU, 3.5, 0.1, 0.2, Planet::Large), FRACTION_MAX);
    assert_eq!(t14(IAU, 100.0, 1.0, 10.0, Planet::Small), FRACTION_MAX);
    assert!(t14(IAU, 0.13, 1.0, 10.0, Planet::Small) < FRACTION_MAX);
}

#[test]
fn the_grid_steps_evenly_from_shortest_to_longest() {
    for (short, long, step) in [(0.5, 13.5, 1.1), (0.3, 40.0, 1.05), (1.0, 2.0, 1.3)] {
        let durations = grid(IAU, Stars::TLS, short, long, step);
        let first = Stars::TLS.shortest(IAU, long);
        let last = Stars::TLS.longest(IAU, short);
        assert_eq!(durations[0], first);
        assert_eq!(*durations.last().unwrap(), last);
        let n = durations.len();
        for pair in durations[..n - 1].windows(2) {
            assert_eq!(pair[1], pair[0] * step);
        }
        // The last step is at most `step`, and more than 1.
        let ratio = durations[n - 1] / durations[n - 2];
        assert!(ratio > 1.0 && ratio <= step, "{ratio}");
    }
}

#[test]
fn a_grid_of_one_period_still_has_both_ends() {
    // Durations from the shortest to the longest transit at one period.
    let durations = grid(IAU, Stars::TLS, 5.0, 5.0, 1.1);
    assert!(durations.len() > 2);
    assert!(durations.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn a_grid_whose_ends_meet_has_two_durations() {
    // One star and one size of planet at one period: the shortest is the
    // longest, and the grid is that duration twice (as TLS's).
    let stars = Stars {
        r_min: 1.0,
        r_max: 1.0,
        m_min: 1.0,
        m_max: 1.0,
    };
    let only = t14(IAU, 1.0, 1.0, 3.0, Planet::Small);
    let durations = grid(
        Constants {
            r_jupiter: 0.0,
            ..IAU
        },
        stars,
        3.0,
        3.0,
        1.1,
    );
    assert_eq!(durations, vec![only, only]);
}

// TLS's constants (`tls_constants.py`): G 6.673e-11, a solar mass of
// 1.989e30 kg (1.9890000000000002e30 as Python computes `1.989 * 10 ** 30`),
// R_sun 695,508 km, R_jup 69,911 km.
const TLS: Constants = Constants {
    gm_sun: 6.673e-11 * 1.989_000_000_000_000_2e30,
    r_sun: 6.955_08e8,
    r_jupiter: 6.9911e7,
};

// transitleastsquares 1.32's `T14` (its `py_func`, so without fast-math),
// for (R_star, M_star, period, planet, fraction).
const TLS_T14: [(f64, f64, f64, Planet, f64); 36] = [
    (0.13, 0.1, 0.5, Planet::Small, 0.033_618_194_675_526_59),
    (0.13, 0.1, 0.5, Planet::Large, 0.085_606_350_935_427_82),
    (0.13, 0.1, 13.0, Planet::Small, 0.003_830_529_598_200_043),
    (0.13, 0.1, 13.0, Planet::Large, 0.009_754_172_233_727_173),
    (0.13, 0.1, 100.0, Planet::Small, 0.000_983_001_975_579_330_7),
    (0.13, 0.1, 100.0, Planet::Large, 0.002_503_144_886_388_665),
    (0.13, 1.0, 0.5, Planet::Small, 0.015_604_183_701_214_48),
    (0.13, 1.0, 0.5, Planet::Large, 0.039_734_948_258_821_86),
    (0.13, 1.0, 13.0, Planet::Small, 0.001_777_974_340_982_856_4),
    (0.13, 1.0, 13.0, Planet::Large, 0.004_527_485_692_120_386_5),
    (0.13, 1.0, 100.0, Planet::Small, 0.000_456_269_099_326_832_3),
    (0.13, 1.0, 100.0, Planet::Large, 0.001_161_856_935_357_655_4),
    (1.0, 0.1, 0.5, Planet::Small, 0.12),
    (1.0, 0.1, 0.5, Planet::Large, 0.12),
    (1.0, 0.1, 13.0, Planet::Small, 0.029_465_612_293_846_487),
    (1.0, 0.1, 13.0, Planet::Large, 0.035_389_254_929_373_615),
    (1.0, 0.1, 100.0, Planet::Small, 0.007_561_553_658_302_543),
    (1.0, 0.1, 100.0, Planet::Large, 0.009_081_696_569_111_878),
    (1.0, 1.0, 0.5, Planet::Small, 0.12),
    (1.0, 1.0, 0.5, Planet::Large, 0.12),
    (1.0, 1.0, 13.0, Planet::Small, 0.013_676_725_699_868_126),
    (1.0, 1.0, 13.0, Planet::Large, 0.016_426_237_051_005_654),
    (1.0, 1.0, 100.0, Planet::Small, 0.003_509_762_302_514_094_5),
    (1.0, 1.0, 100.0, Planet::Large, 0.004_215_350_138_544_917),
    (3.5, 0.1, 0.5, Planet::Small, 0.12),
    (3.5, 0.1, 0.5, Planet::Large, 0.12),
    (3.5, 0.1, 13.0, Planet::Small, 0.103_129_643_028_462_68),
    (3.5, 0.1, 13.0, Planet::Large, 0.109_053_285_663_989_82),
    (3.5, 0.1, 100.0, Planet::Small, 0.026_465_437_804_058_9),
    (3.5, 0.1, 100.0, Planet::Large, 0.027_985_580_714_868_237),
    (3.5, 1.0, 0.5, Planet::Small, 0.12),
    (3.5, 1.0, 0.5, Planet::Large, 0.12),
    (3.5, 1.0, 13.0, Planet::Small, 0.047_868_539_949_538_436),
    (3.5, 1.0, 13.0, Planet::Large, 0.050_618_051_300_675_97),
    (3.5, 1.0, 100.0, Planet::Small, 0.012_284_168_058_799_33),
    (3.5, 1.0, 100.0, Planet::Large, 0.012_989_755_894_830_155),
];

#[test]
fn tls_t14_with_tls_constants() {
    for (r, m, p, planet, expected) in TLS_T14 {
        let ours = t14(TLS, r, m, p, planet);
        assert!(
            (ours / expected - 1.0).abs() < 1e-15,
            "{r} {m} {p} {planet:?}: {ours} against {expected}"
        );
        // With IAU values instead, the Sun's radius is 2.8e-4 larger and its
        // mass parameter 1e-4 smaller: a small planet's duration moves by
        // 3.1e-4, a large one's by less (Jupiter's radius is the same).
        if expected < 0.12 {
            let shift = t14(IAU, r, m, p, planet) / expected - 1.0;
            assert!((1e-4..3.2e-4).contains(&shift), "{shift}");
        }
    }
}

// transitleastsquares 1.32's `duration_grid` for (shortest period, longest
// period, step): its length, first, eighth and last durations.
#[test]
fn tls_duration_grid_with_tls_constants() {
    let cases = [
        (
            (0.5, 13.5, 1.1),
            38,
            [0.003_735_354_963_947_398_7, 0.007_279_150_092_814_182, 0.12],
        ),
        (
            (0.2, 300.0, 1.05),
            115,
            [
                0.000_472_578_058_963_541_05,
                0.000_664_964_786_505_669_1,
                0.12,
            ],
        ),
        (
            (5.0, 5.0, 1.1),
            29,
            [
                0.007_242_820_482_519_965,
                0.014_114_208_126_516_913,
                0.095_709_339_752_407_88,
            ],
        ),
    ];
    for ((short, long, step), len, [first, eighth, last]) in cases {
        let durations = grid(TLS, Stars::TLS, short, long, step);
        assert_eq!(durations.len(), len);
        for (ours, theirs) in [
            (durations[0], first),
            (durations[7], eighth),
            (durations[len - 1], last),
        ] {
            assert!((ours / theirs - 1.0).abs() < 1e-14, "{ours} {theirs}");
        }
    }
}
