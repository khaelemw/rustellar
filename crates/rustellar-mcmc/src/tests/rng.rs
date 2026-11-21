// The generators against their authors' reference outputs, and the uniform
// numbers' range and moments.

use crate::rng::{Rng, splitmix64};

#[test]
fn splitmix64_reference() {
    // Vigna's splitmix64.c from 1234567.
    let mut state = 1_234_567;
    let want = [
        6_457_827_717_110_365_317,
        3_203_168_211_198_807_973,
        9_817_491_932_198_370_423,
        4_593_380_528_125_082_431,
        16_408_922_859_458_223_821,
    ];
    for w in want {
        assert_eq!(splitmix64(&mut state), w);
    }
}

#[test]
fn xoshiro256starstar_reference() {
    // xoshiro256starstar.c from the state 1, 2, 3, 4.
    let mut rng = Rng::from_state([1, 2, 3, 4]);
    let want = [
        11_520,
        0,
        1_509_978_240,
        1_215_971_899_390_074_240,
        1_216_172_134_540_287_360,
        607_988_272_756_665_600,
        16_172_922_978_634_559_625,
        8_476_171_486_693_032_832,
        10_595_114_339_597_558_777,
        2_904_607_092_377_533_576,
    ];
    for w in want {
        assert_eq!(rng.next_u64(), w);
    }
}

#[test]
fn seeded_and_uniform() {
    assert_eq!(Rng::new(5), Rng::new(5));
    assert_ne!(Rng::new(5), Rng::new(6));
    let mut rng = Rng::new(2026);
    let n = 100_000;
    let draws: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
    assert!(draws.iter().all(|&u| (0.0..1.0).contains(&u)));
    let mean = draws.iter().sum::<f64>() / f64::from(n);
    let var = draws.iter().map(|u| (u - mean).powi(2)).sum::<f64>() / f64::from(n);
    // Within four standard errors of 1/2 and 1/12.
    assert!((mean - 0.5).abs() < 4.0 * (1.0 / 12.0 / f64::from(n)).sqrt());
    assert!((var - 1.0 / 12.0).abs() < 4.0 * (1.0 / 180.0 / f64::from(n)).sqrt());
}

#[test]
fn normals_and_open_units() {
    let mut rng = Rng::new(12);
    let n = 200_000;
    let (mut m1, mut m2, mut m4) = (0.0, 0.0, 0.0);
    for _ in 0..n {
        let z = rng.normal();
        m1 += z;
        m2 += z * z;
        m4 += z * z * z * z;
    }
    let n = f64::from(n);
    assert!((m1 / n).abs() < 0.01);
    assert!((m2 / n - 1.0).abs() < 0.01);
    assert!((m4 / n - 3.0).abs() < 0.06);
    for _ in 0..1000 {
        let u = rng.open_unit();
        assert!(u > 0.0 && u < 1.0);
    }
    // The extremes: the midpoints of the first and last steps.
    assert_eq!(Rng::midpoint(0), f64::EPSILON / 2.0);
    assert_eq!(Rng::midpoint(u64::MAX), 1.0 - f64::EPSILON / 2.0);
}
