// Frequency grids: checked inputs, and the automatic grid bit for bit
// against astropy 7.2's `LombScargle.autofrequency`.

use rustellar_ls::{AutoFrequency, Frequencies, LightCurve, LsError};

/// `n` uneven times from `offset`, about `scale` apart, increasing (the
/// recipe the astropy values below were made with).
fn curve(n: u32, scale: f64, offset: f64) -> LightCurve {
    let time: Vec<f64> = (0..n)
        .map(|i| offset + scale * (f64::from(i) + 0.1 * f64::from((i * i) % 7)))
        .collect();
    let len = time.len();
    LightCurve::from_days(time, vec![1.0; len], vec![1.0; len]).unwrap()
}

/// A light curve's recipe, the grid's settings, and astropy's grid.
type Case = (
    u32,
    f64,
    f64,
    f64,
    f64,
    Option<f64>,
    Option<f64>,
    usize,
    f64,
    f64,
);

#[test]
fn automatic_grid_as_astropy() {
    // (n, scale, offset, samples per peak, Nyquist factor, minimum,
    // maximum) and astropy's count, first and last frequency.
    #[rustfmt::skip]
    let cases: [Case; 12] = [
        (137, 0.02, 0.0, 5.0, 5.0, None, None, 1713, 0.036_710_719_530_102_79, 125.734_214_390_602_06),
        (137, 0.02, 0.0, 10.0, 2.0, None, None, 1371, 0.018_355_359_765_051_395, 50.312_041_116_005_88),
        (137, 0.02, 0.0, 5.0, 5.0, Some(0.3), Some(4.0), 51, 0.3, 3.971_071_953_010_279),
        (137, 0.02, 0.0, 1.5, 5.0, None, None, 514, 0.122_369_065_100_342_64, 125.673_029_858_051_89),
        (1000, 0.020_833_3, 1000.0, 5.0, 5.0, None, None, 12501, 0.004_802_889_413_660_489, 120.077_038_230_925_88),
        (1000, 0.020_833_3, 1000.0, 10.0, 2.0, None, None, 10001, 0.002_401_444_706_830_244_5, 48.031_295_581_311_72),
        (1000, 0.020_833_3, 1000.0, 5.0, 5.0, Some(0.3), Some(4.0), 386, 0.3, 3.998_224_848_518_576_3),
        (1000, 0.020_833_3, 1000.0, 1.5, 5.0, None, None, 3751, 0.016_009_631_378_868_298, 120.088_244_972_891_1),
        (5, 3.0, 2.5, 5.0, 5.0, None, None, 63, 0.007_936_507_936_507_936, 0.992_063_492_063_492),
        (5, 3.0, 2.5, 10.0, 2.0, None, None, 51, 0.003_968_253_968_253_968, 0.400_793_650_793_650_76),
        (5, 3.0, 2.5, 5.0, 5.0, Some(0.3), Some(4.0), 234, 0.3, 3.998_412_698_412_698),
        (5, 3.0, 2.5, 1.5, 5.0, None, None, 19, 0.026_455_026_455_026_454, 0.978_835_978_835_978_8),
    ];
    for (n, scale, offset, spp, nyquist, min, max, count, first, last) in cases {
        let mut auto = AutoFrequency::default()
            .with_samples_per_peak(spp)
            .with_nyquist_factor(nyquist);
        if let Some(f) = min {
            auto = auto.with_minimum(f);
        }
        if let Some(f) = max {
            auto = auto.with_maximum(f);
        }
        let grid = auto.frequencies(&curve(n, scale, offset)).unwrap();
        assert_eq!(grid.len(), count, "{n} {spp} {nyquist}");
        assert_eq!(grid.min(), first);
        assert_eq!(grid.get(0), first);
        assert_eq!(grid.max(), last);
        assert_eq!(grid.iter().last(), Some(last));
        assert!(grid.uniform_step().is_some());
        assert!(!grid.is_empty());
    }
}

#[test]
fn automatic_settings_and_errors() {
    let auto = AutoFrequency::default();
    assert_eq!((auto.samples_per_peak(), auto.nyquist_factor()), (5.0, 5.0));
    assert_eq!((auto.minimum(), auto.maximum()), (None, None));
    let set = auto.with_minimum(0.1).with_maximum(2.0);
    assert_eq!((set.minimum(), set.maximum()), (Some(0.1), Some(2.0)));

    let lc = curve(50, 0.1, 0.0);
    let one = LightCurve::from_days(vec![3.0], vec![1.0], vec![1.0]).unwrap();
    assert_eq!(auto.frequencies(&one), Err(LsError::NoBaseline));
    for (bad, err) in [
        (
            f64::NAN,
            LsError::NonFinite {
                name: "samples_per_peak",
            },
        ),
        (
            0.0,
            LsError::NotPositive {
                name: "samples_per_peak",
            },
        ),
    ] {
        assert_eq!(
            auto.with_samples_per_peak(bad).frequencies(&lc),
            Err(err),
            "{bad}"
        );
    }
    // A step too small or too large to hold.
    let long = LightCurve::from_days(vec![0.0, 1e300], vec![1.0; 2], vec![1.0; 2]).unwrap();
    assert_eq!(
        auto.with_samples_per_peak(1e308).frequencies(&long),
        Err(LsError::NotPositive { name: "step" })
    );
    let short = LightCurve::from_days(vec![0.0, 1e-320], vec![1.0; 2], vec![1.0; 2]).unwrap();
    assert_eq!(
        auto.frequencies(&short),
        Err(LsError::NonFinite { name: "step" })
    );
    assert_eq!(
        auto.with_nyquist_factor(-1.0).frequencies(&lc),
        Err(LsError::NotPositive {
            name: "nyquist_factor"
        })
    );
    assert_eq!(
        auto.with_minimum(0.0).frequencies(&lc),
        Err(LsError::NotPositive { name: "minimum" })
    );
    assert_eq!(
        auto.with_maximum(f64::INFINITY).frequencies(&lc),
        Err(LsError::NonFinite { name: "maximum" })
    );
    assert_eq!(
        auto.with_minimum(3.0).with_maximum(2.0).frequencies(&lc),
        Err(LsError::FrequencyRangeEmpty)
    );
    assert_eq!(
        auto.with_samples_per_peak(1e9).frequencies(&lc),
        Err(LsError::GridTooLarge)
    );
    // A Nyquist factor that is ignored need not be valid.
    assert!(
        auto.with_nyquist_factor(f64::NAN)
            .with_maximum(1.0)
            .frequencies(&lc)
            .is_ok()
    );
}

#[test]
fn uniform_grid() {
    let grid = Frequencies::uniform(0.5, 0.25, 4).unwrap();
    assert_eq!(grid.iter().collect::<Vec<_>>(), [0.5, 0.75, 1.0, 1.25]);
    assert_eq!((grid.min(), grid.max(), grid.len()), (0.5, 1.25, 4));
    // One frequency: any finite step, kept as 0.
    let one = Frequencies::uniform(2.0, -7.0, 1).unwrap();
    assert_eq!((one.uniform_step(), one.max()), (Some(0.0), 2.0));
    assert_eq!(
        Frequencies::uniform(2.0, f64::NAN, 1),
        Err(LsError::NonFinite { name: "step" })
    );
    for (start, step, count, err) in [
        (0.0, 1.0, 3, LsError::NotPositive { name: "frequency" }),
        (f64::NAN, 1.0, 3, LsError::NonFinite { name: "frequency" }),
        (1.0, 0.0, 3, LsError::NotPositive { name: "step" }),
        (1.0, f64::INFINITY, 3, LsError::NonFinite { name: "step" }),
        (1.0, 1e308, 3, LsError::NonFinite { name: "frequency" }),
        (1.0, 1.0, 0, LsError::NoFrequencies),
        (1.0, 1.0, 1_000_000_001, LsError::GridTooLarge),
    ] {
        assert_eq!(Frequencies::uniform(start, step, count), Err(err));
    }
}

#[test]
#[should_panic(expected = "frequency 4 of 4")]
fn uniform_grid_index_checked() {
    let _ = Frequencies::uniform(0.5, 0.25, 4).unwrap().get(4);
}

#[test]
fn listed_grid() {
    let grid = Frequencies::from_per_day(&[3.0, 1.0, 2.0]).unwrap();
    assert_eq!(grid.uniform_step(), None);
    assert_eq!(grid.len(), 3);
    assert_eq!((grid.min(), grid.max(), grid.get(1)), (1.0, 3.0, 1.0));
    assert_eq!(Frequencies::from_per_day(&[]), Err(LsError::NoFrequencies));
    assert_eq!(
        Frequencies::from_per_day(&[1.0, f64::INFINITY]),
        Err(LsError::NonFinite { name: "frequency" })
    );
    assert_eq!(
        Frequencies::from_per_day(&[1.0, -1.0]),
        Err(LsError::NotPositive { name: "frequency" })
    );
}

#[test]
fn error_messages() {
    for (err, text) in [
        (
            LsError::NonFinite { name: "frequency" },
            "frequency must be finite",
        ),
        (LsError::NoFrequencies, "at least one frequency is needed"),
        (
            LsError::NotPositive { name: "step" },
            "step must be greater than zero",
        ),
        (
            LsError::NoBaseline,
            "the light curve must span more than one time",
        ),
        (
            LsError::FrequencyRangeEmpty,
            "the minimum frequency must not be above the maximum frequency",
        ),
        (
            LsError::GridTooLarge,
            "the frequency grid would hold more than a billion frequencies",
        ),
    ] {
        assert_eq!(err.to_string(), text);
    }
}
