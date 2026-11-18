// The whole search: an injected transit recovered from Ofir's period grid,
// nothing to fit, the peaks it refines, threads, and against TLS on even
// and uneven light curves.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use rustellar_bls::OfirPeriod;
use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

use super::Random;
use crate::fit::{Points, Position, fit};
use crate::search::{Method, Options, peaks, run};

/// 27 days at 30 minutes, a transit of radius ratio `k` (from the model
/// crate, not the template), noise `noise`.
fn injected(random: &mut Random, k: f64, period: f64, t0: f64, noise: f64) -> LightCurve {
    let time: Vec<f64> = (0..1296).map(|i| f64::from(i) / 48.0).collect();
    let params = TransitParams::new(
        Epoch::from_btjd(t0).unwrap(),
        TimeDelta::from_days(period).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(12.0).unwrap(),
        ImpactParameter::new(0.2).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap();
    let epochs: Vec<Epoch> = time.iter().map(|&t| Epoch::from_btjd(t).unwrap()).collect();
    let flux: Vec<f64> = params
        .light_curve(&epochs)
        .into_iter()
        .map(|f| {
            let n: f64 = (0..4).map(|_| random.uniform() - 0.5).sum::<f64>() * 3_f64.sqrt();
            f + noise * n
        })
        .collect();
    LightCurve::from_btjd(time, flux, vec![noise; 1296]).unwrap()
}

fn periods(lc: &LightCurve) -> Vec<f64> {
    OfirPeriod::default()
        .periods(lc)
        .unwrap()
        .as_slice()
        .iter()
        .map(|p| p.as_days())
        .collect()
}

#[test]
fn an_injected_transit_is_recovered() {
    for method in [Method::Fold, Method::Cells] {
        recovered(method);
    }
}

fn recovered(method: Method) {
    let mut random = Random::new(51);
    for (k, period, t0) in [(0.04, 3.7, 1.3), (0.03, 5.9, 4.4), (0.05, 1.37, 0.2)] {
        let lc = injected(&mut random, k, period, t0, 4e-4);
        let grid = periods(&lc);
        let mut options = Options::tls();
        options.threads = NonZeroUsize::new(4).unwrap();
        options.method = method;
        let outcome = run(&lc, &grid, options);
        let best = outcome.best.unwrap();
        let spectra = outcome.spectra.unwrap();
        let p = best.position;
        assert!((p.period / period - 1.0).abs() < 1e-3, "{p:?}");
        let off = (p.t0 - t0).rem_euclid(period);
        assert!(off.min(period - off) < 0.01, "{p:?}");
        assert!(spectra.sde > 9.0, "{}", spectra.sde);
        assert!(outcome.period_uncertainty > 0.0 && outcome.period_uncertainty < 0.05 * period);
        assert_eq!(outcome.periods.len(), grid.len());
        // The model's depth at mid-transit, 1 - F(b), near k² (limb
        // darkening deepens the middle).
        assert!((best.fit.depth / (k * k) - 1.0).abs() < 0.4, "{best:?}");
        // The truth within a few standard errors in period and T0 (duration
        // and depth are the template's, not the planet's), every transit in
        // the 27 days found with a depth, odd and even alike.
        let errors = outcome.errors.unwrap();
        assert!((p.period - period).abs() < 4.0 * errors.period, "{p:?} {errors:?}");
        let off = (p.t0 - t0) - ((p.t0 - t0) / period).round() * period;
        assert!(off.abs() < 4.0 * errors.t0, "{p:?} {errors:?}");
        assert!(errors.depth < 0.2 * best.fit.depth && errors.duration < 0.2 * p.duration);
        #[expect(clippy::cast_possible_truncation, reason = "a few transits")]
        let expected = (27.0 / period) as usize;
        assert!(outcome.transits.len() >= expected, "{}", outcome.transits.len());
        for t in &outcome.transits {
            assert!((t.depth - best.fit.depth).abs() < 4.0 * t.error, "{t:?}");
        }
        assert!(outcome.odd_even.unwrap().sigma < 3.0, "{:?}", outcome.odd_even);
    }
}

#[test]
fn nothing_to_fit_gives_no_spectra() {
    // Every flux above 1: no depth anywhere.
    let time: Vec<f64> = (0..500).map(|i| f64::from(i) / 20.0).collect();
    let lc = LightCurve::from_btjd(time, vec![1.001; 500], vec![1e-3; 500]).unwrap();
    for method in [Method::Fold, Method::Cells] {
        let mut options = Options::tls();
        options.method = method;
        let outcome = run(&lc, &[1.0, 1.5, 2.0, 3.0], options);
        assert!(outcome.spectra.is_none() && outcome.best.is_none());
        assert!(outcome.errors.is_none() && outcome.transits.is_empty());
        assert!(outcome.odd_even.is_none());
        assert_eq!(outcome.period_uncertainty, f64::INFINITY);
        assert!(outcome.periods.iter().all(|p| p.depth == 0.0));
    }
}

#[test]
fn one_period_is_refined_within_rounding_of_it() {
    let mut random = Random::new(52);
    let lc = injected(&mut random, 0.05, 2.5, 0.7, 3e-4);
    let mut options = Options::tls();
    options.min_depth = 0.0;
    // The peak at the first period and at the last: refined within the
    // neighbouring period, and to the true one.
    for grid in [[2.5, 2.6], [2.4, 2.5]] {
        let best = run(&lc, &grid, options).best.unwrap();
        assert!((best.position.period - 2.5).abs() < 1e-3, "{best:?}");
    }
}

#[test]
fn the_peaks_are_local_maxima_highest_first() {
    let power = [3.0, 1.0, 5.0, 5.0, 2.0, 7.0, 0.0, 4.0];
    // A plateau's peak is its first value.
    assert_eq!(peaks(&power, 10), vec![5, 2, 7, 0]);
    assert_eq!(peaks(&power, 2), vec![5, 2]);
    assert_eq!(peaks(&[1.0], 3), vec![0]);
    assert!(peaks(&power, 0).is_empty());
}

#[test]
fn the_same_outcome_on_any_number_of_threads() {
    let mut random = Random::new(53);
    let lc = injected(&mut random, 0.04, 3.1, 0.9, 5e-4);
    let grid: Vec<f64> = periods(&lc).into_iter().step_by(7).collect();
    for method in [Method::Fold, Method::Cells] {
        let mut options = Options::tls();
        options.method = method;
        let base = run(&lc, &grid, options);
        for threads in [2, 5] {
            options.threads = NonZeroUsize::new(threads).unwrap();
            assert_eq!(run(&lc, &grid, options), base, "{method:?} {threads}");
        }
    }
}

// transitleastsquares 1.32 (one thread, defaults) on injected transits:
// seed, k, period, t0, a gap (days 12.5 to 14.5 dropped); then TLS's
// period, T0, depth, duration and SDE. The light curves are rebuilt here
// from the same recipe: SplitMix64 noise to the bit, the transit from
// this workspace's model rather than batman (within 2e-8).
type Injection = (u64, f64, f64, f64, bool, f64, f64, f64, f64, f64);
const TLS_INJECTED: [Injection; 8] = [
    (
        101,
        0.04,
        3.7,
        1.3,
        false,
        3.701_610_802_206_391_6,
        1.293_798_408_280_662_6,
        0.001_916_942_757_633_877_1,
        0.106_566_429_672_840_98,
        17.071_958_767_588_765,
    ),
    (
        102,
        0.03,
        5.9,
        4.4,
        false,
        5.914_658_798_951_51,
        4.381_408_676_609_794,
        0.001_078_095_083_206_109_8,
        0.186_491_251_927_471_7,
        14.540_919_723_648_262,
    ),
    (
        103,
        0.05,
        1.37,
        0.2,
        false,
        1.369_678_181_420_841,
        0.203_951_050_014_790_44,
        0.002_808_350_734_743_503_3,
        0.041_028_075_424_043_776,
        28.109_399_702_335_356,
    ),
    (
        104,
        0.025,
        2.3,
        0.9,
        false,
        2.301_553_733_093_640_6,
        0.890_585_864_618_333_7,
        0.000_704_877_242_543_866_5,
        0.075_218_138_277_413_62,
        15.451_564_369_384_231,
    ),
    (
        105,
        0.035,
        7.1,
        6.0,
        false,
        7.102_480_035_199_931,
        5.987_136_114_061_559,
        0.001_522_504_439_785_632_4,
        0.205_500_002_123_935_77,
        14.561_275_297_617_32,
    ),
    (
        106,
        0.04,
        3.7,
        1.3,
        true,
        3.696_262_296_431_111,
        1.309_438_537_593_720_5,
        0.002_047_479_423_740_617,
        0.098_742_773_707_677_74,
        17.038_611_329_013_42,
    ),
    (
        107,
        0.03,
        4.6,
        2.2,
        true,
        4.600_537_366_479_616,
        2.204_074_168_635_169_3,
        0.001_082_373_221_273_802_4,
        0.126_719_892_924_853_1,
        13.932_328_263_781_725,
    ),
    (
        108,
        0.045,
        1.9,
        0.5,
        true,
        1.900_854_605_799_152_4,
        0.496_443_326_444_047_04,
        0.002_660_224_533_169_608_4,
        0.044_883_078_958_035_33,
        22.486_869_071_833_63,
    ),
];

// The default method only: the fold's agrees (on 300 such curves the two
// are as close to the truth as each other).
#[test]
fn against_tls_on_injected_transits() {
    let method = Options::tls().method;
    let noise = 4e-4;
    // Sums of the errors, ours and TLS's: T0 in minutes, depth and
    // duration relative to the injected transit's (1 - F(b) and T14).
    let (mut t0_error, mut depth_error, mut duration_error) = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0));
    let mut closer_t0 = 0;
    for (seed, k, period, t0, gap, tls_p, tls_t0, tls_depth, tls_duration, tls_sde) in TLS_INJECTED
    {
        let mut random = Random::new(seed);
        let params = TransitParams::new(
            Epoch::from_btjd(t0).unwrap(),
            TimeDelta::from_days(period).unwrap(),
            RadiusRatio::new(k).unwrap(),
            ScaledSemiMajorAxis::new(12.0).unwrap(),
            ImpactParameter::new(0.2).unwrap(),
            LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
        )
        .unwrap();
        let (mut time, mut flux) = (vec![], vec![]);
        for i in 0..1296 {
            let t = f64::from(i) / 48.0;
            let f = params.flux_at(Epoch::from_btjd(t).unwrap());
            let n: f64 = (0..4).map(|_| random.uniform() - 0.5).sum::<f64>() * 3_f64.sqrt();
            if !(gap && t > 12.5 && t < 14.5) {
                time.push(t);
                flux.push(f + noise * n);
            }
        }
        let n = time.len();
        let lc = LightCurve::from_btjd(time, flux, vec![noise; n]).unwrap();
        let mut options = Options::tls();
        options.threads = NonZeroUsize::new(4).unwrap();
        let outcome = run(&lc, &periods(&lc), options);
        let best = outcome.best.unwrap();
        let ours = best.position;
        let depth = 1.0 - params.flux_at(Epoch::from_btjd(t0).unwrap());
        let t14 = params.t14().unwrap().as_days();
        let off = |x: f64| {
            let d = (x - t0).rem_euclid(period);
            d.min(period - d) * 1440.0
        };
        let ours_sde = outcome.spectra.unwrap().sde;
        let points = Points::new(&lc);
        let tls = Position {
            period: tls_p,
            duration: tls_duration,
            t0: tls_t0,
        };
        assert!(
            best.fit.chi2 < fit(&points, &options.template, tls).chi2,
            "{method:?} {seed}"
        );
        // Both find the period.
        assert!(
            (ours.period / period - 1.0).abs() < 3e-3,
            "{method:?} {seed}: {ours:?}"
        );
        assert!((tls_p / period - 1.0).abs() < 3e-3, "{method:?} {seed}");
        // The SDE about the same: the same signal, the same noise.
        assert!(
            (ours_sde / tls_sde - 1.0).abs() < 0.1,
            "{method:?} {seed}: {ours_sde} {tls_sde}"
        );
        for (total, ours, theirs) in [
            (&mut t0_error, off(ours.t0), off(tls_t0)),
            (
                &mut depth_error,
                (best.fit.depth / depth - 1.0).abs(),
                (tls_depth / depth - 1.0).abs(),
            ),
            (
                &mut duration_error,
                (ours.duration / t14 - 1.0).abs(),
                (tls_duration / t14 - 1.0).abs(),
            ),
        ] {
            total.0 += ours;
            total.1 += theirs;
        }
        if off(ours.t0) < off(tls_t0) {
            closer_t0 += 1;
        }
    }
    // On these 8 (3 with a gap): T0 40 against 98 minutes in all, closer
    // on 7; depth 26% against 36%; duration 45% against 81%.
    assert!(closer_t0 >= 6, "{method:?} {closer_t0}");
    assert!(t0_error.0 < 0.5 * t0_error.1, "{method:?} {t0_error:?}");
    assert!(depth_error.0 < depth_error.1, "{method:?} {depth_error:?}");
    assert!(
        duration_error.0 < 0.7 * duration_error.1,
        "{method:?} {duration_error:?}"
    );
}

// transitleastsquares 1.32 (one thread, defaults) on shallower, noisier
// transits, some sparse: seed, k, period, t0, noise, sparse (after each
// point's noise one more uniform, the point dropped below 0.25); then TLS's
// period, T0, depth, duration and SDE.
type Harder = (u64, f64, f64, f64, f64, bool, f64, f64, f64, f64, f64);
const TLS_HARDER: [Harder; 6] = [
    (
        201,
        0.02,
        3.3,
        1.1,
        0.0004,
        false,
        3.299_750_241_527_508_6,
        1.094_399_679_301_711_2,
        0.000_505_189_873_590_494_7,
        0.093_245_625_963_735_85,
        7.596_524_863_667_848,
    ),
    (
        202,
        0.025,
        6.4,
        2.0,
        0.0006,
        false,
        6.388_527_974_574_47,
        2.015_396_935_905_27,
        0.000_938_272_770_568_016_1,
        0.169_537_501_752_247_02,
        10.307_934_386_581_776,
    ),
    (
        203,
        0.03,
        2.7,
        0.4,
        0.0008,
        false,
        2.700_133_330_225_079,
        0.396_431_478_186_899_1,
        0.000_956_704_748_115_266_8,
        0.082_056_150_848_087_55,
        9.427_302_688_224_088,
    ),
    (
        204,
        0.03,
        4.1,
        3.0,
        0.0004,
        true,
        4.098_806_425_296_886,
        3.006_653_056_008_470_3,
        0.001_114_310_376_579_497,
        0.069_514_881_595_413_9,
        13.076_079_417_170_307,
    ),
    (
        205,
        0.02,
        1.6,
        0.7,
        0.0003,
        true,
        1.600_212_872_769_931_8,
        0.699_509_894_757_587_2,
        0.000_471_823_558_755_746_2,
        0.032_621_980_971_767_16,
        12.368_871_555_982_349,
    ),
    (
        206,
        0.04,
        8.3,
        5.1,
        0.0008,
        true,
        8.284_089_067_560_107,
        5.112_646_040_464_226,
        0.001_464_937_354_960_671_5,
        0.164_643_646_656_991_8,
        9.645_970_750_682_729,
    ),
];

// The default method only: the fold's agrees (on 300 such curves the two
// are as close to the truth as each other).
#[test]
fn against_tls_on_shallower_noisier_and_sparse_transits() {
    let method = Options::tls().method;
    let template = Options::tls().template;
    let (mut depth_error, mut duration_error) = ((0.0, 0.0), (0.0, 0.0));
    for (seed, k, period, t0, noise, sparse, tls_p, tls_t0, tls_depth, tls_duration, tls_sde) in
        TLS_HARDER
    {
        let mut random = Random::new(seed);
        let params = TransitParams::new(
            Epoch::from_btjd(t0).unwrap(),
            TimeDelta::from_days(period).unwrap(),
            RadiusRatio::new(k).unwrap(),
            ScaledSemiMajorAxis::new(12.0).unwrap(),
            ImpactParameter::new(0.2).unwrap(),
            LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
        )
        .unwrap();
        let (mut time, mut flux) = (vec![], vec![]);
        for i in 0..1296 {
            let t = f64::from(i) / 48.0;
            let f = params.flux_at(Epoch::from_btjd(t).unwrap());
            let n: f64 = (0..4).map(|_| random.uniform() - 0.5).sum::<f64>() * 3_f64.sqrt();
            if sparse && random.uniform() < 0.25 {
                continue;
            }
            time.push(t);
            flux.push(f + noise * n);
        }
        let n = time.len();
        let lc = LightCurve::from_btjd(time, flux, vec![noise; n]).unwrap();
        let mut options = Options::tls();
        options.threads = NonZeroUsize::new(4).unwrap();
        let outcome = run(&lc, &periods(&lc), options);
        let best = outcome.best.unwrap();
        let ours = best.position;
        let depth = 1.0 - params.flux_at(Epoch::from_btjd(t0).unwrap());
        let t14 = params.t14().unwrap().as_days();
        let ours_sde = outcome.spectra.unwrap().sde;
        assert!(
            (ours.period / period - 1.0).abs() < 5e-3,
            "{method:?} {seed}: {ours:?}"
        );
        assert!((tls_p / period - 1.0).abs() < 5e-3, "{method:?} {seed}");
        assert!(
            (ours_sde / tls_sde - 1.0).abs() < 0.1,
            "{method:?} {seed}: {ours_sde} {tls_sde}"
        );
        // The least chi-squared of ours, the truth's and TLS's ephemeris
        // and duration (each with this template and its least-squares
        // depth): at this signal to noise the least is not the truth.
        let points = Points::new(&lc);
        let chi2 = |period, duration, t0| {
            fit(
                &points,
                &template,
                Position {
                    period,
                    duration,
                    t0,
                },
            )
            .chi2
        };
        assert!(best.fit.chi2 < chi2(period, t14, t0), "{method:?} {seed}");
        assert!(
            best.fit.chi2 < chi2(tls_p, tls_duration, tls_t0),
            "{method:?} {seed}"
        );
        for (total, ours, theirs) in [
            (
                &mut depth_error,
                (best.fit.depth / depth - 1.0).abs(),
                (tls_depth / depth - 1.0).abs(),
            ),
            (
                &mut duration_error,
                (ours.duration / t14 - 1.0).abs(),
                (tls_duration / t14 - 1.0).abs(),
            ),
        ] {
            total.0 += ours;
            total.1 += theirs;
        }
    }
    // On these 6: depth 69% against 69%, duration 56% against 107%. T0 is
    // not compared: TLS's is nearer the truth on 4, by noise, not fit.
    assert!(
        depth_error.0 < 1.05 * depth_error.1,
        "{method:?} {depth_error:?}"
    );
    assert!(
        duration_error.0 < 0.7 * duration_error.1,
        "{method:?} {duration_error:?}"
    );
}

// transitleastsquares 1.32 (one thread, defaults, given the errors) on
// uneven light curves: seed, k, period, t0, noise; then TLS's period, T0,
// depth, duration and SDE. Mixed cadence: 13.5 days at 30 minutes, then
// from day 14 at 10 minutes with errors sqrt(3) larger (the same precision
// per unit time). Varying errors: 27 days at 30 minutes, errors
// noise (2 - cos(2 pi t / 13.5)), from 1 to 3 times noise. Each the first
// six of 200 that both recover.
type Uneven = (u64, f64, f64, f64, f64, f64, f64, f64, f64, f64);
const TLS_MIXED: [Uneven; 6] = [
    (
        2000,
        0.033_382_301_215_446_62,
        8.455_420_933_866_634,
        1.274_399_988_520_075,
        0.000_493_123_845_350_846_8,
        8.456_952_671_664_641,
        1.275_976_730_251_069_8,
        0.001_154_387_901_870_945_8,
        0.177_043_616_726_419_25,
        13.068_719_479_980_697,
    ),
    (
        2001,
        0.026_997_899_181_132_094,
        7.871_199_674_220_626,
        1.155_930_952_334_575,
        0.000_544_906_181_756_190_1,
        7.873_473_061_001_296,
        1.149_635_467_972_518_3,
        0.000_882_811_960_081_331_1,
        0.146_317_038_616_875_4,
        11.941_637_580_322_968,
    ),
    (
        2002,
        0.019_112_668_858_696_84,
        1.359_375_866_663_073_8,
        0.260_175_632_927_864_66,
        0.000_313_411_108_501_357,
        1.359_811_934_951_399_3,
        0.254_669_142_639_908_36,
        0.000_423_402_392_230_931_57,
        0.024_184_634_482_128_157,
        13.238_822_291_474_012,
    ),
    (
        2004,
        0.029_662_447_268_763_79,
        1.165_712_143_317_366,
        0.213_222_482_173_234_25,
        0.000_754_558_456_747_037_3,
        1.166_455_097_420_633_4,
        0.205_205_505_205_220_57,
        0.001_058_840_647_091_385_8,
        0.017_380_261_934_695_05,
        11.920_497_393_980_181,
    ),
    (
        2005,
        0.042_696_445_229_643_76,
        1.943_954_024_423_093_5,
        1.437_852_093_019_428_7,
        0.000_584_306_679_273_314_1,
        1.942_991_845_604_620_2,
        1.447_597_401_054_914_9,
        0.001_961_913_512_111_168_6,
        0.038_004_425_614_772_824,
        17.482_689_078_304_8,
    ),
    (
        2006,
        0.039_899_741_665_330_46,
        8.010_719_784_298_303,
        4.399_328_689_938_431,
        0.000_350_144_481_532_505_1,
        8.006_692_134_580_83,
        4.405_753_291_072_397,
        0.001_949_250_987_648_976,
        0.121_135_158_938_783_65,
        14.715_895_152_455_127,
    ),
];
const TLS_VARYING: [Uneven; 6] = [
    (
        3000,
        0.027_503_610_192_387_635,
        5.655_211_655_651_549_5,
        3.818_851_061_375_670_3,
        0.000_378_819_428_986_153_5,
        5.652_650_674_234_365,
        3.821_301_715_546_546,
        0.000_996_154_929_136_849_9,
        0.149_193_001_541_977_37,
        10.848_661_882_915_449,
    ),
    (
        3001,
        0.024_477_521_934_854_91,
        8.236_476_229_257_64,
        4.761_278_840_496_468,
        0.000_398_442_120_757_876_94,
        8.276_739_848_590_491,
        4.722_072_737_285_428,
        0.000_807_360_740_791_240_9,
        0.226_050_002_336_329_35,
        6.634_857_295_068_678,
    ),
    (
        3004,
        0.033_111_756_008_141_91,
        2.864_108_800_775_504_4,
        1.389_401_247_985_954_3,
        0.000_692_489_700_002_986,
        2.864_713_104_478_716_4,
        1.386_774_589_487_404_4,
        0.001_074_175_612_460_592_8,
        0.091_173_500_942_319_52,
        6.524_286_199_278_212,
    ),
    (
        3005,
        0.037_634_949_089_691_29,
        2.224_761_121_484_644_6,
        1.268_484_400_978_510_8,
        0.000_404_474_815_922_269_57,
        2.223_800_509_664_458_4,
        1.277_091_581_064_955_5,
        0.001_679_395_242_268_233_4,
        0.068_380_125_706_739_64,
        17.174_100_236_484_563,
    ),
    (
        3006,
        0.032_791_233_060_799_26,
        1.291_104_375_011_761,
        0.926_217_250_378_607_2,
        0.000_346_549_368_800_090_7,
        1.290_395_877_795_182_3,
        0.932_287_608_684_799_1,
        0.001_327_515_982_200_933,
        0.039_074_357_546_708_356,
        19.611_510_873_464_837,
    ),
    (
        3007,
        0.032_515_134_773_630_175,
        4.295_134_024_739_159,
        0.718_801_538_675_302_2,
        0.000_506_751_475_125_066_2,
        4.286_549_713_228_742,
        0.736_362_497_210_774,
        0.001_458_522_470_884_404_7,
        0.117_223_072_640_125_1,
        10.992_179_264_198_116,
    ),
];

/// A light curve of the uneven sets, and its transit.
fn uneven(
    mixed: bool,
    seed: u64,
    k: f64,
    period: f64,
    t0: f64,
    noise: f64,
) -> (LightCurve, TransitParams) {
    let (time, dy): (Vec<f64>, Vec<f64>) = if mixed {
        (0..648)
            .map(|i| (f64::from(i) / 48.0, noise))
            .chain((0..1872).map(|i| (14.0 + f64::from(i) / 144.0, noise * 3_f64.sqrt())))
            .unzip()
    } else {
        (0..1296)
            .map(|i| {
                let t = f64::from(i) / 48.0;
                (t, noise * (2.0 - (2.0 * PI * t / 13.5).cos()))
            })
            .unzip()
    };
    let params = TransitParams::new(
        Epoch::from_btjd(t0).unwrap(),
        TimeDelta::from_days(period).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(12.0).unwrap(),
        ImpactParameter::new(0.2).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap();
    let mut random = Random::new(seed);
    let flux: Vec<f64> = time
        .iter()
        .zip(&dy)
        .map(|(&t, &e)| {
            let n: f64 = (0..4).map(|_| random.uniform() - 0.5).sum::<f64>() * 3_f64.sqrt();
            params.flux_at(Epoch::from_btjd(t).unwrap()) + e * n
        })
        .collect();
    (LightCurve::from_btjd(time, flux, dy).unwrap(), params)
}

// Where TLS's own assumptions break: with mixed cadence it takes the
// sampling as even, and with varying errors its depth is an unweighted
// mean. On 200 of each: mixed, durations closer on 151 of 160 (median 4.8%
// against 34%), timing on 109; varying, depths closer on 77 of 124 (median
// 8.3% against 10.3%), timing on 79. Here: recovered, fitted better than at
// TLS's position, and the totals.
#[test]
fn against_tls_on_mixed_cadence_and_varying_errors() {
    let template = Options::tls().template;
    for (mixed, cases) in [(true, TLS_MIXED), (false, TLS_VARYING)] {
        let (mut depth_error, mut duration_error) = ((0.0, 0.0), (0.0, 0.0));
        for (seed, k, period, t0, noise, tls_p, tls_t0, tls_depth, tls_duration, _) in cases {
            let (lc, params) = uneven(mixed, seed, k, period, t0, noise);
            let mut options = Options::tls();
            options.threads = NonZeroUsize::new(4).unwrap();
            let best = run(&lc, &periods(&lc), options).best.unwrap();
            let ours = best.position;
            assert!(
                (ours.period / period - 1.0).abs() < 1e-2,
                "{seed}: {ours:?}"
            );
            let points = Points::new(&lc);
            let at_tls = fit(
                &points,
                &template,
                Position {
                    period: tls_p,
                    duration: tls_duration,
                    t0: tls_t0,
                },
            );
            assert!(best.fit.chi2 < at_tls.chi2, "{seed}");
            let depth = 1.0 - params.flux_at(Epoch::from_btjd(t0).unwrap());
            let t14 = params.t14().unwrap().as_days();
            depth_error.0 += (best.fit.depth / depth - 1.0).abs();
            depth_error.1 += (tls_depth / depth - 1.0).abs();
            duration_error.0 += (ours.duration / t14 - 1.0).abs();
            duration_error.1 += (tls_duration / t14 - 1.0).abs();
        }
        // Mixed: duration 39% against 203%, depth 24% against 36%. Varying:
        // depth 33% against 54%, duration 20% against 50%.
        let (depth_bound, duration_bound) = if mixed { (0.8, 0.3) } else { (0.8, 0.6) };
        assert!(
            depth_error.0 < depth_bound * depth_error.1,
            "{mixed} {depth_error:?}"
        );
        assert!(
            duration_error.0 < duration_bound * duration_error.1,
            "{mixed} {duration_error:?}"
        );
    }
}
