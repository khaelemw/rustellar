// The whole search: an injected transit recovered from Ofir's period grid,
// nothing to fit, the peaks it refines, and threads.

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
