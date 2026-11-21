// The fit on several threads: the same to the bit as on one.

use std::num::NonZeroUsize;

use rustellar_fit::{
    Epoch, Exposure, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
    ScaledSemiMajorAxis, TimeDelta, TransitFit, TransitParams,
};

fn params(t0: f64, k: f64, a: f64, b: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(t0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(a).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

/// `n` points every two minutes, with a deterministic wobble as noise.
fn curve(truth: TransitParams, n: u32) -> LightCurve {
    let time: Vec<f64> = (0..n).map(|i| f64::from(i) * 2.0 / 1440.0).collect();
    let flux = time
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let wobble = 1e-3 * ((i as f64) * 0.754_877_666_246_692_7).fract() - 5e-4;
            truth.flux_at(Epoch::from_days(t).unwrap()) + wobble
        })
        .collect();
    LightCurve::from_days(time, flux, vec![3e-4; n as usize]).unwrap()
}

fn same(fit: &TransitFit<'_>) {
    let one = fit.run().unwrap();
    for threads in [2, 3, 7] {
        let many = fit
            .clone()
            .with_threads(NonZeroUsize::new(threads).unwrap())
            .run()
            .unwrap();
        for p in Parameter::ALL {
            assert_eq!(one.estimate(p), many.estimate(p), "{p}, {threads} threads");
        }
        assert_eq!(one.chi_squared(), many.chi_squared());
        assert_eq!(one.iterations(), many.iterations());
    }
}

#[test]
fn threads_change_nothing() {
    let truth = params(1.3, 0.1, 10.0, 0.3);
    let lc = curve(truth, 20_000);
    let start = params(1.302, 0.105, 9.0, 0.4);
    let fit = TransitFit::new(&lc, start);
    assert_eq!(fit.threads(), NonZeroUsize::MIN);
    same(&fit);
    // Over exposures.
    let exposure = Exposure::new(
        TimeDelta::from_days(10.0 / 1440.0).unwrap(),
        NonZeroUsize::new(5).unwrap(),
    )
    .unwrap();
    same(&TransitFit::new(&lc, start).with_exposure(exposure));
    // The baseline held.
    same(&TransitFit::new(&lc, start).with_fixed(Parameter::Baseline));
}

#[test]
fn more_threads_than_blocks() {
    // 300 points: two blocks, so most of seven threads have none.
    let truth = params(0.2, 0.1, 5.0, 0.3);
    let lc = curve(truth, 300);
    same(&TransitFit::new(&lc, params(0.201, 0.11, 5.0, 0.3)));
}
