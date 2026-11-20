// Speed of each step of vetting, for the numbers in the README: a TESS
// sector at 2 minutes (19,440 points) and four years of Kepler long
// cadence (70,128 points), each with a planet in white noise.
//
// Run with `cargo bench -p rustellar-vet`.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use rustellar_model::{ScaledSemiMajorAxis, TransitParams};
use rustellar_vet::{
    Candidate, Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, Shape, TimeDelta,
    aliases, evidence, fit_depth, refine, secondary, signal_to_noise,
};

// A 3.7-day planet (k 0.05, b 0.5, a/R* 15) over `days` days every `step`
// days, with noise of 5e-4 from a fixed sequence.
fn case(days: f64, step: f64) -> (LightCurve, Candidate) {
    let at = |t: f64| Epoch::from_btjd(t).unwrap();
    let p = TransitParams::new(
        at(1.3),
        TimeDelta::from_days(3.7).unwrap(),
        RadiusRatio::new(0.05).unwrap(),
        ScaledSemiMajorAxis::new(15.0).unwrap(),
        ImpactParameter::new(0.5).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a count"
    )]
    let n = (days / step) as u32;
    let time: Vec<f64> = (0..n).map(|i| f64::from(i) * step).collect();
    let mut state = 1u64;
    let flux = time
        .iter()
        .map(|&t| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[expect(clippy::cast_precision_loss, reason = "a uniform draw")]
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            p.flux_at(at(t)) + 1e-3 * (u - 0.5)
        })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![5e-4; n as usize]).unwrap();
    let candidate = Candidate::new(p.period(), at(1.3), p.t14().unwrap()).unwrap();
    (lc, candidate)
}

fn vet(c: &mut Criterion) {
    let cases = [
        ("tess", case(27.0, 2.0 / 1440.0)),
        ("kepler", case(1461.0, 30.0 / 1440.0)),
    ];
    let mut group = c.benchmark_group("vet");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for (name, (lc, candidate)) in &cases {
        for (shape_name, shape) in [("limb", Shape::default()), ("box", Shape::BOX)] {
            let id = |step: &str| format!("{step}/{name}/{shape_name}");
            group.bench_function(id("fit_depth"), |b| {
                b.iter(|| fit_depth(black_box(lc), candidate, shape));
            });
            group.bench_function(id("refine"), |b| {
                b.iter(|| refine(black_box(lc), candidate, shape));
            });
            group.bench_function(id("evidence"), |b| {
                b.iter(|| evidence(black_box(lc), candidate, shape));
            });
            group.bench_function(id("signal_to_noise"), |b| {
                b.iter(|| signal_to_noise(black_box(lc), candidate, shape));
            });
            group.bench_function(id("aliases"), |b| {
                b.iter(|| aliases(black_box(lc), candidate, shape));
            });
        }
        group.bench_function(format!("secondary/{name}"), |b| {
            b.iter(|| secondary(black_box(lc), candidate));
        });
    }
    group.finish();
}

criterion_group!(benches, vet);
criterion_main!(benches);
