// Speed of the transit model, for the numbers in the README. The planet,
// times and sample counts match the batman and PyTransit timings recorded
// there, so the comparison is like for like.
//
// Run with `cargo bench -p rustellar-model`.

use std::{hint::black_box, num::NonZeroUsize};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitModel,
    TransitParams, flux,
};

// A hot Jupiter: P = 3 d, k = 0.1, a_rs = 10, b = 0.3, Sun-like limb
// darkening. Its transit lasts 2.4 hours.
fn hot_jupiter() -> TransitParams {
    TransitParams::new(
        Epoch::from_btjd(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

// `n` times evenly spread over +-0.1 days (4.8 hours) around mid-transit, so
// about half are in transit.
fn times(n: usize) -> Vec<Epoch> {
    (0..n)
        .map(|i| Epoch::from_btjd(-0.1 + 0.2 * i as f64 / (n - 1) as f64).unwrap())
        .collect()
}

// One call of the analytic flux in each region of the formula.
fn single_flux(c: &mut Criterion) {
    let ld = LimbDarkening::quadratic(0.4, 0.25).unwrap();
    let planet = RadiusRatio::new(0.1).unwrap();
    let binary = RadiusRatio::new(2.0).unwrap();
    let cases = [
        ("out of transit", 1.5, planet),
        ("full transit", 0.5, planet),
        ("ingress", 1.0, planet),
        ("larger than the star", 1.5, binary),
    ];
    let mut group = c.benchmark_group("flux");
    for (name, z, k) in cases {
        group.bench_function(name, |b| b.iter(|| flux(black_box(z), k, ld)));
    }
    group.finish();
}

fn light_curve(c: &mut Criterion) {
    let params = hot_jupiter();
    let mut group = c.benchmark_group("light curve");
    for n in [100, 1_000, 10_000, 100_000] {
        let times = times(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("TransitParams", n), &times, |b, t| {
            b.iter(|| params.light_curve(black_box(t)))
        });
        let mut model = TransitModel::new(times.clone());
        group.bench_function(BenchmarkId::new("TransitModel", n), |b| {
            b.iter(|| model.light_curve(black_box(params)).len())
        });
    }
    group.finish();
}

// Kepler long cadence (29.4 minutes) with the sample counts in common use.
fn exposure(c: &mut Criterion) {
    let params = hot_jupiter();
    let n = 1_000;
    let mut group = c.benchmark_group("long cadence, 1000 times");
    group.throughput(Throughput::Elements(n as u64));
    for samples in [1, 5, 15, 30] {
        let exposure = Exposure::new(
            TimeDelta::from_hours(29.4 / 60.0).unwrap(),
            NonZeroUsize::new(samples).unwrap(),
        )
        .unwrap();
        let mut model = TransitModel::with_exposure(times(n), exposure);
        group.bench_function(BenchmarkId::new("samples", samples), |b| {
            b.iter(|| model.light_curve(black_box(params)).len())
        });
    }
    group.finish();
}

criterion_group!(benches, single_flux, light_curve, exposure);
criterion_main!(benches);
