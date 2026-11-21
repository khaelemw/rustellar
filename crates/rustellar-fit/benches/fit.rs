// Speed of the fit, for the numbers in the README: the reference test's
// light curves and starts, so batman and PyTransit with scipy can be timed
// on the same fits.
//
// Run with `cargo bench -p rustellar-fit`.

use std::{hint::black_box, num::NonZeroUsize};

use criterion::{Criterion, criterion_group, criterion_main};
use rustellar_fit::{
    Epoch, Exposure, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
    ScaledSemiMajorAxis, TimeDelta, TransitFit, TransitParams,
};

const CASES: &str = include_str!("../tests/reference/fit_cases.csv");
const POINTS: &str = include_str!("../tests/reference/fit_points.csv");

fn params(v: &[f64]) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(v[0]).unwrap(),
        TimeDelta::from_days(v[1]).unwrap(),
        RadiusRatio::new(v[2]).unwrap(),
        ScaledSemiMajorAxis::new(v[3]).unwrap(),
        ImpactParameter::new(v[4]).unwrap(),
        LimbDarkening::quadratic(v[5], v[6]).unwrap(),
    )
    .unwrap()
}

fn fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit");
    for line in CASES.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let name = f[0];
        let sigma: f64 = f[1].parse().unwrap();
        let samples: usize = f[3].parse().unwrap();
        let start: Vec<f64> = f[13..21].iter().map(|s| s.parse().unwrap()).collect();
        let (time, flux): (Vec<f64>, Vec<f64>) = POINTS
            .lines()
            .skip(1)
            .filter_map(|l| {
                let p: Vec<&str> = l.split(',').collect();
                (p[0] == name).then(|| (p[1].parse::<f64>().unwrap(), p[2].parse::<f64>().unwrap()))
            })
            .unzip();
        let err = vec![sigma; time.len()];
        let lc = LightCurve::from_days(time, flux, err).unwrap();
        let mut set_up = TransitFit::new(&lc, params(&start));
        if samples > 1 {
            let exposure = Exposure::new(
                TimeDelta::from_days(f[2].parse().unwrap()).unwrap(),
                NonZeroUsize::new(samples).unwrap(),
            )
            .unwrap();
            set_up = set_up.with_exposure(exposure);
        }
        for p in Parameter::ALL {
            if f[4].split(';').any(|n| n == p.name()) {
                set_up = set_up.with_fixed(p);
            }
        }
        group.bench_function(name, |b| b.iter(|| black_box(&set_up).run().unwrap()));
    }
    group.finish();
}

criterion_group!(benches, fit);
criterion_main!(benches);
