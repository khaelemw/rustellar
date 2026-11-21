// Speed of the periodogram, for the numbers in the README: uneven samples
// of a sinusoid in noise, at astropy's automatic grid's sizes.
//
// Run with `cargo bench -p rustellar-ls`.

use std::{f64::consts::TAU, hint::black_box, num::NonZeroUsize, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use rustellar_ls::{Frequencies, LightCurve, LombScargle, Mean, Normalization};

/// `n` points over 30 days, unevenly spaced, a 2.7-day sinusoid in a
/// fixed pseudo-random sequence of noise.
fn curve(n: u32) -> LightCurve {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[expect(clippy::cast_precision_loss, reason = "53 random bits")]
        let u = (state >> 11) as f64 / (1_u64 << 53) as f64;
        u
    };
    let step = 30.0 / f64::from(n);
    let time: Vec<f64> = (0..n)
        .map(|i| step * (f64::from(i) + 0.9 * next()))
        .collect();
    let flux = time
        .iter()
        .map(|t| 1.0 + 1e-3 * (TAU * t / 2.7).sin() + 1e-3 * (next() - 0.5))
        .collect();
    LightCurve::from_days(time, flux, vec![1e-3; n as usize]).unwrap()
}

fn exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for (n, nf) in [(1_000, 10_000), (10_000, 10_000)] {
        let lc = curve(n);
        let grid = Frequencies::uniform(0.01, 0.005, nf).unwrap();
        let ls = LombScargle::new(&lc, Mean::Floating).unwrap();
        let one = ls.clone().with_threads(NonZeroUsize::MIN);
        group.bench_function(format!("{n} points x {nf} frequencies, 1 thread"), |b| {
            b.iter(|| black_box(one.power(&grid, Normalization::Standard)));
        });
        group.bench_function(format!("{n} points x {nf} frequencies, all threads"), |b| {
            b.iter(|| black_box(ls.power(&grid, Normalization::Standard)));
        });
    }
    group.finish();
}

criterion_group!(benches, exact);
criterion_main!(benches);
