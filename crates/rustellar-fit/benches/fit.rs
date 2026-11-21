// The fit's speed: filled in with the fit itself.

use criterion::{Criterion, criterion_group, criterion_main};

fn fit(c: &mut Criterion) {
    c.bench_function("nothing yet", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, fit);
criterion_main!(benches);
