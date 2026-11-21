// The stretch factor's distribution, the uniform index and a stretch
// proposal's geometry and Hastings term.

use crate::Move;
use crate::moves::{index, stretch_factor};
use crate::rng::Rng;

#[test]
fn stretch_factor_range_and_distribution() {
    let a = 2.0;
    assert_eq!(stretch_factor(0.0, a), 0.5);
    assert!((stretch_factor(1.0 - f64::EPSILON / 2.0, a) - a).abs() < 1e-15);
    // g(z) ∝ 1 / sqrt(z) on [1/a, a]: E[z] = (a^1.5 - a^-1.5) / (3 (a^0.5 - a^-0.5)).
    let mut rng = Rng::new(1);
    let n = 200_000;
    let mean = (0..n).map(|_| stretch_factor(rng.unit(), a)).sum::<f64>() / f64::from(n);
    let want = (a.powf(1.5) - a.powf(-1.5)) / (3.0 * (a.sqrt() - 1.0 / a.sqrt()));
    assert!((mean - want).abs() < 2e-3, "{mean} {want}");
}

#[test]
fn uniform_index() {
    let mut rng = Rng::new(2);
    let mut counts = [0u32; 7];
    for _ in 0..70_000 {
        counts[index(&mut rng, 7)] += 1;
    }
    // Each within 5 standard deviations of 10,000.
    assert!(
        counts
            .iter()
            .all(|&c| (f64::from(c) - 1e4).abs() < 5.0 * 92.6)
    );
    assert_eq!(index(&mut rng, 1), 0);
}

#[test]
fn stretch_proposal() {
    let mut rng = Rng::new(3);
    let x = [1.0, 2.0, 3.0];
    let c = [0.0, 0.0, 1.0];
    let others: Vec<&[f64]> = vec![&c];
    let p = Move::default().propose(&x, &others, &mut rng);
    // On the line through c and x: y - c = z (x - c), the same z for all.
    let z = (p.to[0] - c[0]) / (x[0] - c[0]);
    for i in 0..3 {
        assert!((p.to[i] - c[i] - z * (x[i] - c[i])).abs() < 1e-15);
    }
    assert!((p.log_ratio - 2.0 * z.ln()).abs() < 1e-15);
    assert!((0.5..=2.0).contains(&z));
}
