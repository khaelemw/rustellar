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

#[test]
fn distinct_indices() {
    let mut rng = Rng::new(6);
    let mut first = [0u32; 5];
    for _ in 0..50_000 {
        let k = crate::moves::distinct(&mut rng, 5, 3);
        assert!(k[0] != k[1] && k[1] != k[2] && k[0] != k[2] && k.iter().all(|&i| i < 5));
        first[k[0]] += 1;
    }
    // Each first within 5 standard deviations of 10,000.
    assert!(
        first
            .iter()
            .all(|&c| (f64::from(c) - 1e4).abs() < 5.0 * 89.5)
    );
    let mut all = crate::moves::distinct(&mut rng, 4, 4);
    all.sort_unstable();
    assert_eq!(all, [0, 1, 2, 3]);
}

#[test]
fn differential_evolution_proposal() {
    let mut rng = Rng::new(7);
    let x = [1.0, 2.0];
    let (c1, c2) = ([0.0, 0.0], [3.0, -1.0]);
    let others: Vec<&[f64]> = vec![&c1, &c2];
    // With no spread, a step of exactly gamma times the difference.
    let mv = Move::differential_evolution(Some(0.5), 0.0).unwrap();
    let p = mv.propose(&x, &others, &mut rng);
    let step = [p.to[0] - x[0], p.to[1] - x[1]];
    assert!(step == [1.5, -0.5] || step == [-1.5, 0.5], "{step:?}");
    assert_eq!(p.log_ratio, 0.0);
    // The default scale, 2.38 / sqrt(2 d).
    let p = Move::differential_evolution(None, 0.0)
        .unwrap()
        .propose(&x, &others, &mut rng);
    let g = ((p.to[0] - x[0]) / 3.0).abs();
    assert!((g - 2.38 / 2.0).abs() < 1e-15);
}

#[test]
fn snooker_proposal() {
    let mut rng = Rng::new(8);
    let x = [1.0, 1.0, 0.0];
    let (a, b, c) = ([0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 3.0, -1.0]);
    let others: Vec<&[f64]> = vec![&a, &b, &c];
    let p = Move::snooker(1.7).unwrap().propose(&x, &others, &mut rng);
    // On the line through x and one of the others (z): y - z parallel to
    // x - z, and the Hastings term (d - 1) ln(|y - z| / |x - z|).
    let found = others.iter().any(|z| {
        let d: Vec<f64> = (0..3).map(|i| x[i] - z[i]).collect();
        let e: Vec<f64> = (0..3).map(|i| p.to[i] - z[i]).collect();
        let cross = [
            d[1] * e[2] - d[2] * e[1],
            d[2] * e[0] - d[0] * e[2],
            d[0] * e[1] - d[1] * e[0],
        ];
        let norm = |v: &[f64]| v.iter().map(|t| t * t).sum::<f64>().sqrt();
        cross.iter().all(|t| t.abs() < 1e-14)
            && (p.log_ratio - 2.0 * (norm(&e) / norm(&d)).ln()).abs() < 1e-14
    });
    assert!(found, "{p:?}");
    // On top of z: no move.
    let same: Vec<&[f64]> = vec![&x, &x, &x];
    let p = Move::snooker(1.7).unwrap().propose(&x, &same, &mut rng);
    assert_eq!(p.log_ratio, f64::NEG_INFINITY);
}

#[test]
fn walk_proposal() {
    // The steps' covariance is the helpers' covariance.
    let mut rng = Rng::new(9);
    let helpers = [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]];
    let others: Vec<&[f64]> = helpers.iter().map(|h| &h[..]).collect();
    let mv = Move::walk(None).unwrap();
    let n = 100_000;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for _ in 0..n {
        let p = mv.propose(&[5.0, 5.0], &others, &mut rng);
        assert_eq!(p.log_ratio, 0.0);
        let (u, v) = (p.to[0] - 5.0, p.to[1] - 5.0);
        sxx += u * u;
        syy += v * v;
        sxy += u * v;
    }
    let n = f64::from(n);
    // The helpers' covariance over s - 1: 4/3, 1/3 and 0.
    assert!((sxx / n - 4.0 / 3.0).abs() < 0.03, "{}", sxx / n);
    assert!((syy / n - 1.0 / 3.0).abs() < 0.01, "{}", syy / n);
    assert!((sxy / n).abs() < 0.01);
}

#[test]
fn walkers_needed() {
    assert_eq!(Move::default().walkers_needed(), 2);
    assert_eq!(Move::walk(None).unwrap().walkers_needed(), 4);
    assert_eq!(Move::walk(Some(5)).unwrap().walkers_needed(), 10);
    assert_eq!(Move::snooker(1.0).unwrap().walkers_needed(), 6);
}

#[test]
#[should_panic(expected = "not a proposal")]
fn slice_is_not_a_proposal() {
    let mut rng = Rng::new(1);
    let _ = Move::slice(1.0, 0)
        .unwrap()
        .propose(&[0.0], &[&[1.0]], &mut rng);
}
