// The fit at one position: the compensated sum, the depth and chi-squared
// against exact integer arithmetic, noiseless transits, the least-squares
// property, invariances and the template's placement in phase.

use rustellar_core::LightCurve;

use super::Random;
use crate::fit::{Points, Position, Sum, chi2_shapes, fit, fit_shapes};
use crate::template::Template;

fn points(time: Vec<f64>, residual: Vec<f64>, weight: Vec<f64>) -> Points {
    Points {
        time,
        residual,
        weight,
    }
}

#[test]
fn the_sum_keeps_what_plain_addition_loses() {
    let mut sum = Sum::default();
    for x in [1e16, 1.0, -1e16] {
        sum.add(x);
    }
    assert_eq!(sum.value(), 1.0);
    let mut sum = Sum::default();
    for x in [1.0, 1e100, 1.0, -1e100] {
        sum.add(x);
    }
    assert_eq!(sum.value(), 2.0);
    assert_eq!(Sum::default().value(), 0.0);
}

#[test]
fn the_sum_of_dyadic_terms_is_exact() {
    // Terms that are integers over 2^40, any sign, across 60 binary orders:
    // the exact sum in i128 fits a double here, so it must come out exact.
    let mut random = Random::new(5);
    for _ in 0..2000 {
        let mut sum = Sum::default();
        let mut exact: i128 = 0;
        for _ in 0..50 {
            let shift = random.next_u64() % 40;
            let magnitude = i128::from(random.next_u64() % (1 << 12)) << shift;
            let term = if random.next_u64() % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
            exact += term;
            sum.add(term as f64 / (1_u64 << 40) as f64);
        }
        let exact = exact as f64 / (1_u64 << 40) as f64;
        assert_eq!(sum.value(), exact);
    }
}

#[test]
fn depth_and_chi2_against_exact_integers() {
    // r = a / 2^10, s = b / 2^10 and w = c: every term w r s and w s² is
    // an integer over 2^20, exactly a double, and every sum fits in 53
    // bits, so the depth (a quotient of two exact sums) is correctly
    // rounded only if the sums are exact; the chi-squared is checked
    // against (S_rr S_ss - S_rs²) / S_ss in i128.
    let mut random = Random::new(8);
    let scale = 1024.0;
    for _ in 0..3000 {
        let n = 1 + usize::try_from(random.next_u64() % 300).unwrap();
        let (mut a, mut b, mut c) = (vec![], vec![], vec![]);
        for _ in 0..n {
            a.push(i64::try_from(random.next_u64() % 2001).unwrap() - 900);
            let inside = random.next_u64() % 3 != 0;
            b.push(if inside {
                i64::try_from(random.next_u64() % 1025).unwrap()
            } else {
                0
            });
            c.push(1 + i64::try_from(random.next_u64() % 100).unwrap());
        }
        let to = |v: &[i64], s: f64| v.iter().map(|&x| x as f64 / s).collect::<Vec<_>>();
        let p = points(vec![0.0; n], to(&a, scale), to(&c, 1.0));
        let shapes = to(&b, scale);
        let got = fit_shapes(&p, &shapes);

        let (mut rr, mut rs, mut ss) = (0_i128, 0_i128, 0_i128);
        for i in 0..n {
            let (a, b, c) = (i128::from(a[i]), i128::from(b[i]), i128::from(c[i]));
            rr += c * a * a;
            if b > 0 {
                rs += c * a * b;
                ss += c * b * b;
            }
        }
        assert_eq!(
            got.in_transit,
            b.iter().filter(|&&b| b > 0).count(),
            "in transit"
        );
        if rs > 0 {
            assert_eq!(got.depth, rs as f64 / ss as f64);
            let exact = (rr * ss - rs * rs) as f64 / ss as f64 / (scale * scale);
            // The depth is rounded, which adds S_ss (d - d*)², at most
            // S_ss (d ulp)²: all there is where the fit is perfect.
            let rounding = ss as f64 / (scale * scale) * (got.depth * f64::EPSILON).powi(2);
            assert!(
                (got.chi2 - exact).abs() <= 1e-13 * exact + rounding,
                "{} against {exact}",
                got.chi2
            );
        } else {
            assert_eq!(got.depth, 0.0);
            assert_eq!(got.chi2, rr as f64 / (scale * scale));
        }
    }
}

#[test]
fn the_least_squares_depth_is_least() {
    let mut random = Random::new(9);
    for _ in 0..500 {
        let n = 50;
        let shapes: Vec<f64> = (0..n).map(|_| random.uniform().max(0.0)).collect();
        let residual = (0..n).map(|_| random.between(-0.01, 0.02)).collect();
        let weight = (0..n).map(|_| random.between(0.1, 10.0)).collect();
        let p = points(vec![0.0; n], residual, weight);
        let best = fit_shapes(&p, &shapes);
        assert_eq!(chi2_shapes(&p, &shapes, best.depth), best.chi2);
        for _ in 0..20 {
            let other = random.between(-0.05, 0.05);
            assert!(chi2_shapes(&p, &shapes, other) >= best.chi2, "{other}");
        }
    }
}

#[test]
fn no_dip_no_depth() {
    // Every point in transit at or above 1: the depth is 0 and the
    // chi-squared that of the flat model.
    let residual = vec![-0.01, 0.0, -0.002, 0.003, 0.004];
    let shapes = vec![0.5, 1.0, 0.5, 0.0, 0.0];
    let p = points(vec![0.0; 5], residual, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let got = fit_shapes(&p, &shapes);
    assert_eq!(got.depth, 0.0);
    assert_eq!(got.chi2, p.flat_chi2());
    assert_eq!(got.in_transit, 3);
    // And nothing in transit at all.
    let got = fit_shapes(&p, &[0.0; 5]);
    assert_eq!((got.depth, got.in_transit), (0.0, 0));
    assert_eq!(got.chi2, p.flat_chi2());
}

#[test]
fn a_noiseless_transit_is_fitted_exactly() {
    let template = Template::tls_default();
    let position = Position {
        period: 3.7,
        duration: 0.13,
        t0: 1.3,
    };
    let depth = 0.0123;
    let time: Vec<f64> = (0..5000).map(|i| f64::from(i) * 0.0054).collect();
    let flux: Vec<f64> = time
        .iter()
        .map(|&t| 1.0 - depth * template.shape(position.x(t)))
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 5000]).unwrap();
    let got = fit(&Points::new(&lc), &template, position);
    assert!((got.depth / depth - 1.0).abs() < 1e-14, "{}", got.depth);
    // Residuals of a rounding of 1 each: chi2 about n_in (1e-16 / 1e-3)².
    assert!(got.chi2 < 1e-20, "{}", got.chi2);
    assert!(got.in_transit > 100);
    // Shifted by a fifth of the duration, the fit is worse.
    let shifted = Position {
        t0: 1.3 + 0.026,
        ..position
    };
    let worse = fit(&Points::new(&lc), &template, shifted);
    assert!(worse.chi2 > 1.0, "{}", worse.chi2);
    assert!(worse.depth < depth);
}

#[test]
fn scaling_the_weights_scales_the_chi2_only() {
    let mut random = Random::new(12);
    let n = 200;
    let shapes: Vec<f64> = (0..n).map(|_| random.uniform()).collect();
    let residual: Vec<f64> = (0..n).map(|_| random.between(-0.01, 0.02)).collect();
    let weight: Vec<f64> = (0..n).map(|_| random.between(0.1, 10.0)).collect();
    let base = fit_shapes(
        &points(vec![0.0; n], residual.clone(), weight.clone()),
        &shapes,
    );
    for power in [-30, -1, 1, 7, 40] {
        let factor = 2_f64.powi(power);
        let scaled: Vec<f64> = weight.iter().map(|w| w * factor).collect();
        let got = fit_shapes(&points(vec![0.0; n], residual.clone(), scaled), &shapes);
        assert_eq!(got.depth, base.depth);
        assert_eq!(got.chi2, base.chi2 * factor);
    }
}

#[test]
fn points_from_a_light_curve() {
    let lc = LightCurve::from_btjd(vec![1.0, 2.0], vec![0.99, 1.25], vec![0.5, 0.25]).unwrap();
    let p = Points::new(&lc);
    assert_eq!(p.time, vec![1.0, 2.0]);
    assert_eq!(p.residual, vec![1.0 - 0.99, -0.25]);
    assert_eq!(p.weight, vec![4.0, 16.0]);
    assert_eq!(p.flat_chi2(), 4.0 * (1.0 - 0.99) * (1.0 - 0.99) + 1.0);
}

#[test]
fn the_template_is_placed_by_phase() {
    let position = Position {
        period: 2.0,
        duration: 0.25,
        t0: 10.0,
    };
    // At every mid-transit, 0; half an orbit away, -P/2 / D (the half-open
    // end); a quarter of the duration after, 1/4.
    for k in [-1000.0, -3.0, 0.0, 1.0, 500.0] {
        let t = 10.0 + 2.0 * k;
        assert_eq!(position.x(t), 0.0, "{k}");
        assert_eq!(position.x(t + 1.0), -4.0);
        assert_eq!(position.x(t - 1.0), -4.0);
        assert!((position.x(t + 0.0625) - 0.25).abs() < 1e-12);
        assert!((position.x(t - 0.0625) + 0.25).abs() < 1e-12);
    }
}

#[test]
fn in_transit_counts_the_points_the_template_covers() {
    let template = Template::tls_box();
    let position = Position {
        period: 1.0,
        duration: 0.1,
        t0: 0.0,
    };
    // Points every 0.01 d over 10 days: 9 per transit strictly inside
    // +-0.05 of an integer. A point at +-0.05 rounds to a hair either side
    // of contact, where the planet's flux is 1 in doubles: s = 0, out.
    let time: Vec<f64> = (0..1000).map(|i| f64::from(i) / 100.0).collect();
    let n = time.len();
    let p = points(time, vec![0.001; n], vec![1.0; n]);
    let got = fit(&p, &template, position);
    let covered = p
        .time
        .iter()
        .filter(|&&t| template.shape(position.x(t)) > 0.0)
        .count();
    assert_eq!(got.in_transit, covered);
    assert_eq!(covered, 90);
}
