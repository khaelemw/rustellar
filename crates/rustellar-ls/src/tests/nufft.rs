use crate::fft::{Complex, turn};
use crate::nufft::{Nufft, gauss_legendre};

#[test]
fn gauss_legendre_integrates_polynomials() {
    for q in [1, 2, 5, 12, 34] {
        let (nodes, weights) = gauss_legendre(q);
        // Exact for degree up to 2q - 1 on [0, 1]: x^k integrates to 1/(k+1).
        for k in 0..2 * q {
            let sum: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            assert!(
                (sum - 1.0 / (k as f64 + 1.0)).abs() < 1e-14,
                "q {q} k {k}: {sum}"
            );
        }
    }
}

/// `sum_j c_j e^{2 pi i m r_j}` directly.
fn direct(cycles: &[f64], c: &[Complex], m: i64) -> Complex {
    let mut sum = Complex::default();
    for (&r, &v) in cycles.iter().zip(c) {
        let p = v.mul(turn(m as f64 * r));
        sum.re += p.re;
        sum.im += p.im;
    }
    sum
}

#[test]
fn nufft_to_its_tolerance() {
    // Seeded points in [-0.5, 0.5], strengths of mixed sizes.
    let mut state = 12345_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let cycles: Vec<f64> = (0..500).map(|_| next() - 0.5).collect();
    let a: Vec<Complex> = (0..500)
        .map(|_| Complex::new(next() - 0.3, 2.0 * next()))
        .collect();
    let b: Vec<Complex> = (0..500).map(|_| Complex::new(next(), 0.0)).collect();
    let scale = |c: &[Complex]| c.iter().map(|v| v.re.hypot(v.im)).sum::<f64>();
    for (modes, tolerance) in [
        (64, 1e-6),
        (300, 1e-9),
        (301, 1e-12),
        (1000, 1e-14),
        (10, 0.5),
    ] {
        let plan = Nufft::new(modes, tolerance, 4);
        assert_eq!(plan.modes() % 2, 0);
        assert!((2..=16).contains(&plan.width()));
        let [fa, fb] = plan.run(&cycles, [&a, &b], 3);
        // The same bits on any number of threads, the plan's too.
        let other = Nufft::new(modes, tolerance, 1);
        assert_eq!(other.run(&cycles, [&a, &b], 16), [fa.clone(), fb.clone()]);
        let half = (plan.modes() / 2) as i64;
        let mut worst: f64 = 0.0;
        for (k, (x, y)) in fa.iter().zip(&fb).enumerate() {
            let m = k as i64 - half;
            for (got, want, s) in [
                (x, direct(&cycles, &a, m), scale(&a)),
                (y, direct(&cycles, &b, m), scale(&b)),
            ] {
                worst = worst.max(((got.re - want.re).hypot(got.im - want.im)) / s);
            }
        }
        // Within the tolerance asked (clamped to 0.1).
        assert!(
            worst < tolerance.min(0.1),
            "{modes} {tolerance:e}: {worst:e}"
        );
        println!("{modes} {tolerance:e}: {worst:e}");
    }
}
