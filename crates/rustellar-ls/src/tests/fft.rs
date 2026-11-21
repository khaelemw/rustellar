use crate::fft::{Complex, Fft, turn};

/// Mode `m` of the transform by its definition, in exact phases.
fn direct(x: &[Complex], m: usize) -> Complex {
    let n = x.len();
    let mut sum = Complex::default();
    for (l, &v) in x.iter().enumerate() {
        let p = v.mul(turn(((m * l) % n) as f64 / n as f64));
        sum.re += p.re;
        sum.im += p.im;
    }
    sum
}

#[test]
fn fft_as_its_definition_on_any_threads() {
    for n in [1, 2, 4, 8, 64, 1024, 1 << 16] {
        let x: Vec<Complex> = (0..n)
            .map(|i| Complex::new(((i * 37) % 11) as f64 - 5.0, ((i * 13) % 7) as f64))
            .collect();
        let fft = Fft::new(n);
        assert_eq!(fft.len(), n);
        let scale: f64 = x.iter().map(|c| c.re.abs() + c.im.abs()).sum();
        // Every mode of the small, a spread of the large.
        let modes: Vec<usize> = (0..n).step_by((n / 256).max(1)).chain([n - 1]).collect();
        let mut first = None;
        for threads in [1, 3, 16, 100] {
            let mut y = x.clone();
            fft.run(&mut y, threads);
            for &m in &modes {
                let (a, b) = (y[m], direct(&x, m));
                let bound = 1e-15 * scale * (n as f64).log2().max(1.0);
                assert!(
                    (a.re - b.re).abs() + (a.im - b.im).abs() <= bound,
                    "{n} {m}"
                );
            }
            // The same bits on any number of threads.
            assert_eq!(*first.get_or_insert_with(|| y.clone()), y, "{n} {threads}");
        }
    }
}

#[test]
#[should_panic(expected = "length 6")]
fn fft_needs_a_power_of_two() {
    let _ = Fft::new(6);
}

#[test]
fn turns() {
    assert_eq!(turn(0.25), Complex::new(0.0, 1.0));
    assert_eq!(turn(-0.5), Complex::new(-1.0, 0.0));
}
