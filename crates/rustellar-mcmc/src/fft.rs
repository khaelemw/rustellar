//! Autocovariance by the fast Fourier transform: a radix-2 transform with
//! exact twiddle factors, the series padded to a power of two at least
//! twice its length so the circular correlation is the linear one.

use std::f64::consts::TAU;

/// The transform of `(re, im)` in place (length a power of two), forward
/// (`sign` -1) or inverse without the `1 / n` (`sign` 1).
fn transform(re: &mut [f64], im: &mut [f64], sign: f64) {
    let n = re.len();
    // Bit reversal.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // Each twiddle from its own angle, not by recurrence.
        #[allow(clippy::cast_precision_loss)]
        let twiddles: Vec<(f64, f64)> = (0..half)
            .map(|k| {
                let (s, c) = (TAU * k as f64 / len as f64).sin_cos();
                (c, sign * s)
            })
            .collect();
        for start in (0..n).step_by(len) {
            for (k, &(wr, wi)) in twiddles.iter().enumerate() {
                let (a, b) = (start + k, start + k + half);
                let tr = wr * re[b] - wi * im[b];
                let ti = wr * im[b] + wi * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len *= 2;
    }
}

/// The sums `sum_i (x_i - mean)(x_{i+t} - mean)` for every lag `t` below
/// the series' length (not divided by anything).
pub(crate) fn autocovariance_sums(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = x.iter().sum::<f64>() / n as f64;
    let m = (2 * n).next_power_of_two();
    let mut re = vec![0.0; m];
    let mut im = vec![0.0; m];
    for (r, &v) in re.iter_mut().zip(x) {
        *r = v - mean;
    }
    transform(&mut re, &mut im, -1.0);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = r.mul_add(*r, *i * *i);
        *i = 0.0;
    }
    transform(&mut re, &mut im, 1.0);
    #[allow(clippy::cast_precision_loss)]
    let scale = (m as f64).recip();
    re.truncate(n);
    for r in &mut re {
        *r *= scale;
    }
    re
}
