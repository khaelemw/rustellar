// Targets, starting points and checks shared by the integration tests.

#![allow(dead_code)]

use rustellar_mcmc::Chain;

/// A Gaussian with precision matrix `p` (row by row) about `mu`.
pub fn gaussian(mu: Vec<f64>, p: Vec<f64>) -> impl Fn(&[f64]) -> f64 + Sync {
    move |x: &[f64]| {
        let d = mu.len();
        let r: Vec<f64> = x.iter().zip(&mu).map(|(a, m)| a - m).collect();
        let mut q = 0.0;
        for i in 0..d {
            for j in 0..d {
                q += r[i] * p[i * d + j] * r[j];
            }
        }
        -0.5 * q
    }
}

/// `walkers` points in a small ball about `centre`, scaled by `scale`.
pub fn ball(walkers: usize, centre: &[f64], scale: &[f64], seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..walkers)
        .map(|_| {
            centre
                .iter()
                .zip(scale)
                .map(|(c, s)| c + 0.1 * s * next())
                .collect()
        })
        .collect()
}

/// Each mean within `k` of its standard deviation of `mu`, and each
/// covariance within `k` relative to the product of standard deviations.
pub fn assert_moments(chain: &Chain, discard: usize, mu: &[f64], cov: &[f64], k: f64) {
    let d = mu.len();
    let (m, c) = (chain.mean(discard), chain.covariance(discard));
    for i in 0..d {
        let sd = cov[i * d + i].sqrt();
        assert!(
            (m[i] - mu[i]).abs() < k * sd,
            "mean {i}: {} against {}",
            m[i],
            mu[i]
        );
        for j in 0..d {
            let scale = sd * cov[j * d + j].sqrt();
            let gap = (c[i * d + j] - cov[i * d + j]).abs() / scale;
            assert!(
                gap < k,
                "cov {i} {j}: {} against {}",
                c[i * d + j],
                cov[i * d + j]
            );
        }
    }
}

pub fn invert3(m: &[f64]) -> Vec<f64> {
    let c = |r: usize, k: usize| m[r * 3 + k];
    let cof = |r: usize, k: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
        c(r1, k1) * c(r2, k2) - c(r1, k2) * c(r2, k1)
    };
    let det: f64 = (0..3).map(|k| c(0, k) * cof(0, k)).sum();
    let mut out = vec![0.0; 9];
    for r in 0..3 {
        for k in 0..3 {
            out[k * 3 + r] = cof(r, k) / det;
        }
    }
    out
}

/// The three-dimensional test Gaussian: standard deviations 1e-3, 1 and
/// 1e3, correlations 0.9, -0.3 and 0, about (1, -2, 5e3). Its mean,
/// covariance and log-density.
pub fn badly_scaled() -> (Vec<f64>, Vec<f64>, impl Fn(&[f64]) -> f64 + Sync) {
    let sd = [1e-3, 1.0, 1e3];
    let rho = [[1.0, 0.9, -0.3], [0.9, 1.0, 0.0], [-0.3, 0.0, 1.0]];
    let mut cov = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            cov[i * 3 + j] = rho[i][j] * sd[i] * sd[j];
        }
    }
    let mu = vec![1.0, -2.0, 5e3];
    let target = gaussian(mu.clone(), invert3(&cov));
    (mu, cov, target)
}
