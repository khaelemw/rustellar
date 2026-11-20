// The arrow solve against the full normal equations inverted directly
// (Gauss-Jordan with partial pivoting), on random groups, shapes, errors
// and fluxes; and what it cannot fit.

use crate::VetError;
use crate::grouped::solve;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// The inverse of `m` (n by n), by Gauss-Jordan elimination.
fn invert(mut m: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
            .unwrap();
        m.swap(col, pivot);
        inv.swap(col, pivot);
        let p = m[col][col];
        for j in 0..n {
            m[col][j] /= p;
            inv[col][j] /= p;
        }
        for row in 0..n {
            if row != col {
                let f = m[row][col];
                for j in 0..n {
                    m[row][j] -= f * m[col][j];
                    inv[row][j] -= f * inv[col][j];
                }
            }
        }
    }
    inv
}

#[test]
fn the_arrow_is_the_full_solve() {
    let mut r = Random(5);
    for case in 0..100 {
        let groups = 1 + (r.next() % 6) as usize;
        let n = 30 + (r.next() % 200) as usize;
        let mut shapes = vec![];
        let mut group = vec![];
        for _ in 0..n {
            if r.unit() < 0.4 {
                shapes.push(0.2 + 0.8 * r.unit());
                group.push(Some((r.next() % groups as u64) as usize));
            } else {
                shapes.push(0.0);
                group.push(None);
            }
        }
        let errs: Vec<f64> = (0..n).map(|_| 1e-3 * (0.5 + r.unit())).collect();
        let flux: Vec<f64> = (0..n).map(|_| 1.0 - 2e-3 * r.unit()).collect();
        let Ok(fit) = solve(&flux, &errs, &shapes, &group, groups) else {
            continue;
        };
        // The full normal equations over the baseline and the groups with
        // points.
        let used: Vec<usize> = (0..groups).filter(|&k| fit.points[k] > 0).collect();
        let size = used.len() + 1;
        let mut m = vec![vec![0.0; size]; size];
        let mut v = vec![0.0; size];
        for i in 0..n {
            let w = 1.0 / (errs[i] * errs[i]);
            let mut x = vec![0.0; size];
            x[0] = 1.0;
            if let Some(k) = group[i] {
                if let Some(pos) = used.iter().position(|&u| u == k) {
                    x[pos + 1] = shapes[i];
                }
            }
            for a in 0..size {
                v[a] += w * x[a] * flux[i];
                for b in 0..size {
                    m[a][b] += w * x[a] * x[b];
                }
            }
        }
        let cov = invert(m);
        let params: Vec<f64> = (0..size)
            .map(|a| (0..size).map(|b| cov[a][b] * v[b]).sum())
            .collect();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1e-30);
        assert!(close(fit.alpha, params[0]), "{case}");
        assert!(close(fit.var_alpha, cov[0][0]), "{case}");
        for (j, &gj) in used.iter().enumerate() {
            let bj = fit.beta[gj].unwrap();
            assert!(
                (bj - params[j + 1]).abs() <= 1e-7 * bj.abs().max(1e-6),
                "{case}"
            );
            assert!(close(fit.cov_alpha_beta(gj), cov[0][j + 1]), "{case}");
            for (k, &gk) in used.iter().enumerate() {
                assert!(close(fit.cov_beta(gj, gk), cov[j + 1][k + 1]), "{case}");
            }
        }
        for k in 0..groups {
            assert_eq!(fit.beta[k].is_some(), fit.points[k] > 0);
            assert_eq!(fit.depth(k).is_some(), fit.points[k] > 0);
        }
    }
}

#[test]
fn what_it_cannot_fit() {
    let none = [None, None];
    assert_eq!(
        solve(&[1.0, 1.0], &[1.0, 1.0], &[0.0, 0.0], &none, 1),
        Err(VetError::NoPointsInTransit)
    );
    // A shape of 0 in a group is out of transit.
    assert_eq!(
        solve(
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[Some(0), Some(0)],
            1
        ),
        Err(VetError::NoPointsInTransit)
    );
    assert_eq!(
        solve(
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 0.5],
            &[Some(0), Some(0)],
            1
        ),
        Err(VetError::NoPointsOutOfTransit)
    );
    assert_eq!(
        solve(&[-1.0, -1.0], &[1.0, 1.0], &[1.0, 0.0], &[Some(0), None], 1),
        Err(VetError::BaselineNotPositive)
    );
    // A group with no points has no depth and no covariance.
    let fit = solve(&[0.9, 1.0], &[1.0, 1.0], &[1.0, 0.0], &[Some(1), None], 2).unwrap();
    assert_eq!(fit.depth(0), None);
    assert_eq!(fit.cov_depth(0, 1), None);
    assert_eq!(fit.cov_depth(1, 0), None);
    assert!((fit.depth(1).unwrap() - 0.1).abs() < 1e-15);
}
