// TEMPORARY: not to be committed. Where the bootstrap's tails come from at
// 2 minutes: how far the misses are, in bootstrap errors, and a wider reach.

use std::num::NonZeroUsize;

use super::{Random, light_curve};
use crate::bootstrap::bootstrap;
use crate::errors::errors;
use crate::fit::{Position, fit};
use crate::refine::{Reach, refine};
use crate::template::{SAMPLED_INTERVALS, Sampled, Template};

const TRUTH: Position = Position {
    period: 3.1,
    duration: 0.12,
    t0: 1.3,
};

fn var(name: &str, default: f64) -> f64 {
    std::env::var(name).map_or(default, |n| n.parse().unwrap())
}

fn quantiles(name: &str, z: &mut [f64]) {
    z.sort_by(f64::total_cmp);
    let q = |p: f64| z[((z.len() - 1) as f64 * p).round() as usize];
    let over = |k: f64| z.iter().filter(|x| **x > k).count();
    eprintln!(
        "RESULT {name}: |z| q50 {:.2} q68 {:.2} q90 {:.2} q95 {:.2} q99 {:.2} max {:.1}; >2 {} >3 {} >5 {} of {}",
        q(0.5),
        q(0.68),
        q(0.9),
        q(0.95),
        q(0.99),
        z[z.len() - 1],
        over(2.0),
        over(3.0),
        over(5.0),
        z.len()
    );
}

#[test]
#[ignore = "temporary"]
fn tails() {
    let noise = var("NOISE", 4.6e-3);
    let cadence = var("CADENCE", 0.001_388_889);
    let count = var("COUNT", 150.0) as u32;
    let resamples = var("B", 50.0) as usize;
    let template = Template::tls_default();
    let sampled = Sampled::new(&template, SAMPLED_INTERVALS);
    let mut random = Random::new(41);
    let names = ["period", "t0", "duration"];
    let mut z: [[Vec<f64>; 3]; 3] = Default::default();
    for _ in 0..count {
        let points = light_curve(&mut random, &template, TRUTH, 2e-3, noise, (0.0, 27.0, cadence));
        let reach = Reach {
            t0: 0.02,
            duration: 1.3,
            period: 5e-3,
        };
        let p = refine(&points, &sampled, TRUTH, reach).position;
        let fitted = fit(&points, &template, p);
        let Some(e) = errors(&points, &template, p, fitted.depth) else {
            continue;
        };
        let sig = [e.period, e.t0, e.duration];
        let four = NonZeroUsize::new(4).unwrap();
        let b = bootstrap(&points, &sampled, &template, p, fitted.depth, sig, resamples, four);
        let wide = bootstrap(
            &points,
            &sampled,
            &template,
            p,
            fitted.depth,
            sig.map(|s| 2.5 * s),
            resamples,
            four,
        );
        let off = [p.period - TRUTH.period, p.t0 - TRUTH.t0, p.duration - TRUTH.duration];
        let boot = [b.period, b.t0, b.duration];
        let boot_wide = [wide.period, wide.t0, wide.duration];
        for i in 0..3 {
            z[0][i].push((off[i] / sig[i]).abs());
            z[1][i].push((off[i] / boot[i]).abs());
            z[2][i].push((off[i] / boot_wide[i]).abs());
        }
    }
    for (k, label) in ["curvature", "bootstrap reach 8", "bootstrap reach 20"].iter().enumerate() {
        for i in 0..3 {
            quantiles(&format!("{label} {}", names[i]), &mut z[k][i]);
        }
    }
}
