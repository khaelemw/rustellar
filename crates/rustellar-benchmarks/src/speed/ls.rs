//! `rustellar-ls` against astropy's `LombScargle` (`slow`, `cython` and
//! `fast`), floating mean, standard normalisation: `n` points over 30 days,
//! each in its own slot of `30 / n` days at a fraction of it set by the
//! golden ratio, a 2.7-day sinusoid and even noise from the plastic
//! number's multiples (no generator, so both packages see the same light
//! curve), errors 1e-3; an even grid of frequencies from 0.01 per day,
//! step 0.005.

use std::fmt::Write;
use std::num::NonZeroUsize;

use rustellar_core::LightCurve;
use rustellar_ls::{Frequencies, LombScargle, Mean, Normalization};

use super::bls::thousands;
use super::{Reference, best, package, seconds, short};

/// The references' times, as the script that ran them recorded them.
#[must_use]
pub fn reference() -> &'static str {
    include_str!("../../data/speed_ls.csv")
}

/// The exact method's sizes: points and frequencies.
pub const EXACT: [(u32, u32); 2] = [(1_000, 10_000), (10_000, 10_000)];
/// The fast method's.
pub const FAST: [(u32, u32); 5] = [
    (1_000, 10_000),
    (10_000, 10_000),
    (10_000, 100_000),
    (100_000, 100_000),
    (1_000_000, 1_000_000),
];
/// The fast method's tolerances.
pub const TOLERANCES: [f64; 2] = [1e-6, 1e-12];

fn fraction(x: f64) -> f64 {
    x - x.floor()
}

/// The light curve of `n` points.
///
/// # Panics
///
/// Never: the times rise and every value is finite.
#[must_use]
pub fn curve(n: u32) -> LightCurve {
    let slot = 30.0 / f64::from(n);
    let time: Vec<f64> = (0..n)
        .map(|i| slot * (f64::from(i) + fraction(f64::from(i + 1) * 0.618_033_988_749_894_9)))
        .collect();
    let flux = (0..n)
        .zip(&time)
        .map(|(i, t)| {
            let noise = 3.464 * (fraction(f64::from(i + 1) * 0.754_877_666_246_692_7) - 0.5);
            1.0 + 1e-3 * (std::f64::consts::TAU * t / 2.7).sin() + 1e-3 * noise
        })
        .collect();
    LightCurve::from_days(time, flux, vec![1e-3; n as usize]).expect("valid")
}

fn grid(count: u32) -> Frequencies {
    Frequencies::uniform(0.01, 0.005, count as usize).expect("valid")
}

/// One time: a case (`exact 1000 x 10000 16`: the sizes and threads;
/// `fast 1000 x 10000 1e-6`: the sizes and tolerance, on every thread) and
/// seconds.
pub type Timing = (String, f64);

/// The exact method at `exact`'s sizes on one thread and on `threads`, the
/// fast one at `fast`'s on `threads` to each tolerance; the best of three
/// each, as astropy's.
#[must_use]
pub fn run(threads: NonZeroUsize, exact: &[(u32, u32)], fast: &[(u32, u32)]) -> Vec<Timing> {
    let mut out = vec![];
    for &(n, nf) in exact {
        let (lc, frequencies) = (curve(n), grid(nf));
        for t in [NonZeroUsize::MIN, threads] {
            let ls = LombScargle::new(&lc, Mean::Floating)
                .expect("valid")
                .with_threads(t);
            let s = best(3, 1, || ls.power(&frequencies, Normalization::Standard));
            out.push((format!("exact {n} x {nf} {t}"), s));
        }
    }
    for &(n, nf) in fast {
        let (lc, frequencies) = (curve(n), grid(nf));
        let ls = LombScargle::new(&lc, Mean::Floating)
            .expect("valid")
            .with_threads(threads);
        for tolerance in TOLERANCES {
            let s = best(3, 1, || {
                ls.power_fast(&frequencies, Normalization::Standard, tolerance)
                    .expect("an even grid")
            });
            out.push((format!("fast {n} x {nf} {tolerance:e}"), s));
        }
    }
    out
}

fn ours(timings: &[Timing], case: &str) -> String {
    short(timings.iter().find(|(c, _)| c == case).map(|&(_, s)| s))
}

fn size(n: u32, nf: u32) -> String {
    format!("{} x {}", thousands(f64::from(n)), thousands(f64::from(nf)))
}

/// The README's two tables: the exact method against astropy's slow and
/// cython ones, then the fast method against astropy's fast one.
#[must_use]
pub fn report(timings: &[Timing], references: &[Reference], threads: NonZeroUsize) -> String {
    let astropy = package(references, "astropy");
    let theirs = |case: String| short(seconds(references, "astropy", &case));
    let mut out = format!(
        "| Points x frequencies | exact, 1 thread | exact, {threads} threads | {astropy} slow | {astropy} cython |\n\
         |---|--:|--:|--:|--:|\n"
    );
    for (n, nf) in EXACT {
        writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            size(n, nf),
            ours(timings, &format!("exact {n} x {nf} 1")),
            ours(timings, &format!("exact {n} x {nf} {threads}")),
            theirs(format!("slow {n} x {nf}")),
            theirs(format!("cython {n} x {nf}")),
        )
        .expect("a string");
    }
    write!(
        out,
        "\nThe fast method on {threads} threads, against astropy's (which has no tolerance):\n\n\
         | Points x frequencies | tolerance 1e-6 | tolerance 1e-12 | {astropy} fast |\n\
         |---|--:|--:|--:|\n"
    )
    .expect("a string");
    for (n, nf) in FAST {
        writeln!(
            out,
            "| {} | {} | {} | {} |",
            size(n, nf),
            ours(timings, &format!("fast {n} x {nf} 1e-6")),
            ours(timings, &format!("fast {n} x {nf} 1e-12")),
            theirs(format!("fast {n} x {nf}")),
        )
        .expect("a string");
    }
    out
}
