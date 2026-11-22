//! `rustellar-vet`'s steps, one thread: a 3.7-day planet (k 0.05, b 0.5,
//! a/R* 15) in a 27-day sector at 2 minutes (19,440 points) and in four
//! years at 30 minutes (70,128 points), with uniform noise of 1e-3 from a
//! fixed sequence, as the crate's own benches. No Python package runs the
//! same checks, so there is no reference column.

use std::fmt::Write;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

use rustellar_model::{ScaledSemiMajorAxis, TransitParams};
use rustellar_vet::{
    Candidate, Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, Shape, TimeDelta,
    aliases, evidence, fit_depth, refine, refine_with_threads, secondary, signal_to_noise,
};

use super::{best, short};

/// The steps timed, in the README's order.
pub const STEPS: [&str; 6] = [
    "fit_depth",
    "refine",
    "evidence",
    "signal_to_noise",
    "aliases",
    "secondary",
];

/// The light curves: a name, the light curve and its candidate.
pub type Case = (&'static str, LightCurve, Candidate);

/// The planet over `days` days every `step` days.
///
/// # Panics
///
/// Never for positive `days` and `step`.
#[must_use]
pub fn case(name: &'static str, days: f64, step: f64) -> Case {
    let at = |t: f64| Epoch::from_days(t).expect("finite");
    let p = TransitParams::new(
        at(1.3),
        TimeDelta::from_days(3.7).expect("positive"),
        RadiusRatio::new(0.05).expect("valid"),
        ScaledSemiMajorAxis::new(15.0).expect("valid"),
        ImpactParameter::new(0.5).expect("valid"),
        LimbDarkening::quadratic(0.4804, 0.1867).expect("valid"),
    )
    .expect("valid");
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a count"
    )]
    let n = (days / step) as u32;
    let time: Vec<f64> = (0..n).map(|i| f64::from(i) * step).collect();
    let mut state = 1u64;
    let flux = time
        .iter()
        .map(|&t| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[expect(clippy::cast_precision_loss, reason = "a uniform draw")]
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            p.flux_at(at(t)) + 1e-3 * (u - 0.5)
        })
        .collect();
    let lc = LightCurve::from_days(time, flux, vec![5e-4; n as usize]).expect("valid");
    let candidate =
        Candidate::new(p.period(), at(1.3), p.t14().expect("a transit")).expect("valid");
    (name, lc, candidate)
}

/// The two light curves.
#[must_use]
pub fn cases() -> Vec<Case> {
    vec![
        case("sector, 2 minutes", 27.0, 2.0 / 1440.0),
        case("four years, 30 minutes", 1461.0, 30.0 / 1440.0),
    ]
}

/// One time: the light curve, the step, the shape ("limb" or "box";
/// "limb" for `secondary`, which takes none) and seconds.
pub type Timing = (String, &'static str, &'static str, f64);

fn once(step: &str, lc: &LightCurve, c: &Candidate, shape: Shape) {
    match step {
        "fit_depth" => drop(black_box(fit_depth(lc, c, shape))),
        "refine" => drop(black_box(refine(lc, c, shape))),
        "evidence" => drop(black_box(evidence(lc, c, shape))),
        "signal_to_noise" => drop(black_box(signal_to_noise(lc, c, shape))),
        "aliases" => drop(black_box(aliases(lc, c, shape))),
        _ => drop(black_box(secondary(lc, c))),
    }
}

/// Every step timed: the best of `repeats` repeats, each the mean of as
/// many calls as take about `budget` seconds.
#[must_use]
pub fn run(cases: &[Case], repeats: usize, budget: f64) -> Vec<Timing> {
    let mut out = Vec::new();
    for (name, lc, c) in cases {
        for step in STEPS {
            let shapes: &[(&'static str, Shape)] = if step == "secondary" {
                &[("limb", Shape::BOX)]
            } else {
                &[("limb", Shape::default()), ("box", Shape::BOX)]
            };
            for &(shape_name, shape) in shapes {
                let start = Instant::now();
                once(step, lc, c, shape);
                let first = start.elapsed().as_secs_f64().max(1e-6);
                #[expect(clippy::cast_possible_truncation, reason = "a small count")]
                #[expect(clippy::cast_sign_loss, reason = "positive")]
                let number = ((budget / first) as u32).max(1);
                let secs = best(repeats, number, || once(step, lc, c, shape));
                out.push(((*name).to_owned(), step, shape_name, secs));
            }
        }
    }
    out
}

/// The limb-darkened `refine` on `threads`, as a step `refine threads`;
/// the best of `repeats` calls on each light curve.
#[must_use]
pub fn threaded(cases: &[Case], threads: NonZeroUsize, repeats: usize) -> Vec<Timing> {
    cases
        .iter()
        .map(|(name, lc, c)| {
            let secs = best(repeats, 1, || {
                refine_with_threads(lc, c, Shape::default(), threads)
            });
            ((*name).to_owned(), "refine threads", "limb", secs)
        })
        .collect()
}

/// The README's table: each step on each light curve and shape; then the
/// limb-darkened `refine` on more threads, if timed.
#[must_use]
pub fn report(cases: &[Case], timings: &[Timing], threads: NonZeroUsize) -> String {
    let mut out = String::from("| Step |");
    for (name, _, _) in cases {
        write!(out, " {name}, limb-darkened | {name}, box |").expect("a string");
    }
    out.push_str("\n|---|");
    out.push_str(&"--:|".repeat(2 * cases.len()));
    out.push('\n');
    for step in STEPS {
        write!(out, "| `{step}` |").expect("a string");
        for (name, _, _) in cases {
            for shape in ["limb", "box"] {
                let t = timings
                    .iter()
                    .find(|(n, s, sh, _)| n == name && *s == step && *sh == shape)
                    .map(|t| t.3);
                write!(out, " {} |", short(t)).expect("a string");
            }
        }
        out.push('\n');
    }
    let threaded: Vec<String> = cases
        .iter()
        .filter_map(|(name, _, _)| {
            timings
                .iter()
                .find(|(n, s, _, _)| n == name && *s == "refine threads")
                .map(|t| format!("{} on the {name}", short(Some(t.3))))
        })
        .collect();
    if !threaded.is_empty() {
        write!(
            out,
            "\nThe limb-darkened `refine` with `refine_with_threads` on {threads} threads: {}.\n",
            threaded.join(", ")
        )
        .expect("a string");
    }
    out
}
