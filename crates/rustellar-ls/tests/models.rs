//! Refined peaks and the fitted sinusoid against the same least squares at
//! 40 digits (mpmath, rounded once) on the six seeded cases of the
//! periodogram's reference: each case's three highest peaks with every
//! mean, refined to the power's maximum within a grid step, and the
//! sinusoid there with its standard errors; astropy 7.2's
//! `model_parameters` and the usual parabola through three powers beside.

use std::collections::BTreeMap;

use rustellar_ls::{Frequencies, LightCurve, LombScargle, Mean, Normalization};

const POINTS: &str = include_str!("reference/ls_points.csv");
const MODELS: &str = include_str!("reference/ls_models.csv");

/// Each case's light curve.
fn curves() -> BTreeMap<String, LightCurve> {
    let mut columns: BTreeMap<String, [Vec<f64>; 3]> = BTreeMap::new();
    for line in POINTS.lines().skip(1) {
        let mut f = line.split(',');
        let case = f.next().unwrap().to_string();
        let entry = columns.entry(case).or_default();
        for column in entry.iter_mut() {
            column.push(f.next().unwrap().parse().unwrap());
        }
    }
    columns
        .into_iter()
        .map(|(case, [t, y, dy])| (case, LightCurve::from_days(t, y, dy).unwrap()))
        .collect()
}

fn mean(name: &str) -> Mean {
    match name {
        "floating" => Mean::Floating,
        "subtracted" => Mean::Subtracted,
        _ => Mean::Zero,
    }
}

/// The reference's columns by name, parsed.
struct Row<'a> {
    names: Vec<&'a str>,
    fields: Vec<&'a str>,
}

impl Row<'static> {
    fn text(&self, name: &str) -> &'static str {
        self.fields[self.names.iter().position(|n| *n == name).unwrap()]
    }

    fn num(&self, name: &str) -> f64 {
        self.text(name).parse().unwrap()
    }
}

fn rows() -> Vec<Row<'static>> {
    let mut lines = MODELS.lines();
    let names: Vec<&str> = lines.next().unwrap().split(',').collect();
    lines
        .map(|line| Row {
            names: names.clone(),
            fields: line.split(',').collect(),
        })
        .collect()
}

/// `|a - b|` over `|b|`, or `|a - b|` where `b` is 0.
fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

/// The angle between two phases, in radians.
fn angle(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

#[test]
fn peaks_and_models_against_forty_digits() {
    let curves = curves();
    let mut worst: BTreeMap<&str, [f64; 4]> = BTreeMap::new();
    let mut ours = [0.0_f64; 7];
    let mut theirs = [0.0_f64; 2];
    for row in rows() {
        let case = row.text("case");
        let lc = &curves[case];
        let span = lc.time()[lc.len() - 1].as_days() - lc.time()[0].as_days();
        let ls = LombScargle::new(lc, mean(row.text("mean"))).unwrap();
        let nf = row.text("nf").parse().unwrap();
        let grid = Frequencies::uniform(row.num("f0"), row.num("df"), nf).unwrap();
        let periodogram = ls.power(&grid, Normalization::Standard);
        let want = row.num("f_peak");
        let peak = ls
            .peaks(&periodogram, nf)
            .into_iter()
            .min_by(|a, b| {
                (a.frequency() - want)
                    .abs()
                    .total_cmp(&(b.frequency() - want).abs())
            })
            .unwrap();
        let k: usize = row.text("k").parse().unwrap();
        // Errors in the frequency as a fraction of the peak's width, 1 / T.
        let w = worst.entry(case).or_default();
        let ours_f = (peak.frequency() - want).abs() * span;
        w[0] = w[0].max(ours_f);
        w[1] = w[1].max((row.num("parabola") - want).abs() * span);
        w[2] = w[2].max((grid.get(k) - want).abs() * span);
        w[3] = w[3].max((peak.power() - row.num("power")).abs());
        assert!(
            ours_f <= 1e-9 && (peak.power() - row.num("power")).abs() <= 1e-14,
            "{case} {} {k}: {} {want}",
            row.text("mean"),
            peak.frequency()
        );
        // The sinusoid at the exact peak.
        let model = ls.model(want).unwrap();
        let errors = [
            relative(model.offset(), row.num("offset")),
            relative(model.amplitude(), row.num("amplitude")),
            angle(model.phase(), row.num("phase")),
            relative(model.offset_error(), row.num("offset_err")),
            relative(model.amplitude_error(), row.num("amplitude_err")),
            relative(model.phase_error(), row.num("phase_err")),
            relative(model.cosine().hypot(model.sine()), row.num("amplitude")),
        ];
        for (o, e) in ours.iter_mut().zip(errors) {
            *o = o.max(e);
        }
        assert!(errors.iter().all(|&e| e <= 1e-9), "{case} {k}: {errors:?}");
        theirs[0] = theirs[0].max(relative(row.num("astropy_constant"), row.num("offset")));
        theirs[1] = theirs[1].max(relative(row.num("astropy_amplitude"), row.num("amplitude")));
    }
    println!("peak frequency error x T (ours, parabola, grid), power error:");
    for (case, w) in &worst {
        println!(
            "  {case:>8}: {:.0e} {:.0e} {:.0e} {:.0e}",
            w[0], w[1], w[2], w[3]
        );
    }
    println!(
        "model, ours: offset {:.0e}, amplitude {:.0e}, phase {:.0e}, errors {:.0e} {:.0e} {:.0e}",
        ours[0], ours[1], ours[2], ours[3], ours[4], ours[5]
    );
    println!(
        "model, astropy: offset {:.0e}, amplitude {:.0e}",
        theirs[0], theirs[1]
    );
}
