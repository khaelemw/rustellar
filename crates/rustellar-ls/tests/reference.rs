//! The exact periodogram against the same least squares at 40 digits
//! (mpmath, rounded once) on six seeded cases: uneven sampling with a
//! sinusoid in noise, the same at BJD-sized times, a near-perfect sinusoid,
//! frequencies far below 1 / T, even sampling through its Nyquist
//! frequency, and five points; each with every mean and normalisation.
//! astropy 7.2's slow, cython and fast methods, on the same frequencies,
//! are measured beside it (standard normalisation).

use std::collections::BTreeMap;

use rustellar_ls::{Frequencies, LightCurve, LombScargle, Mean, Normalization};

const POINTS: &str = include_str!("reference/ls_points.csv");
const POWERS: &str = include_str!("reference/ls_powers.csv");

const NORMALIZATIONS: [Normalization; 4] = [
    Normalization::Standard,
    Normalization::Model,
    Normalization::Log,
    Normalization::Psd,
];

struct Row {
    reduction: f64,
    chi2: f64,
    chi2_ref: f64,
    /// How close the fit's terms come to depending on each other.
    independence: f64,
    /// astropy slow, cython and fast, standard normalisation.
    astropy: [f64; 3],
}

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

/// Each (case, mean)'s grid and rows.
fn rows() -> BTreeMap<(String, String), (f64, f64, Vec<Row>)> {
    let mut out: BTreeMap<(String, String), (f64, f64, Vec<Row>)> = BTreeMap::new();
    for line in POWERS.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let num = |k: usize| f[k].parse::<f64>().unwrap();
        let entry =
            out.entry((f[0].to_string(), f[1].to_string()))
                .or_insert((num(2), num(3), Vec::new()));
        entry.2.push(Row {
            reduction: num(5),
            chi2: num(6),
            chi2_ref: num(7),
            independence: num(8),
            astropy: [num(9), num(10), num(11)],
        });
    }
    out
}

fn mean(name: &str) -> Mean {
    match name {
        "floating" => Mean::Floating,
        "subtracted" => Mean::Subtracted,
        _ => Mean::Zero,
    }
}

/// The exact power of a row, rounded once.
fn exact(row: &Row, normalization: Normalization) -> f64 {
    match normalization {
        Normalization::Standard => row.reduction / row.chi2_ref,
        Normalization::Model => row.reduction / row.chi2,
        Normalization::Log => (row.reduction / row.chi2).ln_1p(),
        _ => 0.5 * row.reduction,
    }
}

/// How far `got` may be from `want`: ten roundings of the periodogram's
/// peak at any frequency; for the model and log normalisations, which
/// divide by the chi-squared left, also as many roundings of the flux as
/// that is smaller than the reference's (a nearly perfect fit's residuals
/// are differences of nearly equal numbers).
fn allowed(row: &Row, normalization: Normalization, want: f64, peak: f64) -> f64 {
    let base = 1e-14 * peak;
    match normalization {
        Normalization::Model | Normalization::Log => {
            base + 8.0 * f64::EPSILON * (row.chi2_ref / row.chi2).sqrt() * want
        }
        _ => base,
    }
}

/// Terms degenerate to f64 rounding: the exact fit depends on digits no
/// `f64` computation of the terms holds.
fn degenerate(row: &Row) -> bool {
    row.independence < 1e-12
}

#[test]
fn exact_against_forty_digits_and_astropy() {
    let curves = curves();
    let mut skipped = 0;
    println!("max error / peak, standard normalisation:");
    println!(
        "{:<10} {:<10} {:>10} {:>10} {:>10} {:>10}",
        "case", "mean", "ours", "slow", "cython", "fast"
    );
    for ((case, mean_name), (f0, df, rows)) in rows() {
        let ls = LombScargle::new(&curves[&case], mean(&mean_name)).unwrap();
        assert_eq!(ls.mean(), mean(&mean_name));
        let want_ref = rows[0].chi2_ref;
        assert!(
            (ls.chi2_ref() - want_ref).abs() <= 1e-15 * want_ref,
            "{case}"
        );
        let grid = Frequencies::uniform(f0, df, rows.len()).unwrap();
        skipped += rows.iter().filter(|r| degenerate(r)).count();
        for normalization in NORMALIZATIONS {
            let p = ls.power(&grid, normalization);
            let want: Vec<f64> = rows.iter().map(|r| exact(r, normalization)).collect();
            let peak = want.iter().copied().fold(0.0, f64::max);
            for (k, row) in rows.iter().enumerate() {
                let (got, want) = (p.power()[k], want[k]);
                if !degenerate(row) {
                    assert!(
                        (got - want).abs() <= allowed(row, normalization, want, peak),
                        "{case} {mean_name} {normalization:?} {k}: {got:e} {want:e}"
                    );
                }
            }
            if normalization == Normalization::Standard {
                // Each method's largest error over the peak, the degenerate
                // frequencies left out.
                let worst = |value: &dyn Fn(usize) -> f64| {
                    (0..rows.len())
                        .filter(|&k| !degenerate(&rows[k]))
                        .map(|k| {
                            let e = (value(k) - want[k]).abs() / peak;
                            if e.is_nan() { f64::INFINITY } else { e }
                        })
                        .fold(0.0, f64::max)
                };
                let ours = worst(&|k| p.power()[k]);
                let theirs: Vec<f64> = (0..3).map(|m| worst(&|k| rows[k].astropy[m])).collect();
                println!(
                    "{case:<10} {mean_name:<10} {ours:>10.1e} {:>10.1e} {:>10.1e} {:>10.1e}",
                    theirs[0], theirs[1], theirs[2]
                );
                // As accurate as astropy's most accurate method, or more.
                let best = theirs.iter().copied().fold(f64::INFINITY, f64::min);
                assert!(ours <= best.max(1e-15), "{case} {mean_name}");
            }
        }
    }
    // Only five points at 2.5 cycles per day, where the cosine is 1 less
    // the sine to rounding.
    assert_eq!(skipped, 1);
}

#[test]
fn fast_within_its_tolerance_of_forty_digits() {
    let curves = curves();
    println!("fast method, max error (standard), floating mean:");
    for ((case, mean_name), (f0, df, rows)) in rows() {
        let ls = LombScargle::new(&curves[&case], mean(&mean_name)).unwrap();
        let grid = Frequencies::uniform(f0, df, rows.len()).unwrap();
        for tolerance in [1e-4, 1e-8, 1e-12] {
            let mut worst = 0.0_f64;
            for normalization in [Normalization::Standard, Normalization::Psd] {
                let p = ls.power_fast(&grid, normalization, tolerance).unwrap();
                for (k, row) in rows.iter().enumerate() {
                    if degenerate(row) {
                        continue;
                    }
                    let want = exact(row, normalization);
                    // The tolerance is on the standard power; the PSD's is
                    // the same times half the reference chi-squared.
                    let scale = match normalization {
                        Normalization::Psd => 0.5 * row.chi2_ref,
                        _ => 1.0,
                    };
                    let got = p.power()[k];
                    if normalization == Normalization::Standard {
                        worst = worst.max((got - want).abs());
                    }
                    assert!(
                        (got - want).abs()
                            <= tolerance * scale + 1e-14 * want.abs().max(scale * 1e-2),
                        "{case} {mean_name} {tolerance:e} {k}: {got:e} {want:e}"
                    );
                }
            }
            if mean_name == "floating" {
                println!("  {case:>8} tolerance {tolerance:e}: {worst:.0e}");
            }
        }
    }
}
