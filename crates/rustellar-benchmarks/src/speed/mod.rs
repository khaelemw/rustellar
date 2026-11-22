//! Speed against the Python references. Each crate's cases are timed here
//! as the references were timed by hand-run scripts on the same machine:
//! the best of seven repeats, each the mean of a loop of calls. The
//! references' times are read from `data/speed_<crate>.csv`, which records
//! each package's name and version beside its time.

use std::hint::black_box;
use std::time::Instant;

pub mod bls;
pub mod detrend;
pub mod fit;
pub mod ls;
pub mod mcmc;
pub mod model;
pub mod tls;
pub mod vet;

/// One reference time: a case, the package (with its version) and seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub case: String,
    pub package: String,
    pub seconds: f64,
}

/// The rows of a `case,package,seconds` file, header first.
///
/// # Panics
///
/// On a row that is not three fields with a number last.
#[must_use]
pub fn references(csv: &str) -> Vec<Reference> {
    csv.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 3, "a row of three fields: {line}");
            Reference {
                case: fields[0].to_owned(),
                package: fields[1].to_owned(),
                seconds: fields[2].trim().parse().expect("seconds"),
            }
        })
        .collect()
}

fn first_word(package: &str) -> &str {
    package.split(' ').next().unwrap_or(package)
}

/// The package named `name` in `references`, with its version
/// (`batman 2.5.1` for `batman`), or `name` alone if it is not there.
#[must_use]
pub fn package(references: &[Reference], name: &str) -> String {
    references
        .iter()
        .find(|r| first_word(&r.package) == name)
        .map_or_else(|| name.to_owned(), |r| r.package.clone())
}

/// The seconds `name` took on `case`, if recorded.
#[must_use]
pub fn seconds(references: &[Reference], name: &str, case: &str) -> Option<f64> {
    references
        .iter()
        .find(|r| r.case == case && first_word(&r.package) == name)
        .map(|r| r.seconds)
}

/// Seconds per call of `f`: the best of `repeats` repeats, each the mean of
/// `number` calls, as Python's `min(timeit.repeat(...)) / number`.
pub fn best<T>(repeats: usize, number: u32, mut f: impl FnMut() -> T) -> f64 {
    (0..repeats)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..number {
                black_box(f());
            }
            start.elapsed().as_secs_f64() / f64::from(number)
        })
        .fold(f64::INFINITY, f64::min)
}

/// `seconds` to two significant figures, in microseconds below 0.1 ms,
/// milliseconds below 0.1 s and seconds above: `4.0 µs`, `0.64 ms`. A dash
/// where there is no time.
#[must_use]
pub fn short(seconds: Option<f64>) -> String {
    let Some(s) = seconds else {
        return "-".to_owned();
    };
    let (value, unit) = if s < 1e-4 {
        (s * 1e6, " µs")
    } else if s < 0.1 {
        (s * 1e3, " ms")
    } else {
        (s, " s")
    };
    format!("{}{unit}", two_figures(value))
}

/// A positive `value` to two significant figures, with at least one figure
/// before the point: `4.0`, `40`, `0.64`, `120`.
#[must_use]
pub fn two_figures(value: f64) -> String {
    significant(value, 2)
}

/// A positive `value` to `figures` significant figures (at least one), with
/// at least one figure before the point: `0.0104`, `10.6`, `185`.
#[must_use]
pub fn significant(value: f64, figures: i32) -> String {
    let decimals = |v: f64| {
        #[expect(clippy::cast_possible_truncation, reason = "a small exponent")]
        let exponent = v.log10().floor() as i32;
        usize::try_from(figures.max(1) - 1 - exponent).unwrap_or(0)
    };
    // Rounded first, so 9.96 becomes 10 with no decimal.
    let rounded: f64 = format!("{value:.*}", decimals(value))
        .parse()
        .expect("a number");
    format!("{rounded:.*}", decimals(rounded))
}
