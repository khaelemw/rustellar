//! rustellar-vet's alias table on the TLS sets' candidates: whether the
//! table points from a candidate at the wrong period to the true one.
//!
//! The candidates are transitleastsquares 1.32's, from the sets' reference
//! files, so no search is run. Where TLS found the injected period, the
//! candidate is also given at each other default ratio of it (a third to
//! three times), as a search locked onto an alias would give it; where TLS
//! itself found an alias, that candidate as it is. Each is then pointed
//! at the ratio of its table's lowest chi-squared, and, for comparison, at
//! that of its highest signal to noise, the usual choice between periods.

use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::thread;

use rustellar_core::{Epoch, TimeDelta};
use rustellar_vet::{Candidate, Ratio, Shape, aliases};

use crate::curves::{Kind, Row, light_curve, rows};

/// Where a case's candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// TLS's own candidate.
    Tls,
    /// TLS's candidate at the true period, moved to another ratio of it.
    Moved,
}

/// One candidate and where its alias table points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Case {
    /// The set.
    pub kind: Kind,
    /// The curve's seed.
    pub seed: u64,
    /// Where the candidate came from.
    pub source: Source,
    /// The candidate's period over the true one.
    pub given: Ratio,
    /// The ratio of the table's lowest chi-squared, if any row could be
    /// measured.
    pub by_chi2: Option<Ratio>,
    /// The signal to noise at the true period, if its row was measured.
    pub truth_snr: Option<f64>,
    /// Whether the candidate's transits fall on the true ones: its
    /// mid-transit time within a duration of a true transit, at the two
    /// periods' common spacing (`P / q` for a candidate at `P p / q`).
    pub aligned: bool,
}

impl Case {
    /// The ratio that leads back to the true period: the inverse of
    /// [`given`](Self::given).
    #[must_use]
    pub fn truth(&self) -> Ratio {
        Ratio::new(self.given.denominator(), self.given.numerator()).expect("not zero")
    }
}

/// The default ratio within 1% of `x`, if one is.
fn snap(x: f64) -> Option<Ratio> {
    Ratio::DEFAULT
        .into_iter()
        .find(|r| (x / r.value() - 1.0).abs() < 0.01)
}

/// The cases of one curve.
fn cases_of(kind: Kind, row: &Row) -> Vec<Case> {
    let Some(found) = snap(row.tls.period / row.period) else {
        return vec![];
    };
    let one = Ratio::new(1, 1).expect("one");
    let given: Vec<(Source, Ratio, f64)> = if found == one {
        Ratio::DEFAULT
            .into_iter()
            .map(|r| {
                let source = if r == one { Source::Tls } else { Source::Moved };
                (source, r, row.tls.period * r.value())
            })
            .collect()
    } else {
        vec![(Source::Tls, found, row.tls.period)]
    };
    let lc = light_curve(kind, row);
    given
        .into_iter()
        .filter_map(|(source, given, period)| {
            let candidate = Candidate::new(
                TimeDelta::from_days(period).ok()?,
                Epoch::from_btjd(row.tls.t0).ok()?,
                TimeDelta::from_days(row.tls.duration).ok()?,
            )
            .ok()?;
            let table = aliases(&lc, &candidate, Shape::default()).ok()?;
            let truth = Ratio::new(given.denominator(), given.numerator()).ok()?;
            let truth_snr = table
                .rows
                .iter()
                .find(|r| r.ratio == truth)
                .and_then(|r| r.alias.as_ref().ok())
                .map(|a| a.snr.white);
            let spacing = row.period / f64::from(given.denominator());
            let off = (row.tls.t0 - row.t0 + 0.5 * spacing).rem_euclid(spacing) - 0.5 * spacing;
            Some(Case {
                kind,
                seed: row.seed,
                source,
                given,
                by_chi2: table.best().map(|r| r.ratio),
                truth_snr,
                aligned: off.abs() < row.tls.duration,
            })
        })
        .collect()
}

/// The cases of the first `limit` curves of set `kind`, over `threads`
/// threads.
#[must_use]
pub fn cases(kind: Kind, limit: usize, threads: NonZeroUsize) -> Vec<Case> {
    let rows: Vec<Row> = rows(kind.reference()).into_iter().take(limit).collect();
    let chunk = rows.len().div_ceil(threads.get()).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .flat_map(|row| cases_of(kind, row))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a vetting thread panicked"))
            .collect()
    })
}

/// The median to one decimal, or a dash for none.
pub(crate) fn median_text(values: &[f64]) -> String {
    if values.is_empty() {
        return "-".to_owned();
    }
    format!("{:.1}", crate::stats::median(values))
}

/// `n` of `of` as a count and a percentage.
fn percent(n: usize, of: usize) -> String {
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let p = 100.0 * n as f64 / of as f64;
    format!("{n} ({p:.1}%)")
}

/// The study as Markdown tables: for each ratio of the true period a
/// candidate was given at, how often the table's lowest chi-squared leads
/// back to the true period; then the same by the signal to noise at the
/// true period.
#[must_use]
pub fn report(cases: &[Case]) -> String {
    let mut out = String::from(
        "| Candidate's period | Source | Cases | On the true transits | Pointed to the true period | Median SNR at the true period |\n|---|---|---|---|---|---|\n",
    );
    let right = |c: &&&Case| c.by_chi2 == Some(c.truth());
    let one = Ratio::new(1, 1).expect("one");
    for (source, name) in [(Source::Tls, "TLS 1.32"), (Source::Moved, "moved")] {
        for given in Ratio::DEFAULT {
            let picked: Vec<&Case> = cases
                .iter()
                .filter(|c| c.source == source && c.given == given)
                .collect();
            if picked.is_empty() {
                continue;
            }
            let label = if given == one {
                "true".to_owned()
            } else {
                format!("{given} × true")
            };
            let snr: Vec<f64> = picked.iter().filter_map(|c| c.truth_snr).collect();
            let median = median_text(&snr);
            let _ = writeln!(
                out,
                "| {label} | {name} | {} | {} | {} | {median} |",
                picked.len(),
                picked.iter().filter(|c| c.aligned).count(),
                percent(picked.iter().filter(right).count(), picked.len()),
            );
        }
    }
    out.push_str(
        "\n| SNR at the true period | Cases (true and moved, on the true transits) | Pointed to the true period |\n|---|---|---|\n",
    );
    let bands: [(f64, f64, &str); 5] = [
        (f64::NEG_INFINITY, 7.0, "below 7"),
        (7.0, 10.0, "7 to 10"),
        (10.0, 20.0, "10 to 20"),
        (20.0, 50.0, "20 to 50"),
        (50.0, f64::INFINITY, "50 and above"),
    ];
    for (lo, hi, label) in bands {
        let picked: Vec<&Case> = cases
            .iter()
            .filter(|c| c.aligned && c.truth_snr.is_some_and(|s| lo <= s && s < hi))
            .collect();
        if picked.is_empty() {
            continue;
        }
        let _ = writeln!(
            out,
            "| {label} | {} | {} |",
            picked.len(),
            percent(picked.iter().filter(right).count(), picked.len()),
        );
    }
    out
}
