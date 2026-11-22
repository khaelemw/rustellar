//! rustellar-tls against TLS 1.32 on the same injected light curves.

use std::fmt::Write as _;

use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_detrend::{BreakTolerance, Candidate, Detrend, Masking};
use rustellar_tls::{OfirPeriod, Periods, SearchOptions, Tls, TlsError, TlsResult};

use crate::curves::{Found, Kind, Row, light_curve, raw, rows, transit};
use crate::stats::{median, sign_test, timing};

/// `x` (zero or above) to three significant figures, in plain notation from
/// 0.001 up.
pub(crate) fn figures(x: f64) -> String {
    if x < 1e-3 {
        return format!("{x:.2e}");
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a small count of decimals"
    )]
    let decimals = (2.0 - x.log10().floor()).max(0.0) as usize;
    format!("{x:.decimals$}")
}

/// A period within 1% of the injected one.
fn recovered(found: &Found, row: &Row) -> bool {
    (found.period / row.period - 1.0).abs() < 0.01
}

/// One curve: its row, the truth's depth, duration and span, and ours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Curve {
    /// The injected transit and TLS's result.
    pub row: Row,
    /// The injected depth at mid-transit, as a fraction of the flux.
    pub depth: f64,
    /// The injected duration, first to fourth contact, in days.
    pub duration: f64,
    /// The light curve's last time, in days.
    pub span: f64,
    /// Our best candidate, or `None` where nothing dims.
    pub ours: Option<Found>,
}

/// The curve of `row` in set `kind`, searched and described by `result`.
///
/// # Panics
///
/// If the search failed, which a reference curve never makes it do.
fn curve(kind: Kind, row: &Row, result: Result<TlsResult, TlsError>) -> Curve {
    curve_of(&light_curve(kind, row), row, result)
}

/// The curve of `row`, `lc` as searched, described by `result`.
fn curve_of(lc: &LightCurve, row: &Row, result: Result<TlsResult, TlsError>) -> Curve {
    let params = transit(row);
    let result = result.expect("a search");
    let sde = result.spectra().map_or(0.0, |s| s.sde);
    let ours = result.best().map(|b| Found {
        period: b.period.as_days(),
        t0: b.transit_time.as_days(),
        depth: b.depth,
        duration: b.duration_estimate().as_days(),
        sde,
    });
    Curve {
        row: *row,
        depth: 1.0 - params.flux_at(Epoch::from_days(row.t0).expect("finite")),
        duration: params.t14().expect("a transit").as_days(),
        span: lc.time()[lc.time().len() - 1].as_days(),
        ours,
    }
}

/// Searches the first `limit` curves of set `kind` with `options` (all of
/// them at once, with `Tls::search_many`), each over Ofir's grid with
/// TLS's defaults.
///
/// # Panics
///
/// If a search fails, which a reference curve never makes it do.
#[must_use]
pub fn run(kind: Kind, limit: usize, options: &SearchOptions) -> Vec<Curve> {
    let rows: Vec<Row> = rows(kind.reference()).into_iter().take(limit).collect();
    let jobs: Vec<(Tls, Periods)> = rows
        .iter()
        .map(|row| {
            let lc = light_curve(kind, row);
            let periods = OfirPeriod::default().periods(&lc).expect("a grid");
            (Tls::new(&lc).expect("a light curve"), periods)
        })
        .collect();
    rows.iter()
        .zip(Tls::search_many(&jobs, options))
        .map(|(row, result)| curve(kind, row, result))
        .collect()
}

/// The `Variable` set searched again around its candidates: each curve
/// flattened as [`run`] flattens it and searched, then flattened again
/// with the candidate's transits masked (`Detrend::flatten_around`, its
/// default masking) and searched again. A curve whose first search finds
/// nothing keeps its first flattening.
///
/// # Panics
///
/// If a search or a flattening fails, which a reference curve never makes
/// them do.
#[must_use]
pub fn run_around(limit: usize, options: &SearchOptions) -> Vec<Curve> {
    let kind = Kind::Variable;
    let rows: Vec<Row> = rows(kind.reference()).into_iter().take(limit).collect();
    let job = |lc: &LightCurve| {
        let periods = OfirPeriod::default().periods(lc).expect("a grid");
        (Tls::new(lc).expect("a light curve"), periods)
    };
    let blind: Vec<(Tls, Periods)> = rows.iter().map(|r| job(&light_curve(kind, r))).collect();
    let flattened: Vec<LightCurve> = rows
        .iter()
        .zip(Tls::search_many(&blind, options))
        .map(|(row, result)| {
            let found = result.expect("a search").best().map(|b| Candidate {
                period: b.period,
                t0: b.transit_time,
                duration: b.duration_estimate(),
            });
            let window = TimeDelta::from_days(row.window).expect("finite");
            let half = TimeDelta::from_days(row.window / 2.0).expect("finite");
            Detrend::default()
                .with_window(window)
                .with_break_tolerance(BreakTolerance::Fixed(half))
                .flatten_around(&raw(kind, row), &Masking::default(), |_| found)
                .expect("detrended")
                .flattened()
                .light_curve()
                .clone()
        })
        .collect();
    let jobs: Vec<(Tls, Periods)> = flattened.iter().map(job).collect();
    rows.iter()
        .zip(&flattened)
        .zip(Tls::search_many(&jobs, options))
        .map(|((row, lc), result)| curve_of(lc, row, result))
        .collect()
}

/// The comparison of a set as a Markdown table: how many transits each
/// finds, and on those both find, each one's median error and how often
/// ours is the closer, with the sign test's p-value.
#[must_use]
pub fn report(name: &str, curves: &[Curve]) -> String {
    let ours = |c: &Curve| c.ours.filter(|f| recovered(f, &c.row));
    let found_ours = curves.iter().filter(|c| ours(c).is_some()).count();
    let found_tls = curves
        .iter()
        .filter(|c| recovered(&c.row.tls, &c.row))
        .count();
    let both: Vec<(&Curve, Found)> = curves
        .iter()
        .filter_map(|c| {
            ours(c)
                .filter(|_| recovered(&c.row.tls, &c.row))
                .map(|f| (c, f))
        })
        .collect();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "### {} ({} curves)\n\nFound within 1% of the period: rustellar-tls {found_ours}, TLS 1.32 {found_tls}, both {}.\n",
        name,
        curves.len(),
        both.len()
    );
    if both.is_empty() {
        return out;
    }
    let _ = writeln!(
        out,
        "| On the {} both find | rustellar-tls median | TLS 1.32 median | rustellar-tls closer | sign test p |\n|---|---|---|---|---|",
        both.len()
    );
    type Error = fn(&Curve, &Found) -> f64;
    let measures: [(&str, Error); 4] = [
        ("period, relative", |c, f| {
            (f.period / c.row.period - 1.0).abs()
        }),
        ("mid-transit times, minutes", |c, f| {
            timing(c.row.period, c.row.t0, f.period, f.t0, c.span)
        }),
        ("depth, relative", |c, f| (f.depth / c.depth - 1.0).abs()),
        ("duration, relative", |c, f| {
            (f.duration / c.duration - 1.0).abs()
        }),
    ];
    for (name, error) in measures {
        let a: Vec<f64> = both.iter().map(|(c, f)| error(c, f)).collect();
        let b: Vec<f64> = both.iter().map(|(c, _)| error(c, &c.row.tls)).collect();
        let wins = a.iter().zip(&b).filter(|(x, y)| x < y).count();
        let _ = writeln!(
            out,
            "| {name} | {} | {} | {wins}/{} | {:.2} |",
            figures(median(&a)),
            figures(median(&b)),
            a.len(),
            sign_test(wins, a.len())
        );
    }
    out
}
