//! rustellar-tls against TLS 1.32 on the same injected light curves.

use std::fmt::Write as _;

use rustellar_core::Epoch;
use rustellar_tls::{OfirPeriod, SearchOptions, Tls};

use crate::curves::{Found, Kind, Row, light_curve, rows, transit};
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

/// Searches the curve of `row` in set `kind` with `options`, over Ofir's
/// grid with TLS's defaults.
///
/// # Panics
///
/// If the search fails, which a reference curve never makes it do.
#[must_use]
pub fn search(kind: Kind, row: &Row, options: &SearchOptions) -> Curve {
    let lc = light_curve(kind, row);
    let params = transit(row);
    let periods = OfirPeriod::default().periods(&lc).expect("a grid");
    let result = Tls::new(&lc)
        .expect("a light curve")
        .search(&periods, options)
        .expect("a search");
    let sde = result.spectra().map_or(0.0, |s| s.sde);
    let ours = result.best().map(|b| Found {
        period: b.period.as_days(),
        t0: b.transit_time.as_days(),
        depth: b.depth,
        duration: b.duration.as_days(),
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

/// Searches the first `limit` curves of set `kind`.
#[must_use]
pub fn run(kind: Kind, limit: usize, options: &SearchOptions) -> Vec<Curve> {
    rows(kind.reference())
        .iter()
        .take(limit)
        .map(|row| search(kind, row, options))
        .collect()
}

/// The comparison of a set as a Markdown table: how many transits each
/// finds, and on those both find, each one's median error and how often
/// ours is the closer, with the sign test's p-value.
#[must_use]
pub fn report(kind: Kind, curves: &[Curve]) -> String {
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
        kind.name(),
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
