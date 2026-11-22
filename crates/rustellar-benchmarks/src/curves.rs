//! The injected light curves the references were run on, built again from
//! their seeds: the same times, errors and noise, and the same transit.

use std::f64::consts::PI;

use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_detrend::{BreakTolerance, Detrend};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

use crate::random::Random;

/// A set of injected light curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 27 days at 30 minutes; in 30% of the curves, a quarter of the points
    /// dropped at random.
    Many,
    /// 13.5 days at 30 minutes, then from day 14 to 27 at 10 minutes.
    Mixed,
    /// 27 days at 30 minutes, errors from 1 to 3 times the noise.
    Varying,
    /// 27 days at 2 minutes.
    Short,
    /// Two sectors at 2 minutes, 54 days less two one-day gaps.
    Sectors,
    /// Periods of 9 to 13.5 days in 27 days: two or three transits.
    Long,
    /// Periods of 0.25 to 1 day, 10 days at 10 minutes.
    Ultrashort,
    /// Radius ratios of 0.1 to 0.2.
    Deep,
    /// The planet's disc across the star's limb.
    Grazing,
    /// Eccentricities of 0.1 to 0.7.
    Eccentric,
    /// 6 days at 20 seconds.
    Fast,
    /// Three 27-day blocks over 108 days, one missing, each with a gap.
    Gaps,
    /// Flagged points (dropped), outliers and NaNs.
    Flagged,
    /// A spotted star and red noise, detrended before the search.
    Variable,
}

impl Kind {
    /// Every set, in order.
    pub const ALL: [Self; 14] = [
        Self::Many,
        Self::Mixed,
        Self::Varying,
        Self::Short,
        Self::Sectors,
        Self::Long,
        Self::Ultrashort,
        Self::Deep,
        Self::Grazing,
        Self::Eccentric,
        Self::Fast,
        Self::Gaps,
        Self::Flagged,
        Self::Variable,
    ];

    /// Its name, as in its file `data/tls_<name>.csv`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Many => "many",
            Self::Mixed => "mixed",
            Self::Varying => "varying",
            Self::Short => "short",
            Self::Sectors => "sectors",
            Self::Long => "long",
            Self::Ultrashort => "ultrashort",
            Self::Deep => "deep",
            Self::Grazing => "grazing",
            Self::Eccentric => "eccentric",
            Self::Fast => "fast",
            Self::Gaps => "gaps",
            Self::Flagged => "flagged",
            Self::Variable => "variable",
        }
    }

    /// The set of this name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// TLS 1.32's results on the set, with each curve's parameters.
    #[must_use]
    pub fn reference(self) -> &'static str {
        match self {
            Self::Many => include_str!("../data/tls_many.csv"),
            Self::Mixed => include_str!("../data/tls_mixed.csv"),
            Self::Varying => include_str!("../data/tls_varying.csv"),
            Self::Short => include_str!("../data/tls_short.csv"),
            Self::Sectors => include_str!("../data/tls_sectors.csv"),
            Self::Long => include_str!("../data/tls_long.csv"),
            Self::Ultrashort => include_str!("../data/tls_ultrashort.csv"),
            Self::Deep => include_str!("../data/tls_deep.csv"),
            Self::Grazing => include_str!("../data/tls_grazing.csv"),
            Self::Eccentric => include_str!("../data/tls_eccentric.csv"),
            Self::Fast => include_str!("../data/tls_fast.csv"),
            Self::Gaps => include_str!("../data/tls_gaps.csv"),
            Self::Flagged => include_str!("../data/tls_flagged.csv"),
            Self::Variable => include_str!("../data/tls_variable.csv"),
        }
    }

    /// The times and errors of a curve with this `noise`.
    fn times_and_errors(self, noise: f64) -> Vec<(f64, f64)> {
        match self {
            Self::Many => (0..1296).map(|i| (f64::from(i) / 48.0, noise)).collect(),
            Self::Mixed => (0..648)
                .map(|i| (f64::from(i) / 48.0, noise))
                .chain((0..1872).map(|i| (14.0 + f64::from(i) / 144.0, noise * 3_f64.sqrt())))
                .collect(),
            Self::Varying => (0..1296)
                .map(|i| {
                    let t = f64::from(i) / 48.0;
                    (t, noise * (2.0 - (2.0 * PI * t / 13.5).cos()))
                })
                .collect(),
            Self::Short => (0..27 * 720)
                .map(|i| (f64::from(i) / 720.0, noise * 15_f64.sqrt()))
                .collect(),
            Self::Sectors => (0..54 * 720)
                .map(|i| f64::from(i) / 720.0)
                .filter(|t| !(13.0..14.0).contains(t) && !(40.0..41.0).contains(t))
                .map(|t| (t, noise * 15_f64.sqrt()))
                .collect(),
            Self::Ultrashort => (0..1440)
                .map(|i| (f64::from(i) / 144.0, noise * 3_f64.sqrt()))
                .collect(),
            Self::Fast => (0..25_920)
                .map(|i| (f64::from(i) / 4320.0, noise * 90_f64.sqrt()))
                .collect(),
            Self::Gaps => [0.0, 54.0, 81.0]
                .into_iter()
                .flat_map(|start| {
                    (0..1296)
                        .map(|i| f64::from(i) / 48.0)
                        .filter(|t| !(13.0..14.0).contains(t))
                        .map(move |t| (start + t, noise))
                })
                .collect(),
            Self::Long
            | Self::Deep
            | Self::Grazing
            | Self::Eccentric
            | Self::Flagged
            | Self::Variable => (0..1296).map(|i| (f64::from(i) / 48.0, noise)).collect(),
        }
    }
}

/// One curve of a set: its injected transit and TLS 1.32's result on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    /// The seed of the curve's noise.
    pub seed: u64,
    /// The planet's radius over the star's.
    pub k: f64,
    /// The injected period, in days.
    pub period: f64,
    /// The injected mid-transit time, in days.
    pub t0: f64,
    /// The noise per point at 30 minutes.
    pub noise: f64,
    /// Whether a quarter of the points were dropped (`Many` only).
    pub sparse: bool,
    /// The impact parameter at mid-transit (0.2 where the file has none).
    pub b: f64,
    /// The semi-major axis over the star's radius (12 where none).
    pub a: f64,
    /// The eccentricity (0 where none).
    pub e: f64,
    /// The argument of periastron, in degrees (90 where none).
    pub w: f64,
    /// The star's spots: amplitude, rotation period in days and phase
    /// (`Variable` only; else zero).
    pub spots: (f64, f64, f64),
    /// The detrending window in days (`Variable` only; else zero).
    pub window: f64,
    /// TLS's period, mid-transit time, depth, duration and SDE.
    pub tls: Found,
}

/// A search's best candidate: period, mid-transit time, depth (as a
/// fraction of the flux) and duration in days, and its SDE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Found {
    /// The period, in days.
    pub period: f64,
    /// A mid-transit time, in days.
    pub t0: f64,
    /// The depth, as a fraction of the flux.
    pub depth: f64,
    /// The duration, in days.
    pub duration: f64,
    /// The signal detection efficiency.
    pub sde: f64,
}

/// The rows of a reference file: a header line naming the columns, then a
/// line per curve.
///
/// # Panics
///
/// If a column is missing or a value does not parse.
#[must_use]
pub fn rows(csv: &str) -> Vec<Row> {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("a header").split(',').collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("no column {name}"))
    };
    let names = [
        "seed",
        "k",
        "period",
        "t0",
        "noise",
        "tls_period",
        "tls_t0",
        "tls_depth",
        "tls_duration",
        "tls_sde",
    ];
    let at = names.map(column);
    let sparse = header.iter().position(|h| *h == "sparse");
    let optional = |name: &str| header.iter().position(|h| *h == name);
    let extra = ["b", "a", "e", "w", "amp", "prot", "phase", "window"].map(optional);
    let defaults = [0.2, 12.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0];
    lines
        .map(|line| {
            let v: Vec<&str> = line.split(',').collect();
            let f = |i: usize| v[at[i]].parse::<f64>().expect("a number");
            let x =
                |i: usize| extra[i].map_or(defaults[i], |c| v[c].parse::<f64>().expect("a number"));
            Row {
                seed: v[at[0]].parse().expect("a seed"),
                k: f(1),
                period: f(2),
                t0: f(3),
                noise: f(4),
                sparse: sparse.is_some_and(|i| v[i] == "1"),
                b: x(0),
                a: x(1),
                e: x(2),
                w: x(3),
                spots: (x(4), x(5), x(6)),
                window: x(7),
                tls: Found {
                    period: f(5),
                    t0: f(6),
                    depth: f(7),
                    duration: f(8),
                    sde: f(9),
                },
            }
        })
        .collect()
}

/// The injected transit: radius ratio `k`, the row's `a`, `b` and orbit,
/// quadratic limb darkening (0.4804, 0.1867).
///
/// # Panics
///
/// If the row's values are not a transit.
#[must_use]
pub fn transit(row: &Row) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(row.t0).expect("finite"),
        TimeDelta::from_days(row.period).expect("finite"),
        RadiusRatio::new(row.k).expect("a radius ratio"),
        ScaledSemiMajorAxis::new(row.a).expect("a semi-major axis"),
        ImpactParameter::new(row.b).expect("an impact parameter"),
        LimbDarkening::quadratic(0.4804, 0.1867).expect("TLS's default"),
    )
    .and_then(|p| p.with_eccentricity(row.e, row.w))
    .expect("a transit")
}

/// The curve of `row` in set `kind` that the search is given, as the
/// reference data were built: per point, noise from four uniforms of
/// SplitMix64(seed), then
///
/// - for a sparse curve one more uniform, the point dropped below 0.25;
/// - for `Flagged` one more uniform `u`: below 0.02 the point is flagged
///   (dropped), from 0.02 to 0.03 an outlier of 5 to 20 errors either way
///   (two more uniforms), from 0.03 to 0.04 NaN (dropped);
/// - for `Variable` the transit times the spotted star, plus red noise
///   `r_i = 0.8 r_(i-1) + 0.6 noise n_i` from four more uniforms, then
///   detrended by the biweight over the row's window, break tolerance half
///   of it.
///
/// # Panics
///
/// If the curve is not a light curve.
#[must_use]
pub fn light_curve(kind: Kind, row: &Row) -> LightCurve {
    let lc = raw(kind, row);
    if kind != Kind::Variable {
        return lc;
    }
    let window = TimeDelta::from_days(row.window).expect("finite");
    let half = TimeDelta::from_days(row.window / 2.0).expect("finite");
    Detrend::default()
        .with_window(window)
        .with_break_tolerance(BreakTolerance::Fixed(half))
        .flatten(&lc, None)
        .expect("detrended")
        .light_curve()
        .clone()
}

/// The curve before any detrending.
pub(crate) fn raw(kind: Kind, row: &Row) -> LightCurve {
    let params = transit(row);
    let mut random = Random::new(row.seed);
    let (amp, prot, phase) = row.spots;
    let mut red = 0.0;
    let (mut time, mut flux, mut error) = (vec![], vec![], vec![]);
    for (i, (t, e)) in kind.times_and_errors(row.noise).into_iter().enumerate() {
        let model = params.flux_at(Epoch::from_days(t).expect("finite"));
        let white = e * random.noise();
        let mut f = model + white;
        if row.sparse && random.uniform() < 0.25 {
            continue;
        }
        match kind {
            Kind::Flagged => {
                let u = random.uniform();
                if u < 0.02 || (0.03..0.04).contains(&u) {
                    continue;
                }
                if u < 0.03 {
                    let size = 15.0_f64.mul_add(random.uniform(), 5.0);
                    let sign = if random.uniform() < 0.5 { -1.0 } else { 1.0 };
                    f += sign * e * size;
                }
            }
            Kind::Variable => {
                let angle = 2.0 * PI * t / prot + phase;
                let star = amp.mul_add(0.3_f64.mul_add((2.0 * angle).sin(), angle.sin()), 1.0);
                let n = row.noise * random.noise();
                red = if i == 0 {
                    n
                } else {
                    0.8_f64.mul_add(red, 0.6 * n)
                };
                f = model.mul_add(star, white) + red;
            }
            _ => {}
        }
        time.push(t);
        flux.push(f);
        error.push(e);
    }
    LightCurve::from_days(time, flux, error).expect("a light curve")
}
