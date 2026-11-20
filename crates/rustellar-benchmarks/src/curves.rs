//! The injected light curves the references were run on, built again from
//! their seeds: the same times, errors and noise, and the same transit.

use std::f64::consts::PI;

use rustellar_core::{Epoch, LightCurve, TimeDelta};
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
}

impl Kind {
    /// Every set, in order.
    pub const ALL: [Self; 5] = [
        Self::Many,
        Self::Mixed,
        Self::Varying,
        Self::Short,
        Self::Sectors,
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
    lines
        .map(|line| {
            let v: Vec<&str> = line.split(',').collect();
            let f = |i: usize| v[at[i]].parse::<f64>().expect("a number");
            Row {
                seed: v[at[0]].parse().expect("a seed"),
                k: f(1),
                period: f(2),
                t0: f(3),
                noise: f(4),
                sparse: sparse.is_some_and(|i| v[i] == "1"),
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

/// The injected transit: radius ratio `k`, a = 12, b = 0.2, quadratic limb
/// darkening (0.4804, 0.1867).
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
        ScaledSemiMajorAxis::new(12.0).expect("12"),
        ImpactParameter::new(0.2).expect("0.2"),
        LimbDarkening::quadratic(0.4804, 0.1867).expect("TLS's default"),
    )
    .expect("a transit")
}

/// The curve of `row` in set `kind`, as the reference data were built:
/// per point, noise from four uniforms of SplitMix64(seed), and for a
/// sparse curve one more uniform, the point dropped below 0.25.
///
/// # Panics
///
/// If the curve is not a light curve.
#[must_use]
pub fn light_curve(kind: Kind, row: &Row) -> LightCurve {
    let params = transit(row);
    let mut random = Random::new(row.seed);
    let (mut time, mut flux, mut error) = (vec![], vec![], vec![]);
    for (t, e) in kind.times_and_errors(row.noise) {
        let n = random.noise();
        if row.sparse && random.uniform() < 0.25 {
            continue;
        }
        time.push(t);
        flux.push(params.flux_at(Epoch::from_days(t).expect("finite")) + e * n);
        error.push(e);
    }
    LightCurve::from_days(time, flux, error).expect("a light curve")
}
