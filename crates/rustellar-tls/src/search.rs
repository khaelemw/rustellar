//! The whole search: durations, the fast search at every period, the
//! spectra, and the highest peaks refined exactly.

use std::num::NonZeroUsize;

use rustellar_core::LightCurve;

use crate::durations::{self, Constants, Stars};
use crate::fast::{PeriodBest, Search, Settings, Table};
use crate::fit::{Points, Position};
use crate::refine::{Reach, Refined, refine};
use crate::spectra::{Spectra, period_uncertainty, spectra};
use crate::template::Template;

/// How many grid periods either way of a peak its refinement also starts
/// from.
const NEIGHBOURS: usize = 3;

/// How the search is run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Options {
    pub(crate) template: Template,
    /// Intervals in the template's lookup table.
    pub(crate) table_samples: usize,
    /// Trial mid-transit times per duration in the fast search.
    pub(crate) steps: usize,
    /// The smallest depth a fit may have to count.
    pub(crate) min_depth: f64,
    pub(crate) stars: Stars,
    pub(crate) constants: Constants,
    /// Each trial duration this factor times the one before.
    pub(crate) duration_step: f64,
    /// The period grid's oversampling factor (for the running median).
    pub(crate) oversampling: usize,
    /// How many of the highest peaks are refined.
    pub(crate) refine_peaks: usize,
    pub(crate) threads: NonZeroUsize,
}

impl Options {
    /// TLS's defaults, 50 trial times per duration, the 5 highest peaks
    /// refined, one thread.
    pub(crate) fn tls() -> Self {
        Self {
            template: Template::tls_default(),
            table_samples: 4096,
            steps: 50,
            min_depth: 10e-6,
            stars: Stars::TLS,
            constants: Constants::IAU,
            duration_step: 1.1,
            oversampling: 3,
            refine_peaks: 5,
            threads: NonZeroUsize::MIN,
        }
    }
}

/// What a search found.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Outcome {
    /// The best fit at each period, in increasing period.
    pub(crate) periods: Vec<PeriodBest>,
    /// The spectra, or `None` when nothing fitted at any period.
    pub(crate) spectra: Option<Spectra>,
    /// The best of the refined peaks, if any fitted.
    pub(crate) best: Option<Refined>,
    pub(crate) period_uncertainty: f64,
}

/// Indices of the `count` highest local maxima of `power`, highest first.
pub(crate) fn peaks(power: &[f64], count: usize) -> Vec<usize> {
    let n = power.len();
    let mut found: Vec<usize> = (0..n)
        .filter(|&i| {
            (i == 0 || power[i] > power[i - 1]) && (i + 1 == n || power[i] >= power[i + 1])
        })
        .collect();
    found.sort_by(|&a, &b| power[b].total_cmp(&power[a]).then(a.cmp(&b)));
    found.truncate(count);
    found
}

/// The search of `lc` over `periods` (in days, increasing, positive). The
/// light curve must span more than zero time.
pub(crate) fn run(lc: &LightCurve, periods: &[f64], options: Options) -> Outcome {
    let points = Points::new(lc);
    let origin = points.time[0];
    let span = points.time[points.time.len() - 1] - origin;
    let fractions = durations::grid(
        options.constants,
        options.stars,
        periods[0],
        periods[periods.len() - 1],
        options.duration_step,
    );
    let table = Table::new(&options.template, options.table_samples);
    let search = Search {
        points: &points,
        table: &table,
        fractions: &fractions,
        settings: Settings {
            steps: options.steps,
            min_depth: options.min_depth,
            stars: options.stars,
            constants: options.constants,
        },
        flat_chi2: points.flat_chi2(),
        origin,
        span,
    };
    let found = search.periods(periods, options.threads);

    if found.iter().all(|p| p.depth == 0.0) {
        return Outcome {
            periods: found,
            spectra: None,
            best: None,
            period_uncertainty: f64::INFINITY,
        };
    }
    let chi2: Vec<f64> = found.iter().map(|p| p.chi2).collect();
    let spectra = spectra(&chi2, options.oversampling);
    let uncertainty = period_uncertainty(periods, &spectra.power);

    // Each peak refined from its own grid period and a few either side, each
    // from its own best duration and mid-transit time: at low signal to
    // noise the chi-squared has basins a little apart in period and
    // duration together, which no search along one parameter at a time
    // crosses.
    let mut starts: Vec<usize> = peaks(&spectra.power, options.refine_peaks)
        .into_iter()
        .flat_map(|i| i.saturating_sub(NEIGHBOURS)..=(i + NEIGHBOURS).min(periods.len() - 1))
        .filter(|&i| found[i].depth > 0.0)
        .collect();
    starts.sort_unstable();
    starts.dedup();
    let best = starts
        .into_iter()
        .map(|i| {
            let p = found[i];
            // Reach the neighbouring periods, the neighbouring durations
            // and the neighbouring trial times.
            let below = if i > 0 { periods[i - 1] } else { p.period };
            let above = periods.get(i + 1).copied().unwrap_or(p.period);
            #[expect(clippy::cast_precision_loss, reason = "a small count")]
            let reach = Reach {
                t0: p.duration / options.steps as f64,
                duration: options.duration_step,
                period: (above - below).abs().max(f64::EPSILON * p.period),
            };
            let start = Position {
                period: p.period,
                duration: p.duration,
                t0: p.t0,
            };
            refine(&points, &options.template, start, reach)
        })
        .min_by(|a, b| a.fit.chi2.total_cmp(&b.fit.chi2));

    Outcome {
        periods: found,
        spectra: Some(spectra),
        best,
        period_uncertainty: uncertainty,
    }
}
