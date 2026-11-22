//! Every benchmark suite by name, what it compares and how long it takes,
//! run with one call: the registry behind `rustellar-bench` and `compare`.

use std::num::NonZeroUsize;
use std::thread;
use std::time::Instant;

use rustellar_tls::{Device, SearchOptions};

use crate::{Kind, limb, masking, methods, speed, tls, vet, whitening, window};

/// Accuracy against a reference, or speed against one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Measure {
    Accuracy,
    Speed,
}

/// One suite: its name, the crate it measures, what it is and about how
/// long it takes in full and with `--quick` (minutes, on 16 threads of an
/// AMD Ryzen 7 5700G without a GPU; the `cuda` feature makes the searches
/// several times faster).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Suite {
    pub name: &'static str,
    pub crate_name: &'static str,
    pub measure: Measure,
    pub about: &'static str,
    pub minutes: f64,
    pub quick_minutes: f64,
}

const fn accuracy(
    name: &'static str,
    crate_name: &'static str,
    about: &'static str,
    minutes: f64,
    quick_minutes: f64,
) -> Suite {
    Suite {
        name,
        crate_name,
        measure: Measure::Accuracy,
        about,
        minutes,
        quick_minutes,
    }
}

const fn speed(crate_name: &'static str, about: &'static str, minutes: f64) -> Suite {
    Suite {
        name: crate_name,
        crate_name,
        measure: Measure::Speed,
        about,
        minutes,
        quick_minutes: minutes,
    }
}

/// Every suite, accuracy first.
pub const SUITES: [Suite; 19] = [
    accuracy(
        "many",
        "tls",
        "300 transits, 27 d at 30 min, against transitleastsquares",
        4.0,
        0.3,
    ),
    accuracy(
        "mixed",
        "tls",
        "200 transits, 30 then 10 min, against transitleastsquares",
        3.0,
        0.3,
    ),
    accuracy(
        "varying",
        "tls",
        "200 transits, errors 1 to 3 times the noise, against transitleastsquares",
        2.5,
        0.3,
    ),
    accuracy(
        "short",
        "tls",
        "40 transits, 27 d at 2 min, against transitleastsquares",
        1.0,
        0.5,
    ),
    accuracy(
        "sectors",
        "tls",
        "24 transits, two sectors at 2 min, against transitleastsquares",
        1.6,
        1.3,
    ),
    accuracy(
        "limb",
        "bls",
        "1,000 limb-darkened transits: BLS against TLS at 1% false alarms",
        150.0,
        1.3,
    ),
    accuracy(
        "window",
        "detrend",
        "400 transits on spotted stars: depth kept per window length",
        2.0,
        0.1,
    ),
    accuracy(
        "methods",
        "detrend",
        "the detrending methods on the window curves, with wotan",
        2.0,
        0.1,
    ),
    accuracy(
        "masking",
        "detrend",
        "detrending again with the candidate's transits masked",
        2.5,
        0.2,
    ),
    accuracy(
        "whitening",
        "detrend",
        "an autoregressive model before the search, white and red noise",
        11.0,
        0.6,
    ),
    accuracy(
        "aliases",
        "vet",
        "the alias table on transitleastsquares' candidates",
        0.1,
        0.1,
    ),
    speed(
        "model",
        "light curves against batman, PyTransit and exoplanet-core",
        0.2,
    ),
    speed(
        "detrend",
        "each detrending method against wotan, scipy and statsmodels",
        0.5,
    ),
    speed("bls", "searches against astropy's BoxLeastSquares", 3.0),
    speed("tls", "searches against transitleastsquares", 1.0),
    speed("ls", "periodograms against astropy's LombScargle", 0.3),
    speed("fit", "fits against batman and PyTransit with scipy", 0.2),
    speed(
        "vet",
        "each vetting step (no reference runs the same checks)",
        0.5,
    ),
    speed(
        "mcmc",
        "time per independent sample against emcee and zeus",
        3.0,
    ),
];

/// The suites of `crate_name` that measure `measure`.
#[must_use]
pub fn of(crate_name: &str, measure: Measure) -> Vec<Suite> {
    SUITES
        .into_iter()
        .filter(|s| s.crate_name == crate_name && s.measure == measure)
        .collect()
}

/// The suite named `name` that measures `measure`.
#[must_use]
pub fn find(name: &str, measure: Measure) -> Option<Suite> {
    SUITES
        .into_iter()
        .find(|s| s.name == name && s.measure == measure)
}

/// How a run is made: the curves each suite takes, the threads and the
/// searches' options.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Each TLS set's first N curves (for the alias table too), and the
    /// window, methods, masking and whitening studies' first N of 400.
    pub limit: usize,
    /// Every N-th block of the limb-darkened study.
    pub step: NonZeroUsize,
    pub threads: NonZeroUsize,
    pub options: SearchOptions,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            limit: usize::MAX,
            step: NonZeroUsize::MIN,
            threads: thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
            options: SearchOptions::default().with_bootstrap(0),
        }
    }
}

impl Settings {
    /// A smoke check: 20 curves a set, every 125th limb-darkened block (8
    /// curves).
    #[must_use]
    pub fn quick(mut self) -> Self {
        self.limit = self.limit.min(20);
        self.step = self.step.max(NonZeroUsize::new(125).expect("positive"));
        self
    }

    /// The searches kept off the GPU.
    #[must_use]
    pub fn cpu(mut self) -> Self {
        self.options = self.options.with_device(Device::Cpu);
        self
    }
}

fn count(settings: &Settings) -> u64 {
    u64::try_from(settings.limit.min(400)).expect("small")
}

fn elapsed(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

/// `suite` run, as Markdown: a heading, its tables and the time it took.
#[must_use]
pub fn run(suite: Suite, settings: &Settings) -> String {
    let start = Instant::now();
    let threads = settings.threads;
    let body = match (suite.measure, suite.name) {
        (Measure::Speed, name) => return speed_report(name, threads),
        (Measure::Accuracy, "limb") => {
            let cases = limb::cases(settings.step.get());
            let hits = limb::search_all(&cases, &settings.options, threads);
            format!(
                "{}\n({} curves, {:.0} s)\n",
                limb::report(&cases, &hits),
                cases.len(),
                elapsed(start)
            )
        }
        (Measure::Accuracy, "window") => {
            let cases = window::cases(count(settings));
            let mut out = String::from("## Detrending window against transit duration\n\n");
            if !cases.is_empty() {
                out += &window::report(&cases, &window::measure_all(&cases, threads));
                out += "\n### A blind BLS search after each unmasked window\n\n";
                out += &window::search_report(&cases, &window::search_all(&cases, threads));
            }
            format!("{out}\n({} curves, {:.0} s)\n", cases.len(), elapsed(start))
        }
        (Measure::Accuracy, "methods") => {
            let cases = window::cases(count(settings));
            let mut out = String::from("## Detrending methods on the window study's curves\n\n");
            if !cases.is_empty() {
                out += &methods::report(&cases, &methods::measure_all(&cases, threads));
            }
            format!("{out}\n({} curves, {:.0} s)\n", cases.len(), elapsed(start))
        }
        (Measure::Accuracy, "masking") => {
            let cases = window::cases(count(settings));
            let mut out = String::from("## Transit masking on the window study's curves\n\n");
            if !cases.is_empty() {
                out += &masking::report(&cases, &masking::measure_all(&cases, threads));
            }
            format!("{out}\n({} curves, {:.0} s)\n", cases.len(), elapsed(start))
        }
        (Measure::Accuracy, "whitening") => {
            let cases = window::cases(count(settings));
            let mut out =
                String::from("## Whitening before the search on the window study's curves\n\n");
            if !cases.is_empty() {
                out += &whitening::report(&whitening::measure_all(&cases, threads));
            }
            format!("{out}\n({} curves, {:.0} s)\n", cases.len(), elapsed(start))
        }
        (Measure::Accuracy, "aliases") => {
            let cases: Vec<vet::Case> = Kind::ALL
                .into_iter()
                .flat_map(|kind| vet::cases(kind, settings.limit, threads))
                .collect();
            format!(
                "## rustellar-vet's alias table on transitleastsquares 1.32's candidates\n\n{}\n({} candidates, {:.0} s)\n",
                vet::report(&cases),
                cases.len(),
                elapsed(start)
            )
        }
        (Measure::Accuracy, name) => {
            let kind = Kind::from_name(name).expect("a TLS set");
            let curves = tls::run(
                kind,
                settings.limit,
                &settings.options.with_threads(threads),
            );
            format!(
                "## rustellar-tls against transitleastsquares 1.32\n\n{}\n({:.0} s)\n",
                tls::report(kind, &curves),
                elapsed(start)
            )
        }
    };
    format!("{body}\n")
}

fn speed_report(name: &str, threads: NonZeroUsize) -> String {
    use speed::{bls, detrend, fit, ls, mcmc, model, references, tls, vet};
    let start = Instant::now();
    let table = match name {
        "model" => model::report(&model::run(7), &references(model::reference())),
        "detrend" => detrend::report(
            &detrend::run(7, threads, &detrend::light_curves(), 1),
            &references(detrend::reference()),
            threads,
        ),
        "bls" => bls::report(
            &bls::run(&bls::cases(), threads),
            &references(bls::reference()),
            threads,
        ),
        "fit" => fit::report(
            &fit::run(&fit::cases(), 7, 0.2),
            &references(fit::reference()),
        ),
        "ls" => ls::report(
            &ls::run(threads, &ls::EXACT, &ls::FAST),
            &references(ls::reference()),
            threads,
        ),
        "vet" => {
            let cases = vet::cases();
            vet::report(&cases, &vet::run(&cases, 7, 0.2))
        }
        "mcmc" => mcmc::report(
            &mcmc::run(&mcmc::light_curve(), threads, mcmc::RUN),
            &references(mcmc::reference()),
        ),
        _ => tls::report(
            &tls::run(&tls::cases(), threads),
            &references(tls::reference()),
            threads,
        ),
    };
    format!(
        "## rustellar-{name} speed\n\n{table}\n({:.0} s)\n\n",
        elapsed(start)
    )
}
