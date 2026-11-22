//! Prints the comparisons as Markdown tables.
//!
//! ```text
//! cargo run --release -p rustellar-benchmarks --bin compare -- [SET...] [--limit N] [--step N] [--threads N] [--cpu]
//! ```
//!
//! SET is many, mixed, varying, short or sectors (rustellar-tls against
//! transitleastsquares), or limb (rustellar-bls against rustellar-tls on
//! limb-darkened transits), or window (how long a detrending window
//! should be, on transits over spotted stars), or methods (the detrending
//! methods on the same curves, with wotan's), or masking (detrending
//! again around a search's candidate on the same curves), or whitening
//! (an autoregressive model before the search, with red noise too), or vet (rustellar-vet's
//! alias table on TLS's candidates); default all of those.
//! `--export-methods FILE` writes those curves for wotan to be run on, and
//! `--search-flattened IN OUT` searches the curves wotan flattened.
//! `speed` times every crate against its Python references, and
//! `speed-model`, `speed-detrend`, `speed-bls`, `speed-tls`, `speed-ls`,
//! `speed-fit`, `speed-vet` or `speed-mcmc`
//! one crate;
//! never by default, as it wants a quiet machine. `--limit` searches only
//! each TLS set's first N curves (for vet too), and the window, methods,
//! masking and whitening studies' first N of 400;
//! `--step` only every N-th curve of limb; `--threads` sets the threads
//! (default all cores); `--cpu` keeps the searches off the GPU with the
//! `cuda` feature.
//!
//! `rustellar-bench` runs the same suites by crate, with `list` and
//! `--quick`.

use std::num::NonZeroUsize;
use std::process::ExitCode;

use rustellar_benchmarks::suites::{self, Measure, Settings, Suite};
use rustellar_benchmarks::{methods, window};

const HELP: &str =
    "compare [SET...] [--limit N] [--step N] [--threads N] [--cpu] [--export-methods FILE]
        [--search-flattened IN OUT]
  SET: many, mixed, varying, short, sectors, limb, window, methods, masking, whitening or vet (default all),
       speed or speed-CRATE: model, detrend, bls, tls, ls, fit, vet or mcmc (never by default)";

fn main() -> ExitCode {
    let mut chosen: Vec<Suite> = vec![];
    let mut export: Option<String> = None;
    let mut search: Option<(String, String)> = None;
    let mut settings = Settings::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut number = || args.next().and_then(|n| n.parse::<usize>().ok());
        match arg.as_str() {
            "--limit" => match number() {
                Some(n) => settings.limit = n,
                None => return usage(),
            },
            "--step" => match number().and_then(NonZeroUsize::new) {
                Some(n) => settings.step = n,
                None => return usage(),
            },
            "--threads" => match number().and_then(NonZeroUsize::new) {
                Some(n) => settings.threads = n,
                None => return usage(),
            },
            "--cpu" => settings = settings.cpu(),
            "--export-methods" => match args.next() {
                Some(path) => export = Some(path),
                None => return usage(),
            },
            "--search-flattened" => match (args.next(), args.next()) {
                (Some(from), Some(to)) => search = Some((from, to)),
                _ => return usage(),
            },
            "speed" => chosen.extend(
                suites::SUITES
                    .into_iter()
                    .filter(|s| s.measure == Measure::Speed),
            ),
            "vet" => chosen.extend(suites::find("aliases", Measure::Accuracy)),
            name => {
                let found = match name.strip_prefix("speed-") {
                    Some(c) => suites::find(c, Measure::Speed),
                    None => suites::find(name, Measure::Accuracy),
                };
                match found {
                    Some(suite) => chosen.push(suite),
                    None => return usage(),
                }
            }
        }
    }
    if let Some(path) = export {
        let count = u64::try_from(settings.limit.min(400)).expect("small");
        return match std::fs::write(&path, methods::export(&window::cases(count))) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{path}: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if let Some((from, to)) = search {
        let lines = match std::fs::File::open(&from) {
            Ok(file) => {
                std::io::BufRead::lines(std::io::BufReader::new(file)).map_while(Result::ok)
            }
            Err(e) => {
                eprintln!("{from}: {e}");
                return ExitCode::FAILURE;
            }
        };
        return match std::fs::write(&to, methods::search_flattened(lines, settings.threads)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{to}: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if chosen.is_empty() {
        chosen = suites::SUITES
            .into_iter()
            .filter(|s| s.measure == Measure::Accuracy)
            .collect();
    }
    for suite in chosen {
        print!("{}", suites::run(suite, &settings));
    }
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!("{HELP}");
    ExitCode::FAILURE
}
