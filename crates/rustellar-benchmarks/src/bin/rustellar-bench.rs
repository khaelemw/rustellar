//! The benchmark suites by crate: accuracy and speed against the Python
//! references, printed as Markdown.
//!
//! ```text
//! cargo validate list
//! cargo validate accuracy tls --quick
//! cargo validate speed fit mcmc
//! cargo validate all --quick
//! ```
//!
//! (`cargo validate` is the workspace alias for `cargo run --release -p
//! rustellar-benchmarks --bin rustellar-bench --`; `cargo validate-gpu`
//! adds the `cuda` feature.)

use std::fmt::Write;
use std::num::NonZeroUsize;
use std::process::ExitCode;

use rustellar_benchmarks::suites::{self, Measure, SUITES, Settings, Suite};

const HELP: &str = "rustellar-bench COMMAND [TARGET...] [OPTIONS]

Commands:
  list                  every suite: its crate, what it compares, how long it takes
  accuracy [TARGET...]  the accuracy suites of each crate or suite named (default all)
  speed [TARGET...]     the speed suites of each crate named (default all)
  all                   every accuracy suite, then every speed suite
  help                  this text

TARGET is a crate (model, detrend, bls, tls, ls, fit, vet, mcmc) or a
suite's name from `list`; `all` means every one.

Options:
  --quick       a smoke check: 20 curves a set, 8 of the limb-darkened study
  --limit N     each set's first N curves
  --step N      every N-th block of the limb-darkened study
  --threads N   threads (default all cores)
  --cpu         keep the searches off the GPU (with the cuda feature)
  --dry-run     list the suites a command would run, and stop

Speed wants a quiet machine. The references' times were recorded on an
AMD Ryzen 7 5700G; on another processor compare the ratios only.";

/// The crates in the workspace order.
const CRATES: [&str; 8] = ["model", "detrend", "bls", "tls", "ls", "fit", "vet", "mcmc"];

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        return usage();
    };
    let mut targets: Vec<String> = vec![];
    let mut settings = Settings::default();
    let (mut quick, mut dry_run) = (false, false);
    while let Some(arg) = args.next() {
        let mut number = || args.next().and_then(|n| n.parse::<usize>().ok());
        match arg.as_str() {
            "--quick" => quick = true,
            "--dry-run" => dry_run = true,
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
            "--help" | "-h" => return help(),
            flag if flag.starts_with("--") => return usage(),
            target => targets.push(target.to_owned()),
        }
    }
    if quick {
        settings = settings.quick();
    }
    let chosen = match command.as_str() {
        "list" => {
            print!("{}", list());
            return ExitCode::SUCCESS;
        }
        "help" | "--help" | "-h" => return help(),
        "accuracy" => pick(&targets, Measure::Accuracy),
        "speed" => pick(&targets, Measure::Speed),
        "all" if targets.is_empty() => Some(
            pick(&[], Measure::Accuracy)
                .into_iter()
                .flatten()
                .chain(pick(&[], Measure::Speed).into_iter().flatten())
                .collect(),
        ),
        _ => None,
    };
    let Some(chosen) = chosen else {
        return usage();
    };
    let minutes: f64 = chosen
        .iter()
        .map(|s| if quick { s.quick_minutes } else { s.minutes })
        .sum();
    eprintln!(
        "{} suites, about {minutes:.0} minutes on 16 CPU threads",
        chosen.len()
    );
    if dry_run {
        for suite in chosen {
            println!("{} ({})", suite.name, suite.crate_name);
        }
        return ExitCode::SUCCESS;
    }
    for suite in chosen {
        eprintln!("running {} ({})", suite.name, suite.crate_name);
        print!("{}", suites::run(suite, &settings));
    }
    ExitCode::SUCCESS
}

/// The suites `targets` name, each crate's in the registry's order; every
/// one of `measure` for none or `all`. `None` for a name that is neither a
/// crate nor a suite.
fn pick(targets: &[String], measure: Measure) -> Option<Vec<Suite>> {
    let all = || {
        SUITES
            .into_iter()
            .filter(|s| s.measure == measure)
            .collect()
    };
    if targets.is_empty() || targets.iter().any(|t| t == "all") {
        return Some(all());
    }
    let mut out: Vec<Suite> = vec![];
    for t in targets {
        let found = if CRATES.contains(&t.as_str()) {
            suites::of(t, measure)
        } else {
            suites::find(t, measure).into_iter().collect()
        };
        if found.is_empty() && !CRATES.contains(&t.as_str()) {
            return None;
        }
        for s in found {
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
    Some(out)
}

fn list() -> String {
    let mut out = String::from(
        "| Suite | Crate | Measures | Minutes (quick) | Compares |\n|---|---|---|--:|---|\n",
    );
    for s in SUITES {
        let measure = match s.measure {
            Measure::Accuracy => "accuracy",
            Measure::Speed => "speed",
        };
        writeln!(
            out,
            "| {} | {} | {measure} | {} ({}) | {} |",
            s.name, s.crate_name, s.minutes, s.quick_minutes, s.about
        )
        .expect("a string");
    }
    out
}

fn help() -> ExitCode {
    println!("{HELP}");
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!("{HELP}");
    ExitCode::FAILURE
}
