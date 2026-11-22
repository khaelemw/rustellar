//! The benchmark suites by crate: accuracy and speed against the Python
//! references, printed as Markdown.
//!
//! ```text
//! cargo validate list
//! cargo validate accuracy tls --quick
//! cargo validate speed fit mcmc
//! cargo validate all --quick
//! cargo validate all --save --repeat 3       # a release's results
//! cargo validate diff 0.1.0 0.2.0
//! ```
//!
//! (`cargo validate` is the workspace alias for `cargo run --release -p
//! rustellar-benchmarks --bin rustellar-bench --`; `cargo validate-gpu`
//! adds the `cuda` feature.)

use std::fmt::Write;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rustellar_benchmarks::results;
use rustellar_benchmarks::suites::{self, Measure, SUITES, Settings, Suite};

const HELP: &str = "rustellar-bench COMMAND [TARGET...] [OPTIONS]

Commands:
  list                  every suite: its crate, what it compares, how long it takes
  accuracy [TARGET...]  the accuracy suites of each crate or suite named (default all)
  speed [TARGET...]     the speed suites of each crate named (default all)
  all                   every accuracy suite, then every speed suite
  report [RUN]          a saved run's tables (default this version's)
  diff OLD NEW          every table cell that differs between two saved runs
  help                  this text

TARGET is a crate (model, detrend, bls, tls, ls, fit, vet, mcmc) or a
suite's name from `list`; `all` means every one. A RUN is a version
saved under results/ (`0.1.0`) or a folder.

Options:
  --quick       a smoke check: 20 curves a set, 8 of the limb-darkened study
  --limit N     each set's first N curves
  --step N      every N-th block of the limb-darkened study
  --threads N   threads (default all cores)
  --cpu         keep the searches off the GPU (with the cuda feature)
  --dry-run     list the suites a command would run, and stop
  --save        also save each suite's tables, and the machine's card, in
                results/<version>/ of this crate
  --to DIR      save in DIR instead (implies --save)
  --repeat N    run each speed suite N times: each time its best, with the
                spread kept beside it for diff (default 1)
  --tolerance F diff: the change in a time that counts (default 0.1, or
                twice either run's spread where larger)

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
    let mut save: Option<PathBuf> = None;
    let mut repeat = 1;
    let mut tolerance = 0.1;
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
            "--save" => save = save.or_else(|| Some(results::default_folder())),
            "--to" => match args.next() {
                Some(dir) => save = Some(PathBuf::from(dir)),
                None => return usage(),
            },
            "--repeat" => match args.next().and_then(|n| n.parse::<usize>().ok()) {
                Some(n) if n > 0 => repeat = n,
                _ => return usage(),
            },
            "--tolerance" => match args.next().and_then(|n| n.parse::<f64>().ok()) {
                Some(f) if f >= 0.0 => tolerance = f,
                _ => return usage(),
            },
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
        "report" if targets.len() <= 1 => {
            return report(&run_folder(targets.first().map(String::as_str)));
        }
        "diff" if targets.len() == 2 => {
            return diff(
                &run_folder(Some(&targets[0])),
                &run_folder(Some(&targets[1])),
                tolerance,
            );
        }
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
    if chosen.iter().any(|s| s.measure == Measure::Speed) {
        let spread = results::steadiness(9);
        if spread > 0.05 {
            eprintln!(
                "warning: the machine is not quiet (a fixed loop's times spread {:.0}%); speed tables will be noisy",
                spread * 100.0
            );
        }
    }
    if let Some(folder) = &save {
        let when = format!("{command} {}", targets.join(" "));
        let existing = std::fs::read_to_string(folder.join("machine.md")).ok();
        let card = results::card(
            existing.as_deref(),
            &results::machine(settings.threads.get(), cfg!(feature = "cuda"), &when),
            &when,
        );
        if let Err(e) = results::save(folder, "machine.md", &card) {
            eprintln!("{}: {e}", folder.display());
            return ExitCode::FAILURE;
        }
    }
    for suite in chosen {
        eprintln!("running {} ({})", suite.name, suite.crate_name);
        let (text, spreads) = if suite.measure == Measure::Speed && repeat > 1 {
            let runs: Vec<String> = (0..repeat).map(|_| suites::run(suite, &settings)).collect();
            results::merge(&runs)
        } else {
            (suites::run(suite, &settings), vec![])
        };
        print!("{text}");
        if let Some(folder) = &save {
            if let Err(e) = save_suite(folder, suite, &text, &spreads) {
                eprintln!("{}: {e}", folder.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(folder) = &save {
        eprintln!("saved in {}", folder.display());
    }
    ExitCode::SUCCESS
}

/// A suite's tables saved, with its spreads beside them when it was
/// repeated.
fn save_suite(
    folder: &Path,
    suite: Suite,
    text: &str,
    spreads: &[(String, f64)],
) -> std::io::Result<()> {
    let measure = match suite.measure {
        Measure::Accuracy => "accuracy",
        Measure::Speed => "speed",
    };
    let name = results::file_name(measure, suite.name);
    results::save(folder, &name, text)?;
    if !spreads.is_empty() {
        let csv = name.replace(".md", ".spread.csv");
        results::save(folder, &csv, &results::spreads_csv(spreads))?;
    }
    Ok(())
}

/// A saved run's folder: `run` as a folder if there is one, else
/// `results/<run>` of this crate; this version's for none.
fn run_folder(run: Option<&str>) -> PathBuf {
    match run {
        None => results::default_folder(),
        Some(r) if Path::new(r).is_dir() => PathBuf::from(r),
        Some(r) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("results")
            .join(r),
    }
}

fn report(folder: &Path) -> ExitCode {
    match results::load(folder) {
        Ok(files) if !files.is_empty() => {
            for (_, text) in files {
                println!("{text}");
            }
            ExitCode::SUCCESS
        }
        Ok(_) => {
            eprintln!("{}: no saved results", folder.display());
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("{}: {e}", folder.display());
            ExitCode::FAILURE
        }
    }
}

fn spreads(folder: &Path, name: &str) -> Vec<(String, f64)> {
    std::fs::read_to_string(folder.join(name.replace(".md", ".spread.csv")))
        .map(|csv| results::read_spreads(&csv))
        .unwrap_or_default()
}

fn diff(old: &Path, new: &Path, tolerance: f64) -> ExitCode {
    let (a, b) = match (results::load(old), results::load(new)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) => {
            eprintln!("{}: {e}", old.display());
            return ExitCode::FAILURE;
        }
        (_, Err(e)) => {
            eprintln!("{}: {e}", new.display());
            return ExitCode::FAILURE;
        }
    };
    let mut changed = 0;
    for (name, new_text) in &b {
        if name == "machine.md" {
            continue;
        }
        let Some((_, old_text)) = a.iter().find(|(n, _)| n == name) else {
            println!("## {name}\n\nOnly in {}.\n", new.display());
            continue;
        };
        let known: Vec<(String, f64)> = spreads(old, name)
            .into_iter()
            .chain(spreads(new, name))
            .collect();
        let allowed = |cell: &str| {
            known
                .iter()
                .filter(|(c, _)| c == cell)
                .map(|&(_, s)| 2.0 * s)
                .fold(tolerance, f64::max)
        };
        let changes = results::diff(old_text, new_text, &allowed);
        changed += changes.len();
        println!("## {name}\n\n{}", results::diff_report(&changes));
    }
    for (name, _) in &a {
        if name != "machine.md" && !b.iter().any(|(n, _)| n == name) {
            println!("## {name}\n\nOnly in {}.\n", old.display());
        }
    }
    eprintln!("{changed} cells differ");
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
