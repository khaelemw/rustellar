//! Prints the comparisons as Markdown tables.
//!
//! ```text
//! cargo run --release -p rustellar-validation --bin compare -- [SET...] [--limit N] [--step N] [--threads N] [--cpu]
//! ```
//!
//! SET is many, mixed, varying, short or sectors (rustellar-tls against
//! transitleastsquares), or limb (rustellar-bls against rustellar-tls on
//! limb-darkened transits), or window (how long a detrending window
//! should be, on transits over spotted stars); default all. `--limit` searches only each TLS
//! set's first N curves; `--step` only every N-th curve of limb; `--threads`
//! sets the threads (default all cores); `--cpu` keeps the searches off the
//! GPU with the `cuda` feature.

use std::num::NonZeroUsize;
use std::process::ExitCode;
use std::thread;
use std::time::Instant;

use rustellar_tls::{Device, SearchOptions};
use rustellar_validation::{Kind, limb, tls, window};

const HELP: &str = "compare [SET...] [--limit N] [--step N] [--threads N] [--cpu]
  SET: many, mixed, varying, short, sectors, limb or window (default all)";

fn main() -> ExitCode {
    let (mut kinds, mut with_limb, mut with_window) = (vec![], false, false);
    let (mut limit, mut step) = (usize::MAX, NonZeroUsize::MIN);
    let mut threads = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let mut options = SearchOptions::default().with_bootstrap(0);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut number = || args.next().and_then(|n| n.parse::<usize>().ok());
        match arg.as_str() {
            "--limit" => match number() {
                Some(n) => limit = n,
                None => return usage(),
            },
            "--step" => match number().and_then(NonZeroUsize::new) {
                Some(n) => step = n,
                None => return usage(),
            },
            "--threads" => match number().and_then(NonZeroUsize::new) {
                Some(n) => threads = n,
                None => return usage(),
            },
            "--cpu" => options = options.with_device(Device::Cpu),
            "limb" => with_limb = true,
            "window" => with_window = true,
            name => match Kind::from_name(name) {
                Some(kind) => kinds.push(kind),
                None => return usage(),
            },
        }
    }
    if kinds.is_empty() && !with_limb && !with_window {
        kinds = Kind::ALL.to_vec();
        with_limb = true;
        with_window = true;
    }
    if !kinds.is_empty() {
        println!("## rustellar-tls against transitleastsquares 1.32\n");
    }
    let per_search = options.with_threads(threads);
    for kind in kinds {
        let start = Instant::now();
        let curves = tls::run(kind, limit, &per_search);
        print!("{}", tls::report(kind, &curves));
        println!("\n({:.0} s)\n", start.elapsed().as_secs_f64());
    }
    if with_limb {
        let start = Instant::now();
        let cases = limb::cases(step.get());
        let hits = limb::search_all(&cases, &options, threads);
        print!("{}", limb::report(&cases, &hits));
        println!(
            "\n({} curves, {:.0} s)",
            cases.len(),
            start.elapsed().as_secs_f64()
        );
    }
    if with_window {
        let start = Instant::now();
        let cases = window::cases(400);
        let results = window::measure_all(&cases, threads);
        println!("## Detrending window against transit duration\n");
        print!("{}", window::report(&cases, &results));
        println!("\n### A blind BLS search after each unmasked window\n");
        let found = window::search_all(&cases, threads);
        print!("{}", window::search_report(&cases, &found));
        println!(
            "\n({} curves, {:.0} s)",
            cases.len(),
            start.elapsed().as_secs_f64()
        );
    }
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!("{HELP}");
    ExitCode::FAILURE
}
