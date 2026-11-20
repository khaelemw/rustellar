//! Prints the comparisons as Markdown tables.
//!
//! ```text
//! cargo run --release -p rustellar-validation --bin compare -- [SET...] [--limit N] [--step N] [--threads N] [--cpu]
//! ```
//!
//! SET is many, mixed, varying, short or sectors (rustellar-tls against
//! transitleastsquares), or limb (rustellar-bls against rustellar-tls on
//! limb-darkened transits), or window (how long a detrending window
//! should be, on transits over spotted stars), or vet (rustellar-vet's
//! alias table on TLS's candidates); default all of those.
//! `speed` times every crate against its Python references, and
//! `speed-model`, `speed-detrend`, `speed-bls` or `speed-tls` one crate;
//! never by default, as it wants a quiet machine. `--limit` searches only
//! each TLS set's first N curves (for vet too), and the window study's
//! first N of 400;
//! `--step` only every N-th curve of limb; `--threads` sets the threads
//! (default all cores); `--cpu` keeps the searches off the GPU with the
//! `cuda` feature.

use std::num::NonZeroUsize;
use std::process::ExitCode;
use std::thread;
use std::time::Instant;

use rustellar_tls::{Device, SearchOptions};
use rustellar_validation::{Kind, limb, speed, tls, vet, window};

const HELP: &str = "compare [SET...] [--limit N] [--step N] [--threads N] [--cpu]
  SET: many, mixed, varying, short, sectors, limb, window or vet (default all),
       speed or speed-CRATE: model, detrend, bls or tls (never by default)";

fn main() -> ExitCode {
    let (mut kinds, mut with_limb, mut with_window) = (vec![], false, false);
    let mut with_vet = false;
    let mut speeds: Vec<&str> = vec![];
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
            "vet" => with_vet = true,
            "speed" => speeds.extend(SPEEDS),
            name if name.starts_with("speed-") => {
                match SPEEDS
                    .into_iter()
                    .find(|s| name.strip_prefix("speed-") == Some(s))
                {
                    Some(one) => speeds.push(one),
                    None => return usage(),
                }
            }
            name => match Kind::from_name(name) {
                Some(kind) => kinds.push(kind),
                None => return usage(),
            },
        }
    }
    if kinds.is_empty() && !with_limb && !with_window && !with_vet && speeds.is_empty() {
        kinds = Kind::ALL.to_vec();
        with_limb = true;
        with_window = true;
        with_vet = true;
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
        let count = u64::try_from(limit.min(400)).expect("small");
        let cases = window::cases(count);
        println!("## Detrending window against transit duration\n");
        if !cases.is_empty() {
            let results = window::measure_all(&cases, threads);
            print!("{}", window::report(&cases, &results));
            println!("\n### A blind BLS search after each unmasked window\n");
            let found = window::search_all(&cases, threads);
            print!("{}", window::search_report(&cases, &found));
        }
        println!(
            "\n({} curves, {:.0} s)",
            cases.len(),
            start.elapsed().as_secs_f64()
        );
    }
    if with_vet {
        let start = Instant::now();
        let cases: Vec<vet::Case> = Kind::ALL
            .into_iter()
            .flat_map(|kind| vet::cases(kind, limit, threads))
            .collect();
        println!(
            "## rustellar-vet's alias table on transitleastsquares 1.32's candidates
"
        );
        print!("{}", vet::report(&cases));
        println!(
            "
({} candidates, {:.0} s)
",
            cases.len(),
            start.elapsed().as_secs_f64()
        );
    }
    for name in speeds {
        print!("{}", speed_report(name, threads));
    }
    ExitCode::SUCCESS
}

/// The crates with speed tables.
const SPEEDS: [&str; 4] = ["model", "detrend", "bls", "tls"];

fn speed_report(name: &str, threads: NonZeroUsize) -> String {
    use speed::{bls, detrend, model, references, tls};
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
        _ => tls::report(
            &tls::run(&tls::cases(), threads),
            &references(tls::reference()),
            threads,
        ),
    };
    format!(
        "## rustellar-{name} speed

{table}
({:.0} s)

",
        start.elapsed().as_secs_f64()
    )
}

fn usage() -> ExitCode {
    eprintln!("{HELP}");
    ExitCode::FAILURE
}
