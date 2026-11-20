//! Whitening with an autoregressive model before a search (as ARPS), on
//! the window study's curves: their white noise, and noise as strongly
//! correlated from one cadence to the next as an AR(1) process with 0.5 and
//! 0.8 (the same standard deviation), each with its transit and without.
//! Each curve flattened (the biweight at 2.5 transit durations), searched
//! (BLS), then whitened (`Autoregressive`, AIC order up to 3, 10 and 30)
//! and searched again: how often the best peak is the transit's period with
//! SDE above 7, how often a curve without a transit has a peak that high,
//! and, fairer, how often the transit is found above the SDE that 5% of the
//! curves without one reach.

use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::thread;

use rustellar_core::TimeDelta;
use rustellar_detrend::{Autoregressive, Detrend, Order};

use crate::stats::median;
use crate::window::{Case, Found, best, inject_with, quantile};

/// The noises: white, then AR(1) with these coefficients.
pub const NOISES: [f64; 3] = [0.0, 0.5, 0.8];

/// The highest orders AIC chooses from.
pub const ORDERS: [usize; 3] = [3, 10, 30];

/// For one curve: the search after flattening, then after whitening with
/// each of [`ORDERS`], with the model's order and gain.
#[derive(Debug, Clone)]
pub struct Searched {
    pub flattened: Found,
    pub whitened: Vec<(Found, usize, f64)>,
}

/// For one case and noise: the curve with its transit and without.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub transit: Searched,
    pub empty: Searched,
}

fn search(case: &Case, red: f64, transit: bool) -> Searched {
    let (lc, _) = inject_with(case, red, transit);
    let t14 = case.transit.t14().expect("a transit").as_days();
    let truth = case.transit.period().as_days();
    let flat = Detrend::default()
        .with_window(TimeDelta::from_days(2.5 * t14).expect("finite"))
        .with_threads(NonZeroUsize::MIN)
        .flatten(&lc, None)
        .expect("detrended");
    let whitened = ORDERS
        .iter()
        .map(|&most| {
            let white = Autoregressive::default()
                .with_order(Order::Aic(most))
                .whiten(flat.light_curve())
                .expect("whitened");
            (
                best(white.light_curve(), truth).0,
                white.order(),
                white.gain(),
            )
        })
        .collect();
    Searched {
        flattened: best(flat.light_curve(), truth).0,
        whitened,
    }
}

/// Each noise's outcome on `case`, in the order of [`NOISES`].
#[must_use]
pub fn measure(case: &Case) -> Vec<Outcome> {
    NOISES
        .iter()
        .map(|&red| Outcome {
            transit: search(case, red, true),
            empty: search(case, red, false),
        })
        .collect()
}

/// [`measure`] for every case, on `threads` threads.
#[must_use]
pub fn measure_all(cases: &[Case], threads: NonZeroUsize) -> Vec<Vec<Outcome>> {
    let chunk = cases.len().div_ceil(threads.get()).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(measure).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a study thread panicked"))
            .collect()
    })
}

#[expect(clippy::cast_precision_loss, reason = "counts")]
fn share(items: &[Found], pass: impl Fn(&Found) -> bool) -> f64 {
    items.iter().filter(|f| pass(f)).count() as f64 / items.len() as f64
}

/// The table: per noise and per curve (flattened, or whitened too), how
/// often the search finds the transit (right period, SDE above 7), how
/// often a curve without one has a peak above 7, how often the transit is
/// found above the SDE 5% of the curves without one reach, how often the
/// best peak is right at all, the median SDE, and the whitening model's
/// median order and gain.
#[must_use]
pub fn report(results: &[Vec<Outcome>]) -> String {
    let mut out = String::from(
        "| Noise | Searched | Found, SDE > 7 | False alarms, SDE > 7 \
         | Found at 5% false alarms | Right | Median SDE | Order | Gain |\n\
         |---|---|--:|--:|--:|--:|--:|--:|--:|\n",
    );
    for (j, &red) in NOISES.iter().enumerate() {
        let noise = if red == 0.0 {
            String::from("white")
        } else {
            format!("AR(1) {red}")
        };
        for which in std::iter::once(None).chain((0..ORDERS.len()).map(Some)) {
            let pick = |s: &Searched| which.map_or(s.flattened, |k| s.whitened[k].0);
            let transit: Vec<Found> = results.iter().map(|r| pick(&r[j].transit)).collect();
            let empty: Vec<Found> = results.iter().map(|r| pick(&r[j].empty)).collect();
            let quiet: Vec<f64> = empty.iter().map(|f| f.sde).collect();
            let cut = quantile(&quiet, 0.95);
            let sde: Vec<f64> = transit.iter().map(|f| f.sde).collect();
            let (name, order, gain) = match which {
                None => (
                    String::from("flattened"),
                    String::from("-"),
                    String::from("-"),
                ),
                Some(k) => {
                    #[expect(clippy::cast_precision_loss, reason = "small orders")]
                    let orders: Vec<f64> = results
                        .iter()
                        .map(|r| r[j].transit.whitened[k].1 as f64)
                        .collect();
                    let gains: Vec<f64> =
                        results.iter().map(|r| r[j].transit.whitened[k].2).collect();
                    (
                        format!("whitened, AIC up to {}", ORDERS[k]),
                        format!("{:.0}", median(&orders)),
                        format!("{:.2}", median(&gains)),
                    )
                }
            };
            let line = format!(
                "| {noise} | {name} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1} | {order} | {gain} |",
                share(&transit, |f| f.right && f.sde > 7.0),
                share(&empty, |f| f.sde > 7.0),
                share(&transit, |f| f.right && f.sde > cut),
                share(&transit, |f| f.right),
                median(&sde),
            );
            let _ = writeln!(out, "{line}");
        }
    }
    out
}
