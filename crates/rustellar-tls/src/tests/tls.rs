// The public search: its checks, and its results those of the search
// inside, in public types.

use std::num::NonZeroUsize;

use rustellar_core::LightCurve;
use rustellar_model::{LimbDarkening, RadiusRatio};

use crate::fit::Points;
use crate::search::{self, Options, run};
use crate::{Device, Method, Periods, SearchOptions, Stars, Template, Tls, TlsError};

/// 12 days at 30 minutes, a 0.5% box-shaped dip 0.1 days long every 3
/// days from day 0.7, and an alternating ripple of 1e-4.
fn light_curve() -> LightCurve {
    let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 48.0).collect();
    let flux: Vec<f64> = time
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let dip = if (t - 0.7).rem_euclid(3.0) < 0.1 {
                5e-3
            } else {
                0.0
            };
            let ripple = if i % 2 == 0 { 1e-4 } else { -1e-4 };
            1.0 - dip + ripple
        })
        .collect();
    LightCurve::from_days(time, flux, vec![1e-3; 576]).unwrap()
}

fn grid(from: f64, count: u32) -> Vec<f64> {
    (0..count).map(|i| from + 0.01 * f64::from(i)).collect()
}

fn cpu() -> SearchOptions {
    SearchOptions::default()
        .with_device(Device::Cpu)
        .with_threads(NonZeroUsize::new(2).unwrap())
}

#[test]
fn a_light_curve_is_checked() {
    let one = LightCurve::from_days(vec![1.0], vec![1.0], vec![1e-3]).unwrap();
    assert_eq!(Tls::new(&one).unwrap_err(), TlsError::NoBaseline);
    // A weight too large for full precision.
    let tiny = LightCurve::from_days(vec![0.0, 1.0], vec![1.0; 2], vec![1e-200, 1e-3]).unwrap();
    assert_eq!(Tls::new(&tiny).unwrap_err(), TlsError::OutOfRange);
    // Weights fine, but the flat model's chi-squared overflows.
    let far = LightCurve::from_days(vec![0.0, 1.0], vec![1e10, 1.0], vec![1e-150, 1e-3]).unwrap();
    assert_eq!(Tls::new(&far).unwrap_err(), TlsError::OutOfRange);
    assert!(format!("{:?}", Tls::new(&light_curve()).unwrap()).starts_with("Tls"));
}

#[test]
fn the_settings_are_checked_before_the_search() {
    let tls = Tls::new(&light_curve()).unwrap();
    let periods = Periods::from_days(&[2.0, 3.0]).unwrap();
    let k = RadiusRatio::new(0.1).unwrap();
    let ld = LimbDarkening::uniform();
    let cases = [
        (
            cpu().with_min_depth(f64::NAN),
            TlsError::NonFinite { name: "min_depth" },
        ),
        (
            cpu().with_template(Template::Planet { k, b: 2.0, ld }),
            TlsError::TemplateNotTransiting,
        ),
        (
            cpu().with_stars(Stars {
                mass_max: 0.01,
                ..Stars::default()
            }),
            TlsError::StarRangeEmpty,
        ),
        (
            cpu().with_device(Device::Gpu(4096)),
            TlsError::GpuUnavailable,
        ),
    ];
    for (options, error) in cases {
        assert_eq!(tls.search(&periods, &options), Err(error));
    }
}

// Without the `cuda` feature no GPU can be asked for.
#[cfg(not(feature = "cuda"))]
#[test]
fn without_cuda_no_gpu() {
    let tls = Tls::new(&light_curve()).unwrap();
    let periods = Periods::from_days(&[2.0, 3.0]).unwrap();
    let options = cpu().with_device(Device::Gpu(0));
    assert_eq!(
        tls.search(&periods, &options),
        Err(TlsError::GpuUnavailable)
    );
    let auto = tls.search(&periods, &cpu().with_device(Device::Auto));
    assert_eq!(auto.unwrap().device(), Device::Cpu);
}

// The public result is the search's, in public types, in increasing period
// whatever the order given, for both methods.
#[test]
fn the_result_is_the_searchs() {
    let lc = light_curve();
    let tls = Tls::new(&lc).unwrap();
    let mut days = grid(2.5, 120);
    for method in [Method::Cells, Method::Fold] {
        let options = cpu().with_method(method);
        let mut reversed = days.clone();
        reversed.reverse();
        let result = tls
            .search(&Periods::from_days(&reversed).unwrap(), &options)
            .unwrap();
        let inside = run(
            &Points::new(&lc),
            &days,
            Options {
                threads: options.threads,
                bootstrap: options.bootstrap,
                method: match method {
                    Method::Cells => search::Method::Cells,
                    Method::Fold => search::Method::Fold,
                },
                ..Options::tls()
            },
        );
        assert_eq!(result.device(), Device::Cpu);
        assert_eq!(result.periods().len(), days.len());
        for (fit, period) in result.periods().iter().zip(&inside.periods) {
            assert_eq!(fit.period.as_days(), period.period);
            assert_eq!(fit.duration.as_days(), period.duration);
            assert_eq!(fit.transit_time.as_days(), period.t0);
            assert_eq!((fit.depth, fit.chi2), (period.depth, period.chi2));
        }
        let spectra = result.spectra().unwrap();
        let expected = inside.spectra.unwrap();
        assert_eq!(spectra.sr, expected.sr);
        assert_eq!(spectra.power, expected.power);
        assert_eq!(spectra.power_raw, expected.power_raw);
        assert_eq!(
            (spectra.sde, spectra.sde_raw),
            (expected.sde, expected.sde_raw)
        );

        let best = result.best().unwrap();
        let b = inside.best.unwrap();
        assert_eq!(best.period.as_days(), b.position.period);
        assert_eq!(best.transit_time.as_days(), b.position.t0);
        assert_eq!(best.duration.as_days(), b.position.duration);
        assert_eq!((best.depth, best.chi2), (b.fit.depth, b.fit.chi2));
        assert_eq!(best.points_in_transit, b.fit.in_transit);
        assert!((best.period.as_days() - 3.0).abs() < 0.01, "{best:?}");
        assert_eq!(
            best.period_uncertainty.unwrap().as_days(),
            inside.period_uncertainty
        );
        let (e, ie) = (best.errors.unwrap(), inside.errors.unwrap());
        assert_eq!(e.period.as_days(), ie.period);
        assert_eq!(e.transit_time.as_days(), ie.t0);
        assert_eq!(e.duration.as_days(), ie.duration);
        assert_eq!(e.depth, ie.depth);
        let (e, ie) = (best.bootstrap_errors.unwrap(), inside.bootstrap.unwrap());
        assert_eq!(e.period.as_days(), ie.period);
        assert_eq!(e.transit_time.as_days(), ie.t0);
        assert_eq!(e.duration.as_days(), ie.duration);
        assert_eq!(e.depth, ie.depth);
        assert_eq!(best.transits.len(), inside.transits.len());
        for (t, it) in best.transits.iter().zip(&inside.transits) {
            assert_eq!((t.number, t.time.as_days()), (it.number, it.time));
            assert_eq!(
                (t.depth, t.error, t.points),
                (it.depth, it.error, it.points)
            );
        }
        let (o, io) = (best.odd_even.unwrap(), inside.odd_even.unwrap());
        assert_eq!(
            (o.odd, o.odd_error, o.even, o.even_error, o.sigma),
            (io.odd, io.odd_error, io.even, io.even_error, io.sigma)
        );
        let (s, is) = (best.snr.unwrap(), inside.snr.unwrap());
        assert_eq!(
            (s.white, s.pink, s.white_noise, s.red_noise, s.depth),
            (is.white, is.pink, is.white_noise, is.red_noise, is.depth)
        );
        let (h, ih) = (best.shape.unwrap(), inside.shape.unwrap());
        assert_eq!(h.period.as_days(), ih.period);
        assert_eq!(h.transit_time.as_days(), ih.t0);
        assert_eq!(h.duration.as_days(), ih.duration);
        assert_eq!(
            (h.k, h.b, h.depth, h.a, h.chi2),
            (ih.k, ih.b, ih.depth, ih.a, ih.chi2)
        );
        let (he, ihe) = (h.errors.unwrap(), ih.errors.unwrap());
        assert_eq!(
            [
                he.period.as_days(),
                he.transit_time.as_days(),
                he.duration.as_days(),
                he.k,
                he.b
            ],
            ihe
        );
        days.truncate(60);
    }
}

// Fewer than two resamples: no bootstrap errors, the curvature's still.
#[test]
fn without_resamples_no_bootstrap() {
    let lc = light_curve();
    let periods = Periods::from_days(&grid(2.9, 20)).unwrap();
    for count in [0, 1] {
        let result = Tls::new(&lc)
            .unwrap()
            .search(&periods, &cpu().with_bootstrap(count))
            .unwrap();
        let best = result.best().unwrap();
        assert!(best.errors.is_some());
        assert!(best.bootstrap_errors.is_none());
    }
}

// Nothing dims: no spectra and no candidate. A peak at the edge of the
// periods searched: no period uncertainty.
#[test]
fn no_candidate_and_a_peak_at_the_edge() {
    let time: Vec<f64> = (0..200).map(|i| f64::from(i) / 20.0).collect();
    let above = LightCurve::from_days(time, vec![1.001; 200], vec![1e-3; 200]).unwrap();
    let result = Tls::new(&above)
        .unwrap()
        .search(&Periods::from_days(&[1.0, 2.0]).unwrap(), &cpu())
        .unwrap();
    assert!(result.spectra().is_none() && result.best().is_none());
    assert!(result.periods().iter().all(|p| p.depth == 0.0));

    // The transit's period the shortest searched: the peak never falls to
    // half below it.
    let edge = Tls::new(&light_curve())
        .unwrap()
        .search(&Periods::from_days(&grid(3.0, 50)).unwrap(), &cpu())
        .unwrap();
    let best = edge.best().unwrap();
    assert!((best.period.as_days() - 3.0).abs() < 0.01, "{best:?}");
    assert_eq!(best.period_uncertainty, None);
}

// With the `cuda` feature: the first GPU by default where there is one,
// any GPU asked for where it sets up, the same results as the CPU's.
#[cfg(feature = "cuda")]
#[test]
fn with_cuda_the_gpu() {
    let tls = Tls::new(&light_curve()).unwrap();
    let periods = Periods::from_days(&grid(2.5, 60)).unwrap();
    let on_cpu = tls.search(&periods, &cpu()).unwrap();
    let auto = tls
        .search(&periods, &cpu().with_device(Device::Auto))
        .unwrap();
    let asked = tls.search(&periods, &cpu().with_device(Device::Gpu(0)));
    if crate::cuda::Gpu::new(0).is_ok() {
        assert_eq!(auto.device(), Device::Gpu(0));
        let asked = asked.unwrap();
        assert_eq!(asked.device(), Device::Gpu(0));
        assert_eq!(asked.periods(), on_cpu.periods());
        assert_eq!(asked.best(), on_cpu.best());
    } else {
        assert_eq!(auto.device(), Device::Cpu);
        assert_eq!(asked, Err(TlsError::GpuUnavailable));
    }
    assert_eq!(auto.best(), on_cpu.best());
}
