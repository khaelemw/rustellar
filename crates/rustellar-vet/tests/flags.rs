// Period and time flags against brute force: every ratio p / q tried, every
// transit and window compared.

use std::num::NonZeroU16;

use rustellar_vet::{
    Candidate, Epoch, LightCurve, SuspectPeriod, TimeDelta, TimeWindow, VetError, flag_periods,
    flag_times,
};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
}

fn candidate(period: f64) -> Candidate {
    Candidate::new(days(period), at(1.0), days(0.125)).unwrap()
}

fn nz(n: u16) -> NonZeroU16 {
    NonZeroU16::new(n).unwrap()
}

fn gcd(a: u16, b: u16) -> u16 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Every (suspect, p, q) in lowest terms within tolerance, by trying them
/// all.
fn brute(period: f64, suspects: &[SuspectPeriod]) -> Vec<(usize, u16, u16)> {
    let mut found = Vec::new();
    for (k, s) in suspects.iter().enumerate() {
        for p in 1..=s.multiples().get() {
            for q in 1..=s.fractions().get() {
                let target = s.period().as_days() * f64::from(p) / f64::from(q);
                if gcd(p, q) == 1 && (period / target - 1.0).abs() <= s.tolerance() {
                    found.push((k, p, q));
                }
            }
        }
    }
    found.sort_unstable();
    found
}

#[test]
fn periods_as_brute_force() {
    let suspects = [
        SuspectPeriod::new(days(0.5), 2e-3)
            .unwrap()
            .with_multiples(nz(200)),
        SuspectPeriod::new(days(13.7), 1e-2)
            .unwrap()
            .with_multiples(nz(3))
            .with_fractions(nz(5)),
        SuspectPeriod::new(days(1.0), 0.3)
            .unwrap()
            .with_multiples(nz(4))
            .with_fractions(nz(4)),
        SuspectPeriod::new(days(29.53), 0.0).unwrap(),
    ];
    let mut checked = 0;
    for i in 0..4000 {
        let period = 0.3 + 0.0123 * f64::from(i);
        let flags = flag_periods(&candidate(period), &suspects);
        let mut got: Vec<(usize, u16, u16)> = flags
            .iter()
            .map(|f| (f.suspect, f.ratio.numerator(), f.ratio.denominator()))
            .collect();
        got.sort_unstable();
        assert_eq!(got, brute(period, &suspects), "period {period}");
        for f in &flags {
            let s = &suspects[f.suspect];
            let target = s.period().as_days() * f.ratio.value();
            assert!((f.period.as_days() - target).abs() <= 1e-15 * target);
            assert_eq!(f.offset, period / f.period.as_days() - 1.0);
        }
        // The closest first.
        assert!(
            flags
                .windows(2)
                .all(|w| w[0].offset.abs() <= w[1].offset.abs())
        );
        checked += flags.len();
    }
    assert!(checked > 1000, "{checked}");
    // Exactly on a suspect with no tolerance, and nothing without one.
    let exact = SuspectPeriod::new(days(2.0), 0.0).unwrap();
    assert_eq!(flag_periods(&candidate(2.0), &[exact]).len(), 1);
    assert!(flag_periods(&candidate(2.0), &[]).is_empty());
}

#[test]
fn suspect_settings_and_errors() {
    let s = SuspectPeriod::new(days(13.7), 5e-3).unwrap();
    assert_eq!(s.period(), days(13.7));
    assert_eq!(s.tolerance(), 5e-3);
    assert_eq!((s.multiples().get(), s.fractions().get()), (1, 1));
    for bad in [0.0, -1.0] {
        assert_eq!(
            SuspectPeriod::new(days(bad), 1e-3),
            Err(VetError::NotPositive {
                name: "suspect period"
            })
        );
    }
    for bad in [-1e-9, 1.0, 2.0, f64::NAN] {
        assert_eq!(
            SuspectPeriod::new(days(1.0), bad),
            Err(VetError::ToleranceOutOfRange)
        );
    }
    assert_eq!(
        VetError::ToleranceOutOfRange.to_string(),
        "a suspect period's tolerance must be at least 0 and below 1"
    );
}

/// 10 days every 30 minutes.
fn curve() -> LightCurve {
    let time: Vec<f64> = (0..480).map(|i| f64::from(i) / 48.0).collect();
    LightCurve::from_btjd(time, vec![1.0; 480], vec![1e-3; 480]).unwrap()
}

#[test]
fn times_as_brute_force() {
    let lc = curve();
    // Transits at 1, 4, 7 (and -2, 10, 13 outside), 0.125 days long.
    let c = candidate(3.0);
    let windows = [
        // Overlaps the transit at 4 by its start.
        TimeWindow::new(at(3.9), at(3.95)).unwrap(),
        // Ends exactly where the transit at 7 starts: no overlap.
        TimeWindow::new(at(6.5), at(7.0 - 0.0625)).unwrap(),
        // An instant inside the transit at 7.
        TimeWindow::new(at(7.01), at(7.01)).unwrap(),
        // Covers the transits at 1 and 4.
        TimeWindow::new(at(0.0), at(5.0)).unwrap(),
        // Covers the predicted transit at 10, beyond the last point (9.98),
        // which still overlaps the span.
        TimeWindow::around(at(10.0), days(0.01)).unwrap(),
        // Far beyond the data.
        TimeWindow::around(at(100.0), days(5.0)).unwrap(),
    ];
    let flags = flag_times(&lc, &c, &windows);
    let got: Vec<(i64, Vec<usize>)> = flags.iter().map(|f| (f.epoch, f.windows.clone())).collect();
    // By hand: each predicted transit (as evidence's) against each window.
    let predicted: Vec<i64> = rustellar_vet::evidence(&lc, &c, rustellar_vet::Shape::BOX)
        .unwrap()
        .transits
        .iter()
        .map(|t| t.epoch)
        .collect();
    let mut want = Vec::new();
    for &n in &predicted {
        let centre = 1.0 + 3.0 * n as f64;
        let hit: Vec<usize> = windows
            .iter()
            .enumerate()
            .filter(|(_, w)| {
                w.start().as_btjd() < centre + 0.0625 && w.end().as_btjd() > centre - 0.0625
            })
            .map(|(k, _)| k)
            .collect();
        if !hit.is_empty() {
            want.push((n, hit));
        }
    }
    assert_eq!(got, want);
    assert_eq!(
        got,
        vec![(0, vec![3]), (1, vec![0, 3]), (2, vec![2]), (3, vec![4])]
    );
    assert_eq!(flags[1].time, at(4.0));
    assert!(flag_times(&lc, &c, &[]).is_empty());
}

#[test]
fn window_errors() {
    assert_eq!(
        TimeWindow::new(at(2.0), at(1.0)),
        Err(VetError::WindowReversed)
    );
    assert_eq!(
        TimeWindow::around(at(2.0), days(-0.1)),
        Err(VetError::WindowReversed)
    );
    let w = TimeWindow::around(at(2.0), days(0.5)).unwrap();
    assert_eq!((w.start(), w.end()), (at(1.5), at(2.5)));
    assert_eq!(
        VetError::WindowReversed.to_string(),
        "a time window must not end before it starts"
    );
}
