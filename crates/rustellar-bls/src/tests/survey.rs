// The survey recipe's structure: every point kept at keep 1, each time
// within its own segment, in order, and the box only where it should be.

use super::recipe::Survey;

#[test]
fn survey_recipe_structure() {
    let survey = Survey {
        seed: 3,
        segments: vec![(1325.0, 200), (1330.0, 100)],
        cadence: 0.02,
        noise: 1e-3,
        keep: 1.0,
        period: 0.7,
        tc: 1325.3,
        duration: 0.1,
        depth: 0.01,
    };
    let lc = survey.light_curve();
    assert_eq!(lc.len(), 300);
    let time: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
    assert!(time.windows(2).all(|w| w[0] < w[1]));
    // Each segment from its start to one jittered cadence past its last.
    assert!(time[..200].iter().all(|&t| (1325.0..1329.01).contains(&t)));
    assert!(time[200..].iter().all(|&t| (1330.0..1332.01).contains(&t)));
    // The noise is under 1e-3, so a flux below -5e-3 is in the box.
    for (&t, &y) in time.iter().zip(lc.flux()) {
        let r = (t - 1325.3) / 0.7;
        let off = (r - r.round()).abs() * 0.7;
        assert_eq!(y < -5e-3, off < 0.05, "t {t}");
    }
    // Thinned, the same draws: a subset of the same times.
    let thinned = Survey {
        keep: 0.3,
        ..survey
    }
    .light_curve();
    assert!(thinned.len() < 150);
    assert!(thinned.time().iter().all(|t| time.contains(&t.as_btjd())));
}
