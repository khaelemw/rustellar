// The template's shape: its ends and middle exactly, symmetry, monotony,
// the planets it refuses, and the curved path of a real orbit.

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams, flux,
};

use super::Random;
use crate::template::{SAMPLED_INTERVALS, Sampled, Shape, Template};

fn tls_templates() -> [Template; 3] {
    [
        Template::tls_default(),
        Template::tls_grazing(),
        Template::tls_box(),
    ]
}

fn random_templates(count: usize) -> Vec<Template> {
    let mut random = Random::new(11);
    let mut templates = Vec::new();
    while templates.len() < count {
        let k = RadiusRatio::new(random.between(0.001, 0.5)).unwrap();
        let b = random.between(0.0, 1.0 + k.get());
        let ld = match random.next_u64() % 3 {
            0 => LimbDarkening::uniform(),
            1 => LimbDarkening::linear(random.uniform()).unwrap(),
            _ => {
                let u1 = random.between(0.0, 1.0);
                LimbDarkening::quadratic(u1, random.between(-0.5 * u1, 1.0 - u1)).unwrap()
            }
        };
        if let Some(template) = Template::new(k, b, ld) {
            templates.push(template);
        }
    }
    templates
}

#[test]
fn one_at_mid_transit_and_zero_from_contact() {
    for template in tls_templates().into_iter().chain(random_templates(500)) {
        assert_eq!(template.shape(0.0), 1.0, "{template:?}");
        assert_eq!(template.shape(-0.0), 1.0);
        for x in [0.5, -0.5, 0.500_000_1, 0.7, 3.0, -1e300, f64::INFINITY] {
            assert_eq!(template.shape(x), 0.0, "{template:?} at {x}");
        }
        assert_eq!(template.shape(f64::NAN), 0.0);
        // A hair inside contact the shape is tiny but not negative.
        let inside = template.shape(0.499_999_999);
        assert!((0.0..1e-3).contains(&inside), "{inside}");
    }
}

#[test]
fn symmetric_bounded_and_falling_away_from_the_middle() {
    for template in tls_templates().into_iter().chain(random_templates(200)) {
        let mut last = 1.0;
        for i in 0..=2000 {
            let x = f64::from(i) / 4000.0;
            let s = template.shape(x);
            assert_eq!(s, template.shape(-x), "{template:?} at {x}");
            assert!((0.0..=1.0).contains(&s), "{s}");
            assert!(s <= last, "{template:?}: {s} after {last} at {x}");
            last = s;
        }
    }
}

#[test]
fn the_depth_is_the_planets_dimming() {
    let k = RadiusRatio::new(0.1).unwrap();
    let ld = LimbDarkening::quadratic(0.4, 0.2).unwrap();
    for b in [0.0, 0.3, 0.9, 1.05] {
        let template = Template::new(k, b, ld).unwrap();
        assert_eq!(template.depth(), 1.0 - flux(b, k, ld));
    }
    // The box template: a uniform disk, so k².
    assert!((Template::tls_box().depth() - 0.01).abs() < 1e-16);
}

#[test]
fn the_box_template_is_flat_between_second_and_third_contact() {
    // k = 0.1, b = 0: fully inside the disk while z <= 1 - k, which is
    // |x| <= (1 - k) / (1 + k) / 2 = 0.409…
    let template = Template::tls_box();
    for i in 0..=400 {
        let x = 0.409 * f64::from(i) / 400.0;
        assert_eq!(template.shape(x), 1.0, "{x}");
    }
    assert!(template.shape(0.4095) < 1.0);
    // Uniform disk, so the ingress is the covered area: with the planet's
    // centre on the star's edge (z = 1), a little under half its disk, as
    // the edge curves away from it.
    let half = template.shape(0.5 / 1.1);
    assert!((0.48..0.5).contains(&half), "{half}");
}

#[test]
fn refuses_planets_that_do_not_transit() {
    let k = RadiusRatio::new(0.1).unwrap();
    let ld = LimbDarkening::uniform();
    for b in [
        -1e-300,
        -0.1,
        1.1,
        1.1 + 1e-15,
        2.0,
        f64::NAN,
        f64::INFINITY,
    ] {
        assert!(Template::new(k, b, ld).is_none(), "{b}");
    }
    assert!(Template::new(k, 1.099_999_999, ld).is_some());
    // So small a planet that its flux is 1 in doubles: nothing to fit.
    let tiny = RadiusRatio::new(1e-9).unwrap();
    assert!(Template::new(tiny, 0.0, ld).is_none());
}

// The template assumes a straight path; a real orbit curves. For TLS's own
// planet (a = 23.1, P = 12.9 d) the curve changes the shape by under 3e-4,
// and by more for a close-in planet, as it should.
#[test]
fn close_to_the_curved_path_of_tls_planet() {
    let worst = |a_rs: f64, b: f64, period: f64| {
        let k = RadiusRatio::new(0.03).unwrap();
        let ld = LimbDarkening::quadratic(0.4804, 0.1867).unwrap();
        let t0 = Epoch::from_days(0.0).unwrap();
        let params = TransitParams::new(
            t0,
            TimeDelta::from_days(period).unwrap(),
            k,
            ScaledSemiMajorAxis::new(a_rs).unwrap(),
            ImpactParameter::new(b).unwrap(),
            ld,
        )
        .unwrap();
        let t14 = params.t14().unwrap().as_days();
        let template = Template::new(k, b, ld).unwrap();
        let bottom = 1.0 - params.flux_at(t0);
        (0..=1000)
            .map(|i| {
                let x = f64::from(i) / 2000.0;
                let t = Epoch::from_days(x * t14).unwrap();
                let curved = (1.0 - params.flux_at(t)) / bottom;
                (curved - template.shape(x)).abs()
            })
            .fold(0.0, f64::max)
    };
    let tls = worst(23.1, 0.318_495_043_365_240_5, 12.9);
    assert!(tls < 3e-4, "{tls}");
    let close = worst(3.0, 0.3, 0.5);
    assert!(close > tls, "{close} {tls}");
}

// transitleastsquares 1.32's templates (`reference_transit` at 201 samples,
// as 1 - flux), every 20th sample: x from -1/2 to 1/2 in steps of 1/10.
// TLS samples batman's curve at 10,000 points over a day and interpolates
// it twice, starting at the first sample below 1, so its ends are not 0
// and its two halves differ in the last digits. Its path curves, so it is
// a little wider than ours: by up to 2.8e-4 for the default planet (at the
// ends) and 1.7e-3 for the grazing one, whose shape the curve changes most.
const TLS_DEFAULT: [f64; 11] = [
    0.000_280_612_668_046_242_85,
    0.754_020_970_941_192_5,
    0.887_173_306_739_386_1,
    0.955_105_984_794_026_2,
    0.989_413_563_612_650_4,
    1.0,
    0.989_413_563_612_650_4,
    0.955_105_984_794_026_2,
    0.887_173_306_739_386_1,
    0.754_020_970_941_294_8,
    0.000_280_612_668_046_242_85,
];
const TLS_GRAZING: [f64; 11] = [
    0.000_130_527_308_953_576_34,
    0.224_268_639_912_206_8,
    0.531_701_492_828_637,
    0.787_159_899_862_321_9,
    0.946_683_827_757_749_1,
    1.0,
    0.946_683_827_757_749_1,
    0.787_159_899_862_321_9,
    0.531_701_492_828_339_7,
    0.224_268_639_912_206_8,
    0.000_130_527_308_953_576_34,
];

#[test]
fn close_to_tls_resampled_templates() {
    for (template, theirs, bound) in [
        (Template::tls_default(), TLS_DEFAULT, 3e-4),
        (Template::tls_grazing(), TLS_GRAZING, 2e-3),
    ] {
        let mut worst: f64 = 0.0;
        for (i, &s) in theirs.iter().enumerate() {
            let x = f64::from(u8::try_from(i).unwrap()) / 10.0 - 0.5;
            worst = worst.max((template.shape(x) - s).abs());
        }
        assert!(worst < bound, "{worst}");
    }
    // The box: flat at 1 from x = -0.4 to 0.4 in both.
    let template = Template::tls_box();
    for i in 1..10 {
        assert_eq!(template.shape(f64::from(i) / 10.0 - 0.5), 1.0);
    }
}

// The sampled template against the template itself, at 2^16 intervals in
// each part: within 1.8e-10, 9.8e-11 and 1.9e-10 for TLS's default, grazing
// and box templates, the worst at second contact. Sampled evenly in time it
// was 1.1e-7 there (the shape goes as a distance to the power 3/2).
#[test]
fn the_sampled_template_is_close_to_the_template() {
    let mut random = Random::new(61);
    for template in [
        Template::tls_default(),
        Template::tls_grazing(),
        Template::tls_box(),
    ] {
        let sampled = Sampled::new(&template, SAMPLED_INTERVALS);
        let mut worst: f64 = 0.0;
        for i in 0..200_000 {
            let x = if i % 2 == 0 {
                random.between(-0.5, 0.5)
            } else {
                // Near the contacts.
                random.between(0.45, 0.5) * if i % 4 == 1 { 1.0 } else { -1.0 }
            };
            worst = worst.max((sampled.shape(x) - template.shape(x)).abs());
        }
        assert!(worst < 3e-10, "{worst}");
        assert_eq!(sampled.shape(0.0), 1.0);
        for x in [0.5, -0.5, 0.7, -3.0, f64::NAN, f64::INFINITY] {
            assert_eq!(sampled.shape(x), 0.0, "{x}");
        }
        assert_eq!(sampled.shape(0.123), sampled.shape(-0.123));
    }
}
