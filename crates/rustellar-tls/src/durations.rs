//! Trial durations: TLS's grid, even in the logarithm of the duration as a
//! fraction of the period, between the shortest and the longest transit a
//! range of stars and planets allows.

use std::f64::consts::PI;

const SECONDS_PER_DAY: f64 = 86_400.0;

/// The longest transit ever searched for, as a fraction of the period
/// (TLS's `FRACTIONAL_TRANSIT_DURATION_MAX`): longer ones cost the most and
/// are rare (Kepler-1368 b, 0.117).
pub(crate) const FRACTION_MAX: f64 = 0.12;

/// The physical constants a duration is computed from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Constants {
    /// The Sun's mass parameter `G M`, in m³ s⁻².
    pub(crate) gm_sun: f64,
    /// The Sun's radius, in metres.
    pub(crate) r_sun: f64,
    /// Jupiter's radius, in metres.
    pub(crate) r_jupiter: f64,
}

impl Constants {
    /// IAU 2015 Resolution B3 nominal solar values (as `rustellar-bls`'s
    /// Ofir grid), and Jupiter's mean radius, 69,911 km (as TLS).
    pub(crate) const IAU: Self = Self {
        gm_sun: 1.327_124_4e20,
        r_sun: 6.957e8,
        r_jupiter: 6.9911e7,
    };
}

/// The size of planet a duration bound is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Planet {
    /// A planet of negligible size: the shortest central transit.
    Small,
    /// Twice Jupiter's radius: the longest.
    Large,
}

/// The duration, as a fraction of the period, of a central transit across
/// a star of `r_star` solar radii and `m_star` solar masses at a period of
/// `period` days, capped at [`FRACTION_MAX`] (TLS's `T14`).
///
/// On a circular orbit of radius `a = (G M P² / 4π²)^(1/3)`, a central
/// transit lasts `P (R + r) / (π a)`, which is `(R + r) (4 P / (π G M))^(1/3)`
/// over `P`.
pub(crate) fn t14(
    constants: Constants,
    r_star: f64,
    m_star: f64,
    period: f64,
    planet: Planet,
) -> f64 {
    let seconds = period * SECONDS_PER_DAY;
    let radius = r_star * constants.r_sun
        + match planet {
            Planet::Small => 0.0,
            Planet::Large => 2.0 * constants.r_jupiter,
        };
    let fraction = radius * (4.0 * seconds / (PI * m_star * constants.gm_sun)).cbrt() / seconds;
    fraction.min(FRACTION_MAX)
}

/// The range of stars the durations cover, in solar radii and masses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Stars {
    pub(crate) r_min: f64,
    pub(crate) r_max: f64,
    pub(crate) m_min: f64,
    pub(crate) m_max: f64,
}

impl Stars {
    /// TLS's defaults: 0.13 to 3.5 solar radii, 0.1 to 1 solar mass.
    pub(crate) const TLS: Self = Self {
        r_min: 0.13,
        r_max: 3.5,
        m_min: 0.1,
        m_max: 1.0,
    };

    /// The shortest transit at `period`: a small planet across the smallest,
    /// lightest star.
    pub(crate) fn shortest(self, constants: Constants, period: f64) -> f64 {
        t14(constants, self.r_min, self.m_min, period, Planet::Small)
    }

    /// The longest transit at `period`: a large planet across the largest,
    /// heaviest star.
    pub(crate) fn longest(self, constants: Constants, period: f64) -> f64 {
        t14(constants, self.r_max, self.m_max, period, Planet::Large)
    }
}

/// TLS's duration grid, as fractions of the period: from the shortest
/// transit at the longest period to the longest at the shortest period,
/// each `step` times the one before, and the longest added at the end
/// (closer to the one before it than `step`).
///
/// `step` must be greater than 1, and the periods positive and finite, with
/// `shortest_period <= longest_period`.
pub(crate) fn grid(
    constants: Constants,
    stars: Stars,
    shortest_period: f64,
    longest_period: f64,
    step: f64,
) -> Vec<f64> {
    debug_assert!(step > 1.0 && shortest_period <= longest_period);
    let longest = stars.longest(constants, shortest_period);
    let mut duration = stars.shortest(constants, longest_period);
    let mut durations = vec![duration];
    while duration * step < longest {
        duration *= step;
        durations.push(duration);
    }
    durations.push(longest);
    durations
}

/// The same grid in days, for searches that keep each duration across
/// periods: from the shortest transit at the shortest period to the longest
/// at the longest period, widened by `(n + 1) / n` for `n` the periods in
/// `span` (as the searches widen it at each period) and kept below the
/// longest period, each `step` times the one before, the longest added at
/// the end.
///
/// `step` must be greater than 1, the periods and `span` positive and
/// finite, with `shortest_period <= longest_period`.
pub(crate) fn grid_days(
    constants: Constants,
    stars: Stars,
    shortest_period: f64,
    longest_period: f64,
    span: f64,
    step: f64,
) -> Vec<f64> {
    debug_assert!(step > 1.0 && shortest_period <= longest_period && span > 0.0);
    let orbits = span / longest_period;
    let longest = (stars.longest(constants, longest_period) * longest_period * (orbits + 1.0)
        / orbits)
        .min(f64::from_bits(longest_period.to_bits() - 1));
    let mut duration = stars.shortest(constants, shortest_period) * shortest_period;
    let mut durations = vec![duration];
    while duration * step < longest {
        duration *= step;
        durations.push(duration);
    }
    if longest > duration {
        durations.push(longest);
    }
    durations
}
