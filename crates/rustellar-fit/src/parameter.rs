use std::fmt;

/// One parameter of the transit fit.
///
/// # Examples
///
/// ```
/// use rustellar_fit::Parameter;
///
/// assert_eq!(Parameter::ALL.len(), 8);
/// assert_eq!(Parameter::RadiusRatio.to_string(), "k");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Parameter {
    /// The time of mid-transit, in days.
    MidTime,
    /// The orbital period, in days.
    Period,
    /// The planet's radius over the star's, `k`.
    RadiusRatio,
    /// The orbit's radius over the star's, `a / R*`.
    SemiMajorAxis,
    /// The impact parameter `b`, in stellar radii.
    ImpactParameter,
    /// The linear limb-darkening coefficient `u1`.
    U1,
    /// The quadratic limb-darkening coefficient `u2`.
    U2,
    /// The flux out of transit, which the model is multiplied by.
    Baseline,
}

impl Parameter {
    /// Every parameter, in the order fits report them.
    pub const ALL: [Self; 8] = [
        Self::MidTime,
        Self::Period,
        Self::RadiusRatio,
        Self::SemiMajorAxis,
        Self::ImpactParameter,
        Self::U1,
        Self::U2,
        Self::Baseline,
    ];

    /// The parameter's place in [`ALL`](Self::ALL).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The parameter's short name: `t0`, `P`, `k`, `a_rs`, `b`, `u1`, `u2`,
    /// `baseline`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MidTime => "t0",
            Self::Period => "P",
            Self::RadiusRatio => "k",
            Self::SemiMajorAxis => "a_rs",
            Self::ImpactParameter => "b",
            Self::U1 => "u1",
            Self::U2 => "u2",
            Self::Baseline => "baseline",
        }
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
