# Scope

## Done means

- Primary-transit flux for uniform, linear and quadratic limb darkening, on
  circular orbits, for radius ratios up to 10,000.
- Exposure averaging, orbit geometry (separation, phase, in front or behind),
  durations T14 and T23, contact points and transit times.
- A model object for evaluating many parameter sets on fixed times without
  allocating.
- Accuracy measured against batman, PyTransit, exoplanet-core and an exact
  integral, with the comparisons in the test suite.

All of this is in place.

## May come later, in this crate

- Eccentric orbits.
- Other limb-darkening laws (nonlinear, power-2).
- Secondary eclipses and light from the planet.

Each would be added without breaking the existing API.

## Not in this crate

- Fitting and sampling: separate rustellar crates, built on this one.
- Transit searches (BLS, TLS) and detrending: separate rustellar crates.
- Light-curve containers and time types: `rustellar-core`.
- Threads, plotting and file input.
