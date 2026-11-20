# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Detrend`, `Method`, `BreakTolerance`, `Edges` and `DetrendError`: a
  time-windowed sliding filter, its settings checked when used.
- `Detrend::trend`: the trend at each point, with an optional mask. The
  biweight correctly rounded wherever its sum's sign can be told from zero,
  the median and the mean correctly rounded, every window edge decided
  exactly; the same results, to the bit, on any number of threads.
- `Detrend::flatten` and `Flattened`: the light curve divided by its trend,
  with the original index of each point kept.
- `Method::HuberJoint`: Huber's proposal 2, the location and scale
  estimated together (statsmodels' `Huber`, wotan's `method='huber'`),
  solved exactly: each split of the points solved as a quadratic, the last
  in double-double, with bisection on the scale where the splits do not
  settle.
- `transit_mask`: the points within half a duration of a candidate's
  mid-transit times, to detrend again with its transits left out.
- `Detrend::with_window_for_transit` and `WINDOW_PER_DURATION` (2.5): a
  window from the longest transit duration searched for, chosen by a blind
  BLS search on 400 injected curves (in the README).
- More methods: Welsch's, Andrews' sine wave and Huber's M-estimators
  (correctly rounded wherever their sums' signs can be told from zero, as the
  biweight), and the Hodges-Lehmann estimator and the trimmed and winsorized
  means (correctly rounded). The plain series run four to a vector with AVX2
  where the processor has it, to the same bits as without, and so do the
  double-double sums that settle the last bit, sixteen terms side by side.
- `Method::Spline`: a least-squares cubic spline per segment, knots at most
  a window apart, masked points left out; banded Givens QR with exact knot
  offsets and two refinement steps, about an ulp of the data's scale (scipy's
  FITPACK about 17), and kept smooth where the points leave it undetermined.
- `Method::RobustSpline`: the spline refitted without points beyond `sigma`
  robust standard deviations of it, both sides, until the kept set repeats;
  matches the exact loop to 2 ulps, unbiased on noise.
- `Method::HuberSpline`: Huber's M-estimate of the spline (statsmodels'
  `RLM(HuberT)` with its MAD scale), solved to its fixed point by Newton's
  steps on the points' zones; 67 times faster than statsmodels.
- `Detrend::flatten_around`, `Masking`, `Candidate` and `Masked`: flatten,
  run any transit search, mask its candidate's transits (1.5 durations,
  windows of 2.5) and flatten again, optionally searching again until the
  mask repeats; keeps 98% of the signal to noise on 400 injected curves,
  as much as masking the true transits (in the README).
- `DetrendError::MaskNotInWindow`.
- `Autoregressive`, `Order`, `Whitened` and `MAX_ORDER` (30): an
  autoregressive model fitted by least squares per run of regular cadence,
  its order by AIC (as statsmodels' `ar_select_order`), and the light
  curve whitened by it; coefficients within 0.16 ulps of the
  prediction's largest term (statsmodels up to 4,122), 30 times faster. `DetrendError::TooFewPoints`.
