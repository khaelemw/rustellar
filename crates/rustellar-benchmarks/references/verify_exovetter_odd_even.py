"""Checks five suspected bugs in exovetter 0.0.16's odd/even test by running
its own functions (calc_odd_even, avg_odd_even, compute_phases).

Usage: python crates/rustellar-benchmarks/references/verify_exovetter_odd_even.py
"""

import numpy as np
from exovetter import odd_even
from exovetter.transit_coverage import compute_phases

P, T0, DUR = 3.0, 1.5, 0.1  # days
STEP = 2.0 / 1440.0  # 2-minute cadence
time = np.arange(0.0, 27.0, STEP)


def from_transit(t, epoch=T0):
    return (t - epoch + 0.5 * P) % P - 0.5 * P


def box(depth):
    return 1.0 - depth * (np.abs(from_transit(time)) < 0.5 * DUR)


print("exovetter odd/even, 27 days at 2 minutes, P 3 d, duration 0.1 d\n")

# (a) The window: frac * 0.5 * duration / period, but on phases of twice the
# period, so twice the time intended. Flux = |time from mid-transit|, so the
# median it returns is the median |dt| of the points it picks.
phases = compute_phases(time, 2 * P, T0, offset=0.25)
dt = np.abs(from_transit(time))
for frac in (0.5, 0.3):
    odd, even = odd_even.avg_odd_even(phases, dt, DUR / P, frac=frac, event_phase=0.25)
    intended = frac * 0.5 * DUR  # half-width of the window meant
    print(f"(a) frac {frac}: median |dt| picked {odd[0]:.4f} d; "
          f"a window of +/-{intended:.4f} d gives {intended / 2:.4f}, "
          f"twice that {intended:.4f}")

# (b) The error: the scatter of single points out of transit, used as each
# depth's error. Over 500 planets (equal depths) in noise 1e-3, sigma
# = |odd - even| / err should be half-normal (mean 0.80).
rng = np.random.default_rng(1)
sigmas, diffs, errs = [], [], []
for _ in range(500):
    flux = box(3e-3) + 1e-3 * rng.standard_normal(time.size)
    s, odd, even = odd_even.calc_odd_even(time, flux, P, T0, DUR)
    sigmas.append(s)
    diffs.append(odd[0] - even[0])
    errs.append(np.hypot(odd[1], even[1]))
print(f"\n(b) mean sigma {np.mean(sigmas):.3f} (should be 0.80 for a planet); "
      f"scatter of odd - even {np.std(diffs):.2e} against its stated error "
      f"{np.mean(errs):.2e} ({np.mean(errs) / np.std(diffs):.0f}x too large)")
# An eclipsing binary with depths 3e-3 and 2.7e-3: 10% apart.
flux = 1.0 - np.where(np.abs(from_transit(time)) < 0.5 * DUR,
                      np.where(np.round((time - T0) / P) % 2 == 0, 3e-3, 2.7e-3), 0.0)
flux = flux + 1e-3 * rng.standard_normal(time.size)
s, odd, even = odd_even.calc_odd_even(time, flux, P, T0, DUR)
print(f"    a binary with depths 10% apart (3e-4, about 3.8 standard errors): sigma {s:.2f}")

# (c) The depth: abs(median flux), so for flux normalised to 1 (as the
# OddEven vetter passes it) the "depth" is the flux in transit.
s, odd, even = odd_even.calc_odd_even(time, box(3e-3), P, T0, DUR)
print(f"\n(c) a 3e-3 transit, flux normalised to 1: odd depth {odd[0]:.6f}, "
      f"even depth {even[0]:.6f}")

# (d) Fewer than two points in a window: depths [1, 1] silently, sigma 0.
gap = (time < 10.0)  # the light curve stops before the second transit
s, odd, even = odd_even.calc_odd_even(time[gap & (time < 4.0)], box(3e-3)[gap & (time < 4.0)],
                                      P, T0, DUR)
print(f"\n(d) one transit only: odd {odd}, even {even}, sigma {s}")

# (e) np.fmod keeps the sign: times more than a quarter of 2P before the
# epoch get negative phases and fall in no window, so those transits are
# lost.
epoch = T0 + 4 * P  # a transit in the middle of the light curve
phases = compute_phases(time, 2 * P, epoch, offset=0.25)
in_transit = np.abs(from_transit(time, epoch)) < 0.5 * DUR
lost = in_transit & (phases < 0)
print(f"\n(e) epoch at {epoch} d: {lost.sum()} of {in_transit.sum()} points in transit "
      f"have negative phases ({len(np.unique(np.round((time[lost] - epoch) / P)))} of "
      f"{len(np.unique(np.round((time[in_transit] - epoch) / P)))} transits)")
s1, o1, e1 = odd_even.calc_odd_even(time, box(3e-3) - 1, P, T0, DUR)
s2, o2, e2 = odd_even.calc_odd_even(time, box(3e-3) - 1, P, epoch, DUR)
print(f"    the same planet (flux - 1): depths {o1[0]:.2e}/{e1[0]:.2e} from the first epoch, "
      f"{o2[0]:.2e}/{e2[0]:.2e} from the middle one")
