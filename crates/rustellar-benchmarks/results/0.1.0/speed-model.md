## rustellar-model speed

| | 100 times | 1,000 | 10,000 | 100,000 |
|---|--:|--:|--:|--:|
| rustellar-model | 4.1 µs | 42 µs | 0.41 ms | 4.1 ms |
| batman 2.5.1 | 7.5 µs | 65 µs | 0.63 ms | 6.4 ms |
| PyTransit 2.6.18 | 8.7 µs | 50 µs | 0.46 ms | 4.6 ms |
| rustellar-model, flux only (no orbit) | 3.0 µs | 30 µs | 0.30 ms | 3.0 ms |
| exoplanet-core 0.4.0, flux only (no orbit) | 18 µs | 48 µs | 0.34 ms | 3.9 ms |

Kepler long cadence (29.4 min), 1,000 times:

| Samples per exposure | 5 | 15 | 30 |
|---|--:|--:|--:|
| rustellar-model | 0.18 ms | 0.50 ms | 0.98 ms |
| batman 2.5.1 | 0.33 ms | 0.98 ms | 2.0 ms |
| PyTransit 2.6.18 | 0.20 ms | 0.59 ms | 1.2 ms |

(0 s)

