## rustellar-vet speed

| Step | sector, 2 minutes, limb-darkened | sector, 2 minutes, box | four years, 30 minutes, limb-darkened | four years, 30 minutes, box |
|---|--:|--:|--:|--:|
| `fit_depth` | 0.21 ms | 0.18 ms | 0.81 ms | 0.70 ms |
| `refine` | 0.15 s | 1.2 ms | 0.56 s | 17 ms |
| `evidence` | 0.56 ms | 0.53 ms | 2.9 ms | 2.8 ms |
| `signal_to_noise` | 0.90 ms | 0.87 ms | 6.0 ms | 5.9 ms |
| `aliases` | 15 ms | 14 ms | 84 ms | 80 ms |
| `secondary` | 0.77 ms | - | 5.0 ms | - |

The limb-darkened `refine` with `refine_with_threads` on 16 threads: 17 ms on the sector, 2 minutes, 57 ms on the four years, 30 minutes.

(33 s)

