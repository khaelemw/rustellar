## rustellar-vet speed

| Step | sector, 2 minutes, limb-darkened | sector, 2 minutes, box | four years, 30 minutes, limb-darkened | four years, 30 minutes, box |
|---|--:|--:|--:|--:|
| `fit_depth` | 0.21 ms | 0.18 ms | 0.81 ms | 0.69 ms |
| `refine` | 0.15 s | 1.2 ms | 0.56 s | 17 ms |
| `evidence` | 0.55 ms | 0.52 ms | 3.1 ms | 3.0 ms |
| `signal_to_noise` | 0.87 ms | 0.85 ms | 6.1 ms | 5.9 ms |
| `aliases` | 15 ms | 13 ms | 85 ms | 81 ms |
| `secondary` | 0.75 ms | - | 4.9 ms | - |

The limb-darkened `refine` with `refine_with_threads` on 16 threads: 16 ms on the sector, 2 minutes, 55 ms on the four years, 30 minutes.

(33 s)

