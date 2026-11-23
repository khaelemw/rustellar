## rustellar-bls against rustellar-tls on limb-darkened transits

SDE at 1% false alarms: rustellar-bls 5.03, rustellar-tls 8.83.

| Transits | rustellar-bls recovered | P error | t0 error (min) | rustellar-tls recovered | P error | t0 error (min) |
|---|---|---|---|---|---|---|
| TESS 30 min, 1 sector | 236/250 | 3.64e-4 | 2.85 | 234/250 | 1.74e-4 | 2.09 |
| TESS 2 min, 10% kept | 221/250 | 3.46e-4 | 2.94 | 230/250 | 1.78e-4 | 1.69 |
| TESS 30 min, 2 sectors | 229/250 | 1.99e-4 | 2.88 | 229/250 | 9.94e-5 | 1.85 |
| ground, nights | 104/250 | 1.01e-4 | 1.48 | 145/250 | 4.56e-5 | 1.22 |
| snr below 10 | 140/233 | 3.90e-4 | 5.24 | 142/233 | 3.47e-4 | 3.63 |
| snr 10 to 20 | 447/521 | 2.45e-4 | 2.78 | 472/521 | 1.44e-4 | 2.13 |
| snr 20 and up | 203/246 | 2.01e-4 | 1.46 | 224/246 | 5.01e-5 | 0.74 |
| all | 790/1000 | 2.53e-4 | 2.63 | 838/1000 | 1.19e-4 | 1.73 |

On transitleastsquares 1.32's subset (SDE thresholds 5.03, 8.76, 9.01):

| Search | recovered | P error | t0 error (min) |
|---|---|---|---|
| rustellar-bls | 200/252 | 2.55e-4 | 2.17 |
| rustellar-tls | 212/252 | 1.07e-4 | 1.84 |
| transitleastsquares 1.32 | 212/252 | 2.52e-4 | 3.99 |

(2000 curves, 471 s)

