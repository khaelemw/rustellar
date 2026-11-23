## rustellar-mcmc speed

| Sampler | Move | tau | Time per independent sample |
|---|---|--:|--:|
| emcee 3.1.6 + batman 2.5.1 | stretch | 552 | 39 ms |
| emcee 3.1.6 + batman 2.5.1 | de | 239 | 16 ms |
| emcee 3.1.6 + PyTransit 2.6.18 | stretch | 543 | 24 ms |
| emcee 3.1.6 + PyTransit 2.6.18 | de | 231 | 9.4 ms |
| zeus 2.5.4 + batman 2.5.1 | slice | 128 | 44 ms |
| rustellar-mcmc, 1 thread | stretch | 520 | 7.9 ms |
| rustellar-mcmc, 4 threads | stretch | 520 | 3.8 ms |
| rustellar-mcmc, 1 thread | de | 241 | 3.3 ms |
| rustellar-mcmc, 4 threads | de | 241 | 1.7 ms |
| rustellar-mcmc, 1 thread | slice | 140 | 12 ms |
| rustellar-mcmc, 4 threads | slice | 140 | 4.9 ms |

(170 s)

