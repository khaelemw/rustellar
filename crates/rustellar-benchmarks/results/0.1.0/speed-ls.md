## rustellar-ls speed

| Points x frequencies | exact, 1 thread | exact, 16 threads | astropy 7.2.0 slow | astropy 7.2.0 cython |
|---|--:|--:|--:|--:|
| 1,000 x 10,000 | 65 ms | 8.5 ms | 0.51 s | 0.23 s |
| 10,000 x 10,000 | 0.66 s | 88 ms | 5.4 s | 2.3 s |

The fast method on 16 threads, against astropy's (which has no tolerance):

| Points x frequencies | tolerance 1e-6 | tolerance 1e-12 | astropy 7.2.0 fast |
|---|--:|--:|--:|
| 1,000 x 10,000 | 3.1 ms | 3.1 ms | 7.7 ms |
| 10,000 x 10,000 | 3.7 ms | 4.4 ms | 9.4 ms |
| 10,000 x 100,000 | 28 ms | 29 ms | 80 ms |
| 100,000 x 100,000 | 37 ms | 41 ms | 0.12 s |
| 1,000,000 x 1,000,000 | 0.34 s | 0.37 s | 2.0 s |

(7 s)

