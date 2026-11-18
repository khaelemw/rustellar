// The cells search's periodic stage on the GPU (cells.rs, Search::sums and
// the scan after it): one block a period, each thread a trial time at a
// time, adding its transits' cells oldest first. Each orbit's place among
// the cells (three divisions, slow in doubles on most GPUs) is worked out
// once for the block, in shared memory, not once for each trial time.
//
// Compiled by NVRTC with --fmad=false, and each fused multiply-add the CPU
// makes made here with fma(), so every sum is the CPU's to the bit. The
// best of a period is the first of the highest gains, as the CPU's scan
// finds it.

#ifndef THREADS
#define THREADS 256
#endif

extern "C" __global__ void periodic(
    const double* __restrict__ rs, const double* __restrict__ ss,
    double start, double step,
    const double* __restrict__ periods, const long long* __restrict__ deltas,
    const long long* __restrict__ counts,
    double origin, double from, double to, double min_depth,
    long long* __restrict__ best_k, double* __restrict__ best_gain,
    double* __restrict__ best_depth)
{
    // A chunk of orbits, each worked out once for all the trial times: the
    // trial times its transit is in, and its cell and fraction.
    __shared__ double firsts[THREADS];
    __shared__ double lasts[THREADS];
    __shared__ long long wholes[THREADS];
    __shared__ double fractions[THREADS];
    const int t = threadIdx.x;
    const int job = blockIdx.x;
    const double period = periods[job];
    const long long delta = deltas[job];
    const long long count = counts[job];
    const double spacing = (double)delta * step;
    const double highest = (double)(count - 1);

    double gain_best = 0.0, depth_best = 0.0;
    long long k_best = -1;
    for (long long k0 = 0; k0 < count; k0 += THREADS) {
        const long long k = k0 + t;
        const double kk = (double)k;
        double r = 0.0, s = 0.0;
        for (long long o0 = 0;; o0 += THREADS) {
            // The CPU counts orbits from -1 in steps of 1.0: whole numbers,
            // exact.
            const double orbit = (double)(o0 + t) - 1.0;
            const double base = fma(period, orbit, origin);
            const int kept = base <= to;
            if (kept) {
                firsts[t] = fmax(ceil((from - base) / spacing), 0.0);
                lasts[t] = fmin(floor((to - base) / spacing), highest);
                const double u = (base - start) / step;
                const double whole = floor(u);
                wholes[t] = (long long)whole;
                fractions[t] = u - whole;
            }
            // The orbits up to the first past the light curve.
            const int orbits = __syncthreads_count(kept);
            if (k < count) {
                for (int i = 0; i < orbits; i++) {
                    if (kk < firsts[i] || kk > lasts[i]) continue;
                    const long long j = wholes[i] + k * delta;
                    const double f = fractions[i];
                    r += fma(f, rs[j + 1] - rs[j], rs[j]);
                    s += fma(f, ss[j + 1] - ss[j], ss[j]);
                }
            }
            __syncthreads();
            if (orbits < THREADS) break;
        }
        if (k < count && r > 0.0) {
            const double depth = r / s;
            const double gain = r * depth;
            if (depth > min_depth && gain > gain_best) {
                gain_best = gain;
                depth_best = depth;
                k_best = k;
            }
        }
    }

    // The highest gain, the first trial time of equals.
    __shared__ double gains[THREADS];
    __shared__ double depths[THREADS];
    __shared__ long long ks[THREADS];
    gains[t] = gain_best;
    depths[t] = depth_best;
    ks[t] = k_best;
    __syncthreads();
    for (int half = THREADS / 2; half > 0; half /= 2) {
        if (t < half) {
            const int o = t + half;
            if (ks[o] >= 0 && (ks[t] < 0 || gains[o] > gains[t]
                    || (gains[o] == gains[t] && ks[o] < ks[t]))) {
                gains[t] = gains[o];
                depths[t] = depths[o];
                ks[t] = ks[o];
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        best_k[job] = ks[0];
        best_gain[job] = gains[0];
        best_depth[job] = depths[0];
    }
}
