// The binned search on the GPU: one period a block, every bin in shared
// memory, the best box's duration and start for each period.
//
// Compiled by NVRTC with --fmad=false, so the CPU's arithmetic, where it is
// repeated here, rounds the same way.
//
// - Bins hold the weights and centred fluxes in 64-bit fixed point (each a
//   whole number of quanta, powers of two so small that no total reaches
//   2^62): integer atomics, exact whatever order the points come in, and
//   exact running totals.
// - A point's bin is found from integers: the orbit's fraction is 64 bits of
//   S G, S the elapsed time in units of 2^-40 days and G a 64-bit 2^c / p,
//   and the bin the high word of fraction m. Within 2^-24 bins of an edge,
//   the CPU's own arithmetic decides, so every point is in the CPU's bin.
// - A period whose bins and longest box do not fit in shared memory goes a
//   tile of bins at a time; in each orbit, a tile's points are one run of
//   points in time order, found by a binary search of the times.
// - Boxes are scored in singles from the exact sums, and only those within
//   2^-12 of the thread's best so far again in doubles; the best is the
//   first of the highest in (duration, start) order.

#ifndef THREADS
#define THREADS 256
#endif
#define MAX_RUNS 1024

__device__ __forceinline__ double whole(double q) {
    const double TWO_52 = 4503599627370496.0;
    return (q + TWO_52) - TWO_52;
}

// The CPU's bin (binned.rs, bin_of with offset_after_start_short): the same
// operations in the same order.
__device__ __noinline__ int exact_bin(double s, double e, double inv, double p,
    double p_hi, double p_lo, double bin, int m) {
    const double MARGIN = 8.0 * 2.220446049250313e-16;
    const double q = s * inv;
    double n = whole(q);
    if (!(0.5 - fabs(q - n) > MARGIN * fabs(q))) n = whole(s / p);
    const double offset = ((s - n * p_hi) - n * p_lo) + e;
    const double phase = offset < 0.0 ? offset + p : offset;
    // As Rust's `as u32`: saturating, and NaN to 0.
    const double r = phase / bin;
    const unsigned int k = r > 0.0 ? (unsigned int)fmin(r, 4294967295.0) : 0u;
    return k > (unsigned int)(m - 1) ? m - 1 : (int)k;
}

// A point's bin by integers, or by exact_bin within 2^-24 bins of an edge.
__device__ __forceinline__ int bin_of(unsigned long long S, unsigned long long G, int shift,
    int m, double s, double e, double inv, double p, double p_hi, double p_lo, double bin) {
    const unsigned long long lo = S * G, hi = __umul64hi(S, G);
    const unsigned long long fraction = (hi << (64 - shift)) | (lo >> shift);
    const unsigned long long k = __umul64hi(fraction, (unsigned long long)m);
    const unsigned long long past = fraction * (unsigned long long)m;
    const unsigned long long EDGE = 1ULL << 40;
    if (past > EDGE && past < 0ULL - EDGE) return (int)k;
    return exact_bin(s, e, inv, p, p_hi, p_lo, bin, m);
}

// The first point whose elapsed time is not below x.
__device__ __forceinline__ int first_from(const double* s, int n, double x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (s[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Box a beats box b: higher, or as high and first in (duration, start) order.
__device__ __forceinline__ bool better(double a, int ad, int aj, double b, int bd, int bj) {
    return a > b || (a == b && a > 0.0 && ad >= 0 && (bd < 0 || ad < bd || (ad == bd && aj < bj)));
}

__device__ __forceinline__ void add(long long* W, long long* Y, int k, int j0, int m, int held,
    long long w, long long y) {
    int r = k - j0;
    if (r < 0) r += m;
    for (; r < held; r += m) {
        atomicAdd((unsigned long long*)&W[r], (unsigned long long)w);
        atomicAdd((unsigned long long*)&Y[r], (unsigned long long)y);
    }
}

// Per period b: periods[5b..5b+5] = p, 1/p, p_hi, p_lo, bin; bins[b] = m;
// reciprocals[b] = G and shifts[b] its shift; lengths[b nd + d] each
// duration's length in bins (-1 to skip). Out: the best box's score (0 if
// none has a dip), duration (-1 if none) and start.
extern "C" __global__ void binned(
    const double* __restrict__ s, const double* __restrict__ e,
    const unsigned long long* __restrict__ S,
    const long long* __restrict__ w, const long long* __restrict__ y, int n,
    double quantum_w, double quantum_y, long long total_w, long long total_y,
    const double* __restrict__ periods, const int* __restrict__ bins,
    const unsigned long long* __restrict__ reciprocals, const int* __restrict__ shifts,
    const int* __restrict__ lengths, int nd, int cap, int likelihood,
    double* out_score, int* out_duration, int* out_start)
{
    extern __shared__ long long shared[];   // W and Y, cap + 1 each
    __shared__ long long chunk_w[THREADS], chunk_y[THREADS];
    __shared__ double best_score[THREADS];
    __shared__ int best_duration[THREADS], best_start[THREADS];
    __shared__ int run_from[MAX_RUNS], run_to[MAX_RUNS + 1];
    __shared__ int runs;

    const int b = blockIdx.x, tid = threadIdx.x;
    const double p = periods[5 * b], inv = periods[5 * b + 1], p_hi = periods[5 * b + 2],
                 p_lo = periods[5 * b + 3], bin = periods[5 * b + 4];
    const int m = bins[b];
    const unsigned long long G = reciprocals[b];
    const int shift = shifts[b];
    int longest = 0;
    for (int d = 0; d < nd; d++) longest = max(longest, lengths[b * nd + d]);
    const double W_all = (double)total_w * quantum_w, S_all = (double)total_y * quantum_y;
    const double inverse_W = 1.0 / W_all;
    const float qw = (float)quantum_w, qy = (float)quantum_y, S_f = (float)S_all,
                inverse_W_f = (float)inverse_W, W_f = (float)W_all;
    long long* W = shared;
    long long* Y = shared + cap + 1;

    const bool whole_orbit = m + longest <= cap;
    const int tile = whole_orbit ? m : cap - longest;
    const double last = s[n - 1];
    const double margin = 64.0 * 2.220446049250313e-16 * (last + 4.0 * p);

    double best = 0.0;
    int bd = -1, bj = 0;
    float bar = 0.0f;
    long long carry_w = 0, carry_y = 0;   // the totals before the tile
    for (int j0 = 0; j0 < m; j0 += tile) {
        const int starts = min(tile, m - j0);
        const int held = starts + longest;   // the starts and the longest box past them
        for (int k = tid; k < held; k += THREADS) { W[k] = 0; Y[k] = 0; }
        bool every = whole_orbit;
        if (!whole_orbit) {
            const int first_orbit = -1, orbits = (int)ceil(last * inv) + 3;
            if (orbits > MAX_RUNS) {
                every = true;
            } else {
                for (int k = tid; k < orbits; k += THREADS) {
                    const double at = (double)(first_orbit + k) * p;
                    run_from[k] = first_from(s, n, at + (double)j0 * bin - margin);
                    run_to[k] = first_from(s, n, at + (double)(j0 + held) * bin + margin);
                }
                __syncthreads();
                if (tid == 0) {
                    // Disjoint runs, as offsets: run k is points
                    // run_from[k] + (g - run_to[k]) for g in run_to[k]..run_to[k + 1].
                    int previous = 0, total = 0;
                    for (int k = 0; k < orbits; k++) {
                        const int from = max(run_from[k], previous), to = max(run_to[k], from);
                        previous = to;
                        run_from[k] = from;
                        run_to[k] = total;
                        total += to - from;
                    }
                    run_to[orbits] = total;
                    runs = orbits;
                }
            }
        }
        __syncthreads();
        if (every) {
            for (int i = tid; i < n; i += THREADS) {
                const int k = bin_of(S[i], G, shift, m, s[i], e[i], inv, p, p_hi, p_lo, bin);
                add(W, Y, k, j0, m, held, w[i], y[i]);
            }
        } else {
            const int count = runs, total = run_to[count];
            for (int g = tid; g < total; g += THREADS) {
                int lo = 0, hi = count;
                while (hi - lo > 1) {
                    const int mid = (lo + hi) >> 1;
                    if (run_to[mid] <= g) lo = mid; else hi = mid;
                }
                const int i = run_from[lo] + (g - run_to[lo]);
                const int k = bin_of(S[i], G, shift, m, s[i], e[i], inv, p, p_hi, p_lo, bin);
                add(W, Y, k, j0, m, held, w[i], y[i]);
            }
        }
        __syncthreads();

        // Running totals from the carry: W[k] over the bins before k.
        const int size = (held + THREADS - 1) / THREADS;
        const int lo = min(tid * size, held), hi = min(lo + size, held);
        long long sum_w = 0, sum_y = 0;
        for (int k = lo; k < hi; k++) { sum_w += W[k]; sum_y += Y[k]; }
        chunk_w[tid] = sum_w; chunk_y[tid] = sum_y;
        __syncthreads();
        if (tid == 0) {
            long long at_w = carry_w, at_y = carry_y;
            for (int t = 0; t < THREADS; t++) {
                const long long a = chunk_w[t], c = chunk_y[t];
                chunk_w[t] = at_w; chunk_y[t] = at_y;
                at_w += a; at_y += c;
            }
            W[held] = at_w; Y[held] = at_y;
        }
        __syncthreads();
        {
            long long at_w = chunk_w[tid], at_y = chunk_y[tid];
            for (int k = lo; k < hi; k++) {
                const long long a = W[k], c = Y[k];
                W[k] = at_w; Y[k] = at_y;
                at_w += a; at_y += c;
            }
        }
        __syncthreads();

        for (int d = 0; d < nd; d++) {
            const int L = lengths[b * nd + d];
            if (L < 0) continue;
            for (int x = tid; x < starts; x += THREADS) {
                const long long a = W[x + L] - W[x], c = Y[x + L] - Y[x];
                if (!(a > 0 && total_w - a > 0)) continue;
                {
                    const float win = (float)a * qw, sin = (float)c * qy,
                                wout = (float)(total_w - a) * qw;
                    const float u = sin - S_f * (win * inverse_W_f);
                    float score = u * (u / (win * (wout * inverse_W_f)));
                    if (likelihood) score *= (0.5f * W_f) / wout;
                    if (!(u < 0.0f) || score < bar) continue;
                }
                const double win = (double)a * quantum_w, sin = (double)c * quantum_y;
                const double wout = (double)(total_w - a) * quantum_w;
                const double u = sin - S_all * (win * inverse_W);
                if (!(u < 0.0)) continue;
                double score = u * (u / (win * (wout * inverse_W)));
                if (likelihood) score *= (0.5 * W_all) / wout;
                if (better(score, d, j0 + x, best, bd, bj)) {
                    best = score; bd = d; bj = j0 + x;
                    bar = (float)best * (1.0f - 1.0f / 4096.0f);
                }
            }
        }
        carry_w = W[starts]; carry_y = Y[starts];
        __syncthreads();
    }

    best_score[tid] = best; best_duration[tid] = bd; best_start[tid] = bj;
    __syncthreads();
    for (int h = THREADS / 2; h > 0; h >>= 1) {
        if (tid < h && better(best_score[tid + h], best_duration[tid + h], best_start[tid + h],
                              best_score[tid], best_duration[tid], best_start[tid])) {
            best_score[tid] = best_score[tid + h];
            best_duration[tid] = best_duration[tid + h];
            best_start[tid] = best_start[tid + h];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_score[b] = best_score[0];
        out_duration[b] = best_duration[0];
        out_start[b] = best_start[0];
    }
}
