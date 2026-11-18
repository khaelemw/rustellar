// The cells on the GPU (cells.rs, Cells::new): one thread a cell, the
// template's sums with its centre at that cell's time, over the points
// within half a duration of it, in time order.
//
// Compiled by NVRTC with --fmad=false, and each fused multiply-add the CPU
// makes made here with fma(), so every cell is the CPU's to the bit.

#ifndef THREADS
#define THREADS 256
#endif

// The first of the n increasing times above x (strictly), or at or above x.
__device__ __forceinline__ long long first_after(const double* time, long long n,
    double x, bool strictly) {
    long long lo = 0, hi = n;
    while (lo < hi) {
        const long long mid = lo + (hi - lo) / 2;
        const bool before = strictly ? time[mid] <= x : time[mid] < x;
        if (before) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// The template at u samples from x = -1/2 (fast.rs, Table::at_sample).
__device__ __forceinline__ double at_sample(const double* table, double intervals, double u) {
    if (!(u > 0.0 && u < intervals)) return 0.0;
    const long long j = (long long)u;
    const double f = u - (double)j;
    return fma(f, table[j + 1] - table[j], table[j]);
}

extern "C" __global__ void cells(
    const double* __restrict__ time, const double* __restrict__ residual,
    const double* __restrict__ weight, long long n,
    const double* __restrict__ table, double intervals,
    double duration, double start, double step, long long count,
    double* __restrict__ rs, double* __restrict__ ss)
{
    const long long m = (long long)blockIdx.x * THREADS + threadIdx.x;
    if (m >= count) return;
    const double half = 0.5 * duration;
    const double centre = fma(step, (double)m, start);
    const long long lo = first_after(time, n, centre - half, true);
    const long long hi = first_after(time, n, centre + half, false);
    const double to_sample = intervals / duration;
    // Table samples from x = -1/2: (t - centre) / D + 1/2.
    const double shift = fma(0.5, intervals, -(centre * to_sample));
    double r = 0.0, s2 = 0.0;
    for (long long i = lo; i < hi; i++) {
        const double s = at_sample(table, intervals, fma(time[i], to_sample, shift));
        r += weight[i] * residual[i] * s;
        s2 += weight[i] * s * s;
    }
    rs[m] = r;
    ss[m] = s2;
}
