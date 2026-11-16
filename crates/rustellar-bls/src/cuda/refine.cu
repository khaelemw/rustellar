// Refining the peaks on the GPU: many folds at once (one a row of blocks,
// blockIdx.z or y), each made as the CPU makes it (exact.rs, Folded::new),
// and searched for the best runs as the CPU searches them (Folded::best_run),
// to the bit.
//
// - A point's phase is the CPU's offset_after_start_short, by the same
//   operations in the same order (--fmad=false).
// - The points are sorted by phase, then by index, as the CPU sorts them: a
//   bitonic sort of (key, index), the key a phase's bits made to order as
//   f64::total_cmp does.
// - The running totals are the CPU's compensated sums (sums.rs, Sum::add),
//   added in the same order: one thread a sum, through every point in turn.
// - Every run a box of the search's durations holds is scored as the CPU
//   scores it (totals.rs, Orbit::statistic). The CPU's two pointers become
//   binary searches, which find the same ends as the phases do not
//   decrease. Where a first point has many ends, blocks of ends whose bound
//   (exact.rs, Bounds) is below the best score any thread has found so far,
//   or not above the search's beat, are skipped: no run in them can be the
//   best or tie it.
// - The first of the highest runs of each search, and what its box needs of
//   the fold, go back to the CPU, which builds the box (exact.rs, Run).

#define RUN_BLOCK 32
#define MANY_ENDS 64
#define LOCAL 2048
#define SEARCH_VALUES 3
#define RUN_VALUES 27

typedef unsigned long long u64;

// ---------------------------------------------------------------- folding

// A phase's bits, made to order as f64::total_cmp orders phases.
__device__ __forceinline__ u64 key_of(double x) {
    const u64 b = (u64)__double_as_longlong(x);
    return (b >> 63) ? ~b : (b | (1ULL << 63));
}

__device__ __forceinline__ double phase_of(u64 key) {
    const u64 b = (key >> 63) ? (key & ~(1ULL << 63)) : ~key;
    return __longlong_as_double((long long)b);
}

// Each point's key and index; past the n points, keys above every phase.
// `periods` holds p, 1 / p and p's halves for each fold.
extern "C" __global__ void fold_keys(const double* s, const double* e, int n,
    const double* periods, int width, u64* keys, unsigned int* index)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= width) return;
    const size_t at = (size_t)blockIdx.y * width + k;
    if (k >= n) { keys[at] = ~0ULL; index[at] = 0xFFFFFFFFu; return; }
    const double* c = periods + 4 * blockIdx.y;
    const double p = c[0], inverse = c[1], p_hi = c[2], p_lo = c[3];
    const double TWO_52 = 4503599627370496.0;
    const double MARGIN = 8.0 * 2.220446049250313e-16;
    const double sk = s[k];
    // box_fit.rs, AtPeriod::offset_after_start_short and checked.
    const double q = sk * inverse;
    double whole = (q + TWO_52) - TWO_52;
    if (!(0.5 - fabs(q - whole) > MARGIN * fabs(q))) {
        const double r = sk / p;
        whole = (r + TWO_52) - TWO_52;
    }
    const double offset = ((sk - whole * p_hi) - whole * p_lo) + e[k];
    keys[at] = key_of(offset);
    index[at] = (unsigned int)k;
}

__device__ __forceinline__ bool before_pair(u64 ka, unsigned int ia, u64 kb, unsigned int ib) {
    return ka < kb || (ka == kb && ia < ib);
}

// One step of the bitonic sort, between elements `j` apart, in sequences
// of `k`, for every fold.
extern "C" __global__ void bitonic_step(u64* keys, unsigned int* index, int width, int k, int j)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int partner = i ^ j;
    if (partner <= i) return;
    const size_t base = (size_t)blockIdx.y * width;
    const u64 ka = keys[base + i], kb = keys[base + partner];
    const unsigned int ia = index[base + i], ib = index[base + partner];
    const bool up = (i & k) == 0;
    if (up ? before_pair(kb, ib, ka, ia) : before_pair(ka, ia, kb, ib)) {
        keys[base + i] = kb; keys[base + partner] = ka;
        index[base + i] = ib; index[base + partner] = ia;
    }
}

// The steps between elements less than LOCAL apart, in shared memory: for
// every sequence length up to LOCAL if `k` is 0, else for sequences of `k`.
extern "C" __global__ void bitonic_local(u64* keys, unsigned int* index, int width, int k)
{
    __shared__ u64 sk[LOCAL];
    __shared__ unsigned int si[LOCAL];
    const int chunk = min(width, LOCAL);
    const int first = blockIdx.x * chunk;
    const size_t base = (size_t)blockIdx.y * width + first;
    for (int t = threadIdx.x; t < chunk; t += blockDim.x) {
        sk[t] = keys[base + t];
        si[t] = index[base + t];
    }
    __syncthreads();
    const int from = k == 0 ? 2 : k;
    const int to = k == 0 ? chunk : k;
    for (int kk = from; kk <= to; kk *= 2) {
        for (int j = min(kk, chunk) / 2; j > 0; j /= 2) {
            for (int t = threadIdx.x; t < chunk; t += blockDim.x) {
                const int partner = t ^ j;
                if (partner > t) {
                    const bool up = ((first + t) & kk) == 0;
                    const u64 ka = sk[t], kb = sk[partner];
                    const unsigned int ia = si[t], ib = si[partner];
                    if (up ? before_pair(kb, ib, ka, ia) : before_pair(ka, ia, kb, ib)) {
                        sk[t] = kb; sk[partner] = ka;
                        si[t] = ib; si[partner] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int t = threadIdx.x; t < chunk; t += blockDim.x) {
        keys[base + t] = sk[t];
        index[base + t] = si[t];
    }
}

// The phases in order, then again one orbit on.
extern "C" __global__ void fold_phases(const u64* keys, int n, int width,
    const double* periods, double* phase)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const double x = phase_of(keys[(size_t)blockIdx.y * width + r]);
    double* out = phase + (size_t)blockIdx.y * 2 * n;
    out[r] = x;
    out[n + r] = x + periods[4 * blockIdx.y];
}

// The weights and the centred fluxes in phase order, for each fold.
extern "C" __global__ void fold_gather(const unsigned int* index, const double* weights,
    const double* centred, int n, int width, double* sorted)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const unsigned int k = index[(size_t)blockIdx.y * width + r];
    double* out = sorted + (size_t)blockIdx.y * 2 * n;
    out[r] = weights[k];
    out[n + r] = centred[k];
}

#define AHEAD 16

// The running totals are the CPU's compensated sums (sums.rs, Sum::add):
// with t the plain sum before a term x, t' = t + x, v = t' - t, and the
// compensation c' = c + e, e = (t - (t' - v)) + (x - v). Only t and c carry
// from term to term, so the plain sums go first, one addition a term, then
// every e at once, then the compensations, one addition a term: the same
// operations on the same numbers. Each total is four doubles: the weights'
// sum and compensation, then the fluxes'; the compensations' places hold
// the e until they are added up.

// A sum through `count` values, each one a stride apart (thread 0 of each
// block the weights, thread 1 the fluxes): out[k] the sum of the first k,
// the next AHEAD values read while these are added.
__device__ __forceinline__ void chain(const double* values, int stride, int count, int n,
    double* out)
{
    double sum = 0.0;
    out[0] = 0.0;
    double next[AHEAD];
    #pragma unroll
    for (int a = 0; a < AHEAD; a++) {
        const int r = a < count ? a : 0;
        next[a] = values[(size_t)stride * (r < n ? r : r - n)];
    }
    for (int r0 = 0; r0 < count; r0 += AHEAD) {
        double x[AHEAD];
        #pragma unroll
        for (int a = 0; a < AHEAD; a++) {
            x[a] = next[a];
            int r = r0 + AHEAD + a;
            r = r < count ? r : 0;
            next[a] = values[(size_t)stride * (r < n ? r : r - n)];
        }
        #pragma unroll
        for (int a = 0; a < AHEAD; a++) {
            const int r = r0 + a;
            if (r >= count) break;
            sum = sum + x[a];
            out[(size_t)4 * (r + 1)] = sum;
        }
    }
}

// The plain sums of the points in phase order, twice round.
extern "C" __global__ void fold_sums(const double* sorted, int n, double* totals)
{
    const int which = threadIdx.x;
    if (which > 1) return;
    chain(sorted + (size_t)blockIdx.x * 2 * n + (size_t)which * n, 1, 2 * n, n,
        totals + (size_t)blockIdx.x * 4 * (2 * n + 1) + 2 * which);
}

// Each term's e, in its compensation's place.
extern "C" __global__ void fold_errors(const double* sorted, int n, double* totals)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= 2 * n) return;
    const double* values = sorted + (size_t)blockIdx.y * 2 * n;
    double* out = totals + (size_t)blockIdx.y * 4 * (2 * n + 1);
    for (int which = 0; which < 2; which++) {
        const double x = values[(size_t)which * n + (r < n ? r : r - n)];
        const double t = out[(size_t)4 * r + 2 * which];
        const double t2 = out[(size_t)4 * (r + 1) + 2 * which];
        const double v = t2 - t;
        out[(size_t)4 * (r + 1) + 2 * which + 1] = (t - (t2 - v)) + (x - v);
    }
}

// The compensations, adding up the e in place.
extern "C" __global__ void fold_compensations(int n, double* totals)
{
    const int which = threadIdx.x;
    if (which > 1) return;
    double* c = totals + (size_t)blockIdx.x * 4 * (2 * n + 1) + 2 * which + 1;
    // The e are read 4 doubles apart, from the second total on; chain()
    // writes each sum back over its own e, which it has read by then.
    chain(c + 4, 4, 2 * n, 2 * n + 1, c);
}

// ---------------------------------------------------------------- searching

// A compensated running total's value (sums.rs, Sum::value).
__device__ __forceinline__ double total_value(double sum, double compensation) {
    return isfinite(sum) ? sum + compensation : sum;
}

// The sum of the terms between two compensated running totals (sums.rs).
__device__ __forceinline__ double between(const double* earlier, const double* later) {
    return (later[0] - earlier[0]) + (later[1] - earlier[1]);
}

// Each fold's orbit sums (totals.rs, Orbit::new): W, S, 1 / W and S / W,
// and whether its phases never decrease, the point before the first
// included.
extern "C" __global__ void fold_orbit(const double* totals, const double* phase, int n,
    const double* periods, int folds, double* orbit, int* ordered)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= folds) return;
    const double* t = totals + (size_t)f * 4 * (2 * n + 1);
    const double* tn = t + 4 * n;
    const double w = between(t, tn);
    const double s = between(t + 2, tn + 2);
    orbit[4 * f] = w;
    orbit[4 * f + 1] = s;
    orbit[4 * f + 2] = 1.0 / w;
    orbit[4 * f + 3] = s / w;
    const double* ph = phase + (size_t)f * 2 * n;
    ordered[f] = ph[n - 1] - periods[4 * f] <= ph[0] && ph[n - 1] <= ph[n];
}

// The totals' weights and g = flux - (S / W) weight at the first 2n, and the
// largest |flux| of each fold, as bits (a positive double's bits order as it
// does).
extern "C" __global__ void run_values(const double* totals, int n, const double* orbit,
    double* weight, double* g, u64* largest)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= 2 * n) return;
    const int f = blockIdx.y;
    const double* t = totals + (size_t)f * 4 * (2 * n + 1) + 4 * k;
    const double w = total_value(t[0], t[1]);
    const double flux = total_value(t[2], t[3]);
    const size_t at = (size_t)f * 2 * n + k;
    weight[at] = w;
    g[at] = flux - orbit[4 * f + 3] * w;
    atomicMax(largest + f, (u64)__double_as_longlong(fabs(flux)));
}

// The least of each RUN_BLOCK values, `count` a fold.
extern "C" __global__ void run_mins(const double* values, int count, int out_count, double* mins)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= out_count) return;
    const int from = b * RUN_BLOCK;
    const int to = min(count, from + RUN_BLOCK);
    const double* v = values + (size_t)blockIdx.y * count;
    double least = 1.0 / 0.0;
    for (int k = from; k < to; k++) least = fmin(least, v[k]);
    mins[(size_t)blockIdx.y * out_count + b] = least;
}

struct Bound {
    const double* g;
    const double* weight;
    double margin_u, margin_w, whole;
    int likelihood;

    // An upper bound on the scores of the runs from i to ends low..=high,
    // `least` the least g there (exact.rs, Bounds::bound).
    __device__ double at(int i, double least, int low, int high) const {
        const double EPS = 2.220446049250313e-16;
        const double fall = g[i] - least + margin_u;
        if (fall <= 0.0) return 0.0;
        const double w = whole;
        const double shortest = (weight[low] - weight[i]) - margin_w;
        const double longest = (weight[high] - weight[i]) + margin_w;
        if (shortest <= 0.0 || longest >= w) return 1.0 / 0.0;
        const double ds = shortest * ((w - shortest) / w);
        const double dl = longest * ((w - longest) / w);
        const double least_d = fmin(ds, dl) * (1.0 - 64.0 * EPS);
        double bound = fall * (fall / least_d);
        if (likelihood) bound *= 0.5 * w / (w - longest);
        return bound * (1.0 + 64.0 * EPS);
    }
};

// Whether a beats b: a higher score, or the same and an earlier run.
__device__ __forceinline__ bool beats(double sa, int ia, int ja, double sb, int ib, int jb) {
    if (jb == 0) return ja != 0;
    if (ja == 0) return false;
    return sa > sb || (sa == sb && (ia < ib || (ia == ib && ja < jb)));
}

// Each block's first of the highest runs of each search of each fold
// (blockIdx.y the search, blockIdx.z the fold).
extern "C" __global__ void runs(
    const double* phase, const double* totals, int n, const double* periods,
    const double* searches, // SEARCH_VALUES a search: shortest, longest, beat
    const int* counts,      // a fold's searches
    int most_searches,
    const double* orbit, const int* ordered, int likelihood,
    const double* g, const double* weight, const double* mins0, const double* mins1,
    int count0, int count1, const u64* largest,
    u64* highest,           // a search's best score so far, as bits
    double* out_score, int* out_first, int* out_end, // 0 where none beats the beat
    int* bad)               // set for a fold where a score is not finite
{
    const int f = blockIdx.z, search = blockIdx.y;
    if (search >= counts[f] || !ordered[f]) return;
    const int at = f * most_searches + search;
    const double* sv = searches + (size_t)SEARCH_VALUES * at;
    const double shortest = sv[0], longest = sv[1], beat = sv[2];
    const double period = periods[4 * f];
    const double whole_weight = orbit[4 * f], whole_flux = orbit[4 * f + 1];
    const double inverse_weight = orbit[4 * f + 2], ratio = orbit[4 * f + 3];
    const double half = 0.5 * whole_weight;
    const double* ph = phase + (size_t)f * 2 * n;
    const double* tot = totals + (size_t)f * 4 * (2 * n + 1);
    const double* gf = g + (size_t)f * 2 * n;
    const double* wf = weight + (size_t)f * 2 * n;
    const double* m0 = mins0 + (size_t)f * count0;
    const double* m1 = mins1 + (size_t)f * count1;
    // The margins, as exact.rs, Bounds::new.
    const double EPS = 2.220446049250313e-16;
    const double largest_flux = __longlong_as_double((long long)largest[f]);
    const double largest_weight = wf[2 * n - 1];
    const Bound bound = {gf, wf,
        64.0 * EPS * (largest_flux + fabs(ratio) * largest_weight + fabs(whole_flux)),
        64.0 * EPS * largest_weight, whole_weight, likelihood};
    volatile u64* best_so_far = highest + at;

    double best = beat;
    int best_i = 0, best_j = 0;
    // Each thread a stretch of first points. The first end whose neighbours
    // are `shortest` apart, and one past the last whose run spans under
    // `longest`: by binary search at the stretch's start, then by the CPU's
    // two pointers.
    const int threads = gridDim.x * blockDim.x;
    const int stretch = (n + threads - 1) / threads;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * stretch;
    const int i1 = min(n, i0 + stretch);
    int first_pointer = 0, last_pointer = 0;
    if (i0 < i1) {
        const double before = i0 == 0 ? ph[n - 1] - period : ph[i0 - 1];
        int lo = i0 + 1, hi = i0 + n;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (ph[mid] - before < shortest) lo = mid + 1; else hi = mid;
        }
        first_pointer = lo;
        lo = i0 + 1; hi = i0 + n;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (ph[mid] - ph[i0] < longest) lo = mid + 1; else hi = mid;
        }
        last_pointer = lo;
    }
    for (int i = i0; i < i1; i++) {
        const double before = i == 0 ? ph[n - 1] - period : ph[i - 1];
        const double here = ph[i];
        first_pointer = max(first_pointer, i + 1);
        while (first_pointer < i + n && ph[first_pointer] - before < shortest) first_pointer++;
        last_pointer = max(last_pointer, i + 1);
        while (last_pointer < i + n && ph[last_pointer] - here < longest) last_pointer++;
        if (before >= here) continue;
        const int first_end = first_pointer;
        const int last_end = min(last_pointer, i + n - 1);
        const double* first = tot + 4 * i;
        const double* wrap = tot + 4 * (i + n);
        const bool many = last_end - first_end + 1 >= MANY_ENDS;
        // Blocks of RUN_BLOCK^2 ends, then of RUN_BLOCK, then the runs.
        const int size1 = RUN_BLOCK * RUN_BLOCK;
        for (int k1 = first_end / size1; k1 <= last_end / size1; k1++) {
            const int low1 = max(first_end, k1 * size1);
            const int high1 = min(last_end, k1 * size1 + size1 - 1);
            if (low1 > high1) continue;
            if (many) {
                const double b = bound.at(i, m1[k1], low1, high1);
                if (b <= beat || b < __longlong_as_double((long long)*best_so_far)) continue;
            }
            for (int k0 = low1 / RUN_BLOCK; k0 <= high1 / RUN_BLOCK; k0++) {
                const int low0 = max(low1, k0 * RUN_BLOCK);
                const int high0 = min(high1, k0 * RUN_BLOCK + RUN_BLOCK - 1);
                if (many) {
                    const double b = bound.at(i, m0[k0], low0, high0);
                    if (b <= beat || b < __longlong_as_double((long long)*best_so_far)) continue;
                }
                for (int j = low0; j <= high0; j++) {
                    if (ph[j - 1] >= ph[j]) continue;
                    const double* end = tot + 4 * j;
                    const double weight_in = between(first, end);
                    const double weight_out = weight_in <= half
                        ? whole_weight - weight_in
                        : between(end, wrap);
                    if (weight_in == 0.0 || weight_out == 0.0) continue;
                    const double u = between(first + 2, end + 2)
                        - whole_flux * (weight_in * inverse_weight);
                    // No dip: a score of zero, which never beats the beat.
                    // (No score of the fold is infinite: the CPU checks
                    // that before taking the fold's runs.)
                    if (u >= 0.0) continue;
                    const double share = weight_out * inverse_weight;
                    // A score in singles, to within 2^-20 of itself where
                    // every number in it is far from the singles' limits:
                    // a box that falls short of the best so far by more
                    // cannot be the best, nor tie it.
                    const double floor = fmax(best, __longlong_as_double((long long)*best_so_far));
                    if (floor > 0.0) {
                        const float uf = (float)u, wi = (float)weight_in, ws = (float)share;
                        const float q = uf / (wi * ws);
                        float sf = uf * q;
                        if (likelihood) sf *= (float)half / (float)weight_out;
                        const bool fair = fabsf(uf) > 1e-15f && fabsf(uf) < 1e15f
                            && wi > 1e-15f && wi < 1e15f && ws > 1e-15f && ws < 1e15f
                            && fabsf(q) > 1e-30f && fabsf(q) < 1e30f
                            && sf > 1e-30f && sf < 1e30f;
                        if (fair && (double)sf * (1.0 + 1.0 / 1048576.0) < floor) continue;
                    }
                    const double delta_chi2 = u * (u / (weight_in * share));
                    const double score = likelihood ? delta_chi2 * (half / weight_out) : delta_chi2;
                    if (!isfinite(score)) { bad[f] = 1; continue; }
                    if (u < 0.0 && score > best) {
                        best = score; best_i = i; best_j = j;
                        atomicMax(highest + at, (u64)__double_as_longlong(score));
                    }
                }
            }
        }
    }

    // The block's first of the highest.
    __shared__ double scores[THREADS];
    __shared__ int firsts[THREADS], ends[THREADS];
    const int t = threadIdx.x;
    scores[t] = best; firsts[t] = best_i; ends[t] = best_j;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (t < stride && beats(scores[t + stride], firsts[t + stride], ends[t + stride],
                                scores[t], firsts[t], ends[t])) {
            scores[t] = scores[t + stride];
            firsts[t] = firsts[t + stride];
            ends[t] = ends[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        const size_t out = (size_t)at * gridDim.x + blockIdx.x;
        out_score[out] = scores[0];
        out_first[out] = firsts[0];
        out_end[out] = ends[0];
    }
}

// Each search's first of the highest over its blocks, and what its box
// needs: [found, i, j, 4 phases, 5 totals of 4] (exact.rs, Run).
extern "C" __global__ void run_finish(const double* phase, const double* totals, int n,
    const double* periods, const int* counts, int most_searches, int folds, int blocks,
    const double* scores, const int* firsts, const int* ends, double* out)
{
    const int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= folds * most_searches) return;
    const int f = at / most_searches;
    double* o = out + (size_t)RUN_VALUES * at;
    o[0] = 0.0;
    if (at % most_searches >= counts[f]) return;
    double best = 0.0;
    int bi = 0, bj = 0;
    for (int b = 0; b < blocks; b++) {
        const size_t k = (size_t)at * blocks + b;
        if (beats(scores[k], firsts[k], ends[k], best, bi, bj)) {
            best = scores[k]; bi = firsts[k]; bj = ends[k];
        }
    }
    if (bj == 0) return;
    const double* ph = phase + (size_t)f * 2 * n;
    const double* tot = totals + (size_t)f * 4 * (2 * n + 1);
    o[0] = 1.0; o[1] = bi; o[2] = bj;
    o[3] = bi == 0 ? ph[n - 1] - periods[4 * f] : ph[bi - 1];
    o[4] = ph[bi]; o[5] = ph[bj - 1]; o[6] = ph[bj];
    const int which[5] = {0, n, bi, bj, bi + n};
    for (int w = 0; w < 5; w++)
        for (int c = 0; c < 4; c++) o[7 + 4 * w + c] = tot[4 * which[w] + c];
}
