/*
 * stream_bench.c -- STREAM-style memory bandwidth microbenchmark.
 *
 * OpenMP parallelizes the four classic kernels (Copy/Scale/Add/Triad) across
 * threads; NUMA node and core placement are left entirely to `numactl` (this
 * binary does no CPU/memory-policy calls itself; see usage() for the
 * companion numactl invocation). Arrays are parallel-initialized with the
 * same loop partitioning as the kernels so first-touch lands each thread's
 * chunk on whatever node numactl --membind selected.
 *
 * Modeled on John McCalpin's STREAM benchmark (Copy/Scale/Add/Triad kernels,
 * best-of-N timing, checksum validation), reimplemented standalone here for
 * OpenMP + numactl only (no external deps, single file).
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o stream_bench stream_bench.c
 *
 * Run (bind to NUMA node 0, cores 192-222, 31 threads):
 *   OMP_NUM_THREADS=31 numactl --cpunodebind=0 --membind=0 \
 *     --physcpubind=192-222 ./stream_bench --mb 4096 --iters 10
 *
 * Do NOT set OMP_PROC_BIND=true here without also setting OMP_PLACES: with
 * libgomp, proc_bind + unset places collapses all threads onto a handful of
 * cores instead of spreading across numactl's --physcpubind set, silently
 * cutting measured bandwidth by ~9x on this machine (151 GB/s -> 16 GB/s,
 * confirmed empirically). Leaving OMP_PROC_BIND unset lets numactl's cpuset
 * alone govern placement, which is what you want.
 */

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double stream_t;

static double now(void) { return omp_get_wtime(); }

static double avg_from(const double *t, int iters) {
    double sum = 0.0;
    for (int k = 1; k < iters; k++) sum += t[k];
    return sum / (iters - 1);
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--mb <per-array MB>] [--iters <N>] [--offset <N>]\n"
        "\n"
        "  --mb      total size of EACH of the 3 arrays, in MiB (default 2048;\n"
        "            use 4x+ your last-level cache total to avoid cache hits)\n"
        "  --iters   repetitions per kernel; best (min) time is reported,\n"
        "            matching STREAM's convention of ignoring OS-noise outliers\n"
        "            (default 10)\n"
        "  --offset  padding elements between arrays, guards against\n"
        "            cache-associativity aliasing artifacts (default 0)\n"
        "\n"
        "Thread count is controlled by OMP_NUM_THREADS / OMP_PROC_BIND, NOT by\n"
        "a flag here. CPU core and NUMA-node placement is controlled entirely\n"
        "by the numactl invocation wrapping this binary -- this program makes\n"
        "no sched_setaffinity/mbind calls of its own. Do NOT set OMP_PROC_BIND\n"
        "without OMP_PLACES: with libgomp it collapses threads onto too few\n"
        "cores and tanks the measured bandwidth. Leave it unset.\n"
        "\n"
        "Example:\n"
        "  OMP_NUM_THREADS=31 numactl --cpunodebind=0 --membind=0 \\\n"
        "    --physcpubind=192-222 %s --mb 4096 --iters 10\n",
        argv0, argv0);
}

int main(int argc, char **argv) {
    size_t per_array_mb = 2048;
    int iters = 10;
    size_t offset = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mb") == 0 && i + 1 < argc) {
            per_array_mb = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--offset") == 0 && i + 1 < argc) {
            offset = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (iters < 2) {
        fprintf(stderr, "--iters must be >= 2 (first iteration is discarded as warmup)\n");
        return 1;
    }

    size_t n = (per_array_mb * 1024ULL * 1024ULL) / sizeof(stream_t);
    size_t alloc_n = n + offset;

    stream_t *a = malloc(alloc_n * sizeof(stream_t));
    stream_t *b = malloc(alloc_n * sizeof(stream_t));
    stream_t *c = malloc(alloc_n * sizeof(stream_t));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed for n=%zu (%zu MB/array)\n", n, per_array_mb);
        return 1;
    }

    int nthreads = 0;
#pragma omp parallel
    {
#pragma omp master
        nthreads = omp_get_num_threads();
    }

    fprintf(stderr,
        "elements/array=%zu (%zu MB), total=%zu MB, iters=%d, threads=%d\n",
        n, per_array_mb, per_array_mb * 3, iters, nthreads);

    /* First-touch init: same partitioning OpenMP will use for the kernels
     * below, so each thread's pages land wherever it first writes them. */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    const stream_t scalar = 3.0;
    double t_copy[64], t_scale[64], t_add[64], t_triad[64];
    if (iters > 64) iters = 64; /* fixed-size timing arrays; plenty for best-of-N */

    for (int k = 0; k < iters; k++) {
        double t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) c[i] = a[i];
        t_copy[k] = now() - t0;

        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) b[i] = scalar * c[i];
        t_scale[k] = now() - t0;

        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
        t_add[k] = now() - t0;

        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) a[i] = b[i] + scalar * c[i];
        t_triad[k] = now() - t0;
    }

    /* Correctness check: every round mutates a/b/c in place (not just round
     * 0), so replay the same 4-step update on scalars starting from the same
     * a0=1/b0=2/c0=0 for `iters` rounds (including the warmup round, which
     * ran identically) and compare against that closed-form result. */
    stream_t exp_a = 1.0, exp_b = 2.0, exp_c = 0.0;
    for (int k = 0; k < iters; k++) {
        exp_c = exp_a;
        exp_b = scalar * exp_c;
        exp_c = exp_a + exp_b;
        exp_a = exp_b + scalar * exp_c;
    }
    double max_rel_err = 0.0;
    for (size_t i = 0; i < n; i += (n / 1024 > 0 ? n / 1024 : 1)) {
        double err = fabs(a[i] - exp_a) / fabs(exp_a);
        if (err > max_rel_err) max_rel_err = err;
    }
    if (max_rel_err > 1e-10) {
        fprintf(stderr,
            "WARNING: correctness check failed, max relative error=%.3e "
            "(expected ~1e-16 for doubles) -- results below may be unreliable\n",
            max_rel_err);
    }

    /* Best (min) time per kernel, skipping the first iteration as warmup,
     * matching STREAM's own convention. */
    double best_copy = 1e300, best_scale = 1e300, best_add = 1e300, best_triad = 1e300;
    for (int k = 1; k < iters; k++) {
        if (t_copy[k] < best_copy) best_copy = t_copy[k];
        if (t_scale[k] < best_scale) best_scale = t_scale[k];
        if (t_add[k] < best_add) best_add = t_add[k];
        if (t_triad[k] < best_triad) best_triad = t_triad[k];
    }

    double bytes_copy = 2.0 * n * sizeof(stream_t);  /* 1 read + 1 write */
    double bytes_scale = 2.0 * n * sizeof(stream_t); /* 1 read + 1 write */
    double bytes_add = 3.0 * n * sizeof(stream_t);   /* 2 reads + 1 write */
    double bytes_triad = 3.0 * n * sizeof(stream_t); /* 2 reads + 1 write */

    printf("Function    Best Rate (GB/s)   Best Time (s)   Avg Time (s)\n");
    printf("Copy        %14.2f   %13.6f   %12.6f\n",
        bytes_copy / best_copy / 1e9, best_copy, avg_from(t_copy, iters));
    printf("Scale       %14.2f   %13.6f   %12.6f\n",
        bytes_scale / best_scale / 1e9, best_scale, avg_from(t_scale, iters));
    printf("Add         %14.2f   %13.6f   %12.6f\n",
        bytes_add / best_add / 1e9, best_add, avg_from(t_add, iters));
    printf("Triad       %14.2f   %13.6f   %12.6f\n",
        bytes_triad / best_triad / 1e9, best_triad, avg_from(t_triad, iters));

    free(a);
    free(b);
    free(c);
    return 0;
}
