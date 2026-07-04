#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <dlfcn.h>
#include <unistd.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifndef __APPLE__
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
#endif

typedef void (*cblas_daxpy_fn)(int, double, const double *, int, double *, int);
typedef double (*cblas_dnrm2_fn)(int, const double *, int);
typedef double (*cblas_dasum_fn)(int, const double *, int);
typedef double (*cblas_ddot_fn)(int, const double *, int, const double *, int);
typedef void (*cblas_dscal_fn)(int, double, double *, int);
typedef void (*cblas_dgemv_fn)(int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
typedef void (*cblas_dtrmv_fn)(int, int, int, int, int, const double *, int, double *, int);
typedef void (*cblas_dtrsv_fn)(int, int, int, int, int, const double *, int, double *, int);
typedef void (*cblas_dsymv_fn)(int, int, int, double, const double *, int, const double *, int, double, double *, int);
typedef void (*cblas_dsyr_fn)(int, int, int, double, const double *, int, double *, int);
typedef void (*cblas_dsyr2_fn)(int, int, int, double, const double *, int, const double *, int, double *, int);
typedef void (*cblas_dgemm_fn)(int, int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
typedef void (*cblas_dsyrk_fn)(int, int, int, int, int, double, const double *, int, double, double *, int);
typedef void (*cblas_dsyr2k_fn)(int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
typedef void (*cblas_dsymm_fn)(int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
typedef void (*cblas_dtrmm_fn)(int, int, int, int, int, int, int, double, const double *, int, double *, int);
typedef void (*cblas_dtrsm_fn)(int, int, int, int, int, int, int, double, const double *, int, double *, int);
typedef void (*cblas_dtpmv_fn)(int, int, int, int, int, const double *, double *, int);
typedef void (*cblas_dtpsv_fn)(int, int, int, int, int, const double *, double *, int);
typedef void (*cblas_dtbmv_fn)(int, int, int, int, int, int, const double *, int, double *, int);
typedef void (*cblas_dtbsv_fn)(int, int, int, int, int, int, const double *, int, double *, int);
typedef void (*cblas_dspmv_fn)(int, int, int, double, const double *, const double *, int, double, double *, int);
typedef void (*cblas_dspr_fn)(int, int, int, double, const double *, int, double *);
typedef void (*cblas_dspr2_fn)(int, int, int, double, const double *, int, const double *, int, double *);
typedef void (*cblas_drotm_fn)(int, double *, int, double *, int, const double *);
typedef void (*cblas_drotmg_fn)(double *, double *, double *, double, double *);
typedef void (*openblas_set_num_threads_fn)(int);

typedef struct {
    const char *name;
    void *handle;
    cblas_daxpy_fn daxpy;
    cblas_dnrm2_fn dnrm2;
    cblas_dasum_fn dasum;
    cblas_ddot_fn ddot;
    cblas_dscal_fn dscal;
    cblas_dgemv_fn dgemv;
    cblas_dtrmv_fn dtrmv;
    cblas_dtrsv_fn dtrsv;
    cblas_dsymv_fn dsymv;
    cblas_dsyr_fn dsyr;
    cblas_dsyr2_fn dsyr2;
    cblas_dgemm_fn dgemm;
    cblas_dsyrk_fn dsyrk;
    cblas_dsyr2k_fn dsyr2k;
    cblas_dsymm_fn dsymm;
    cblas_dtrmm_fn dtrmm;
    cblas_dtrsm_fn dtrsm;
    cblas_dtpmv_fn dtpmv;
    cblas_dtpsv_fn dtpsv;
    cblas_dtbmv_fn dtbmv;
    cblas_dtbsv_fn dtbsv;
    cblas_dspmv_fn dspmv;
    cblas_dspr_fn dspr;
    cblas_dspr2_fn dspr2;
    cblas_drotm_fn drotm;
    cblas_drotmg_fn drotmg;
    openblas_set_num_threads_fn set_num_threads;
} BlasLib;

typedef struct {
    FILE *out;
    int first;
} JsonWriter;

static volatile double g_sink = 0.0;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void set_max_thread_env(int threads) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", threads);
    setenv("OPENBLAS_NUM_THREADS", buf, 1);
    setenv("GOTO_NUM_THREADS", buf, 1);
    setenv("OMP_NUM_THREADS", buf, 1);
    setenv("VECLIB_MAXIMUM_THREADS", buf, 1);
    setenv("VECLIB_NUM_THREADS", buf, 1);
}

static void *checked_dlsym_optional(void *handle, const char *symbol) {
    dlerror();
    void *p = dlsym(handle, symbol);
    (void)dlerror();
    return p;
}

static double urand_signed(void) {
    return 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
}

static double *alloc_aligned(size_t count) {
    void *p = NULL;
    if (posix_memalign(&p, 64, count * sizeof(double)) != 0) {
        return NULL;
    }
    return (double *)p;
}

static void fill_random(double *buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        buf[i] = urand_signed();
    }
}

static void copy_buf(double *dst, const double *src, size_t n) {
    memcpy(dst, src, n * sizeof(double));
}

static void make_general_matrix(double *a, int n) {
    fill_random(a, (size_t)n * (size_t)n);
}

static void make_symmetric_matrix(double *a, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            double v = urand_signed();
            if (i == j) {
                v += (double)n;
            }
            a[i + (size_t)j * (size_t)n] = v;
            a[j + (size_t)i * (size_t)n] = v;
        }
    }
}

static void make_upper_triangular_matrix(double *a, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i > j) {
                a[i + (size_t)j * (size_t)n] = 0.0;
            } else {
                double v = urand_signed();
                if (i == j) {
                    v += (double)n;
                }
                a[i + (size_t)j * (size_t)n] = v;
            }
        }
    }
}

static void make_packed_upper_symmetric(double *ap, int n) {
    size_t k = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            double v = urand_signed();
            if (i == j) {
                v += (double)n;
            }
            ap[k++] = v;
        }
    }
}

static void make_packed_upper_triangular(double *ap, int n) {
    make_packed_upper_symmetric(ap, n);
}

static void make_upper_triangular_band(double *ab, int n, int k, int lda) {
    size_t total = (size_t)lda * (size_t)n;
    for (size_t idx = 0; idx < total; idx++) {
        ab[idx] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        int i0 = j - k;
        if (i0 < 0) {
            i0 = 0;
        }
        for (int i = i0; i <= j; i++) {
            int row = k + i - j;
            double v = urand_signed();
            if (i == j) {
                v += (double)n;
            }
            ab[row + (size_t)j * (size_t)lda] = v;
        }
    }
}

static void json_begin(FILE *out, int min_n, int max_n, double step, int iters, const char *openblas_path) {
    fprintf(out, "{\n");
    fprintf(out, "  \"metadata\": {\n");
    fprintf(out, "    \"min_n\": %d,\n", min_n);
    fprintf(out, "    \"max_n\": %d,\n", max_n);
    fprintf(out, "    \"step\": %.17g,\n", step);
    fprintf(out, "    \"iters\": %d,\n", iters);
    fprintf(out, "    \"openblas_path\": \"%s\"\n", openblas_path ? openblas_path : "");
    fprintf(out, "  },\n");
    fprintf(out, "  \"results\": [\n");
}

static void json_end(FILE *out) {
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");
}

static void write_result(JsonWriter *writer, const char *lib, const char *op, int n, int iters, double elapsed) {
    if (!writer->first) {
        fprintf(writer->out, ",\n");
    }
    writer->first = 0;
    fprintf(writer->out,
            "    {\"lib\":\"%s\",\"op\":\"%s\",\"n\":%d,\"iters\":%d,\"avg_seconds\":%.17g}",
            lib, op, n, iters, elapsed / (double)iters);
}

static int next_geometric_n(int current, double step) {
    double next_d = (double)current * step;
    int next = (int)floor(next_d + 1e-12);
    if (next <= current) {
        next = current + 1;
    }
    return next;
}

static void init_accelerate(BlasLib *lib) {
    memset(lib, 0, sizeof(*lib));
    lib->name = "accelerate";
#ifdef __APPLE__
    lib->daxpy = (cblas_daxpy_fn)cblas_daxpy;
    lib->dnrm2 = (cblas_dnrm2_fn)cblas_dnrm2;
    lib->dasum = (cblas_dasum_fn)cblas_dasum;
    lib->ddot = (cblas_ddot_fn)cblas_ddot;
    lib->dscal = (cblas_dscal_fn)cblas_dscal;
    lib->dgemv = (cblas_dgemv_fn)cblas_dgemv;
    lib->dtrmv = (cblas_dtrmv_fn)cblas_dtrmv;
    lib->dtrsv = (cblas_dtrsv_fn)cblas_dtrsv;
    lib->dsymv = (cblas_dsymv_fn)cblas_dsymv;
    lib->dsyr = (cblas_dsyr_fn)cblas_dsyr;
    lib->dsyr2 = (cblas_dsyr2_fn)cblas_dsyr2;
    lib->dgemm = (cblas_dgemm_fn)cblas_dgemm;
    lib->dsyrk = (cblas_dsyrk_fn)cblas_dsyrk;
    lib->dsyr2k = (cblas_dsyr2k_fn)cblas_dsyr2k;
    lib->dsymm = (cblas_dsymm_fn)cblas_dsymm;
    lib->dtrmm = (cblas_dtrmm_fn)cblas_dtrmm;
    lib->dtrsm = (cblas_dtrsm_fn)cblas_dtrsm;
    lib->dtpmv = (cblas_dtpmv_fn)cblas_dtpmv;
    lib->dtpsv = (cblas_dtpsv_fn)cblas_dtpsv;
    lib->dtbmv = (cblas_dtbmv_fn)cblas_dtbmv;
    lib->dtbsv = (cblas_dtbsv_fn)cblas_dtbsv;
    lib->dspmv = (cblas_dspmv_fn)cblas_dspmv;
    lib->dspr = (cblas_dspr_fn)cblas_dspr;
    lib->dspr2 = (cblas_dspr2_fn)cblas_dspr2;
    lib->drotm = (cblas_drotm_fn)cblas_drotm;
    lib->drotmg = (cblas_drotmg_fn)cblas_drotmg;
#endif
}

static int load_openblas(BlasLib *lib, const char *path) {
    memset(lib, 0, sizeof(*lib));
    lib->name = "openblas";
    lib->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib->handle) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror());
        return 0;
    }
    lib->daxpy = (cblas_daxpy_fn)checked_dlsym_optional(lib->handle, "cblas_daxpy");
    lib->dnrm2 = (cblas_dnrm2_fn)checked_dlsym_optional(lib->handle, "cblas_dnrm2");
    lib->dasum = (cblas_dasum_fn)checked_dlsym_optional(lib->handle, "cblas_dasum");
    lib->ddot = (cblas_ddot_fn)checked_dlsym_optional(lib->handle, "cblas_ddot");
    lib->dscal = (cblas_dscal_fn)checked_dlsym_optional(lib->handle, "cblas_dscal");
    lib->dgemv = (cblas_dgemv_fn)checked_dlsym_optional(lib->handle, "cblas_dgemv");
    lib->dtrmv = (cblas_dtrmv_fn)checked_dlsym_optional(lib->handle, "cblas_dtrmv");
    lib->dtrsv = (cblas_dtrsv_fn)checked_dlsym_optional(lib->handle, "cblas_dtrsv");
    lib->dsymv = (cblas_dsymv_fn)checked_dlsym_optional(lib->handle, "cblas_dsymv");
    lib->dsyr = (cblas_dsyr_fn)checked_dlsym_optional(lib->handle, "cblas_dsyr");
    lib->dsyr2 = (cblas_dsyr2_fn)checked_dlsym_optional(lib->handle, "cblas_dsyr2");
    lib->dgemm = (cblas_dgemm_fn)checked_dlsym_optional(lib->handle, "cblas_dgemm");
    lib->dsyrk = (cblas_dsyrk_fn)checked_dlsym_optional(lib->handle, "cblas_dsyrk");
    lib->dsyr2k = (cblas_dsyr2k_fn)checked_dlsym_optional(lib->handle, "cblas_dsyr2k");
    lib->dsymm = (cblas_dsymm_fn)checked_dlsym_optional(lib->handle, "cblas_dsymm");
    lib->dtrmm = (cblas_dtrmm_fn)checked_dlsym_optional(lib->handle, "cblas_dtrmm");
    lib->dtrsm = (cblas_dtrsm_fn)checked_dlsym_optional(lib->handle, "cblas_dtrsm");
    lib->dtpmv = (cblas_dtpmv_fn)checked_dlsym_optional(lib->handle, "cblas_dtpmv");
    lib->dtpsv = (cblas_dtpsv_fn)checked_dlsym_optional(lib->handle, "cblas_dtpsv");
    lib->dtbmv = (cblas_dtbmv_fn)checked_dlsym_optional(lib->handle, "cblas_dtbmv");
    lib->dtbsv = (cblas_dtbsv_fn)checked_dlsym_optional(lib->handle, "cblas_dtbsv");
    lib->dspmv = (cblas_dspmv_fn)checked_dlsym_optional(lib->handle, "cblas_dspmv");
    lib->dspr = (cblas_dspr_fn)checked_dlsym_optional(lib->handle, "cblas_dspr");
    lib->dspr2 = (cblas_dspr2_fn)checked_dlsym_optional(lib->handle, "cblas_dspr2");
    lib->drotm = (cblas_drotm_fn)checked_dlsym_optional(lib->handle, "cblas_drotm");
    lib->drotmg = (cblas_drotmg_fn)checked_dlsym_optional(lib->handle, "cblas_drotmg");
    lib->set_num_threads = (openblas_set_num_threads_fn)checked_dlsym_optional(lib->handle, "openblas_set_num_threads");
    if (!lib->set_num_threads) {
        lib->set_num_threads = (openblas_set_num_threads_fn)checked_dlsym_optional(lib->handle, "openblas_set_num_threads64_");
    }
    return 1;
}

static void unload_lib(BlasLib *lib) {
    if (lib->handle) {
        dlclose(lib->handle);
        lib->handle = NULL;
    }
}

static int pick_iters_level1(int n, int explicit_iters) {
    if (explicit_iters > 0) {
        return explicit_iters;
    }
    double target = 2e8;
    int iters = (int)(target / (double)n);
    if (iters < 256) {
        iters = 256;
    }
    return iters;
}

static int pick_iters_level2(int n, int explicit_iters) {
    if (explicit_iters > 0) {
        return explicit_iters;
    }
    double target = 3e7;
    int iters = (int)(target / ((double)n * (double)n));
    if (iters < 32) {
        iters = 32;
    }
    return iters;
}

static int pick_iters_level3(int n, int explicit_iters) {
    if (explicit_iters > 0) {
        return explicit_iters;
    }
    double target = 5e7;
    int iters = (int)(target / ((double)n * (double)n * (double)n));
    if (iters < 3) {
        iters = 3;
    }
    return iters;
}

static void bench_level1(BlasLib *lib, JsonWriter *writer, int n, int iters) {
    double *x = alloc_aligned((size_t)n);
    double *y = alloc_aligned((size_t)n);
    double *x0 = alloc_aligned((size_t)n);
    double *y0 = alloc_aligned((size_t)n);
    if (!x || !y || !x0 || !y0) {
        fprintf(stderr, "allocation failed for level1 n=%d\n", n);
        free(x);
        free(y);
        free(x0);
        free(y0);
        return;
    }

    fill_random(x0, (size_t)n);
    fill_random(y0, (size_t)n);

    if (lib->daxpy) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            copy_buf(y, y0, (size_t)n);
            lib->daxpy(n, 1.25, x, 1, y, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            copy_buf(y, y0, (size_t)n);
            double t0 = now_seconds();
            lib->daxpy(n, 1.25, x, 1, y, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += y[n - 1];
        }
        write_result(writer, lib->name, "daxpy", n, iters, elapsed);
    }

    if (lib->dscal) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            lib->dscal(n, 2.0, x, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            double t0 = now_seconds();
            lib->dscal(n, 2.0, x, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1];
        }
        write_result(writer, lib->name, "dscal", n, iters, elapsed);
    }

    if (lib->ddot) {
        for (int warm = 0; warm < 4; warm++) {
            g_sink += lib->ddot(n, x0, 1, y0, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            double t0 = now_seconds();
            double v = lib->ddot(n, x0, 1, y0, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += v;
        }
        write_result(writer, lib->name, "ddot", n, iters, elapsed);
    }

    if (lib->dnrm2) {
        for (int warm = 0; warm < 4; warm++) {
            g_sink += lib->dnrm2(n, x0, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            double t0 = now_seconds();
            double v = lib->dnrm2(n, x0, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += v;
        }
        write_result(writer, lib->name, "dnrm2", n, iters, elapsed);
    }

    if (lib->dasum) {
        for (int warm = 0; warm < 4; warm++) {
            g_sink += lib->dasum(n, x0, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            double t0 = now_seconds();
            double v = lib->dasum(n, x0, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += v;
        }
        write_result(writer, lib->name, "dasum", n, iters, elapsed);
    }

    if (lib->drotm) {
        double param[5] = { -1.0, 1.0, 0.5, -0.5, 1.0 };
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            copy_buf(y, y0, (size_t)n);
            lib->drotm(n, x, 1, y, 1, param);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            copy_buf(y, y0, (size_t)n);
            double t0 = now_seconds();
            lib->drotm(n, x, 1, y, 1, param);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1] + y[n - 1];
        }
        write_result(writer, lib->name, "drotm", n, iters, elapsed);
    }

    if (lib->drotmg) {
        for (int warm = 0; warm < 4; warm++) {
            double d1 = 2.0, d2 = 3.0, b1 = 4.0, p[5];
            lib->drotmg(&d1, &d2, &b1, 5.0, p);
            g_sink += p[0];
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            double d1 = 2.0, d2 = 3.0, b1 = 4.0, p[5];
            double t0 = now_seconds();
            lib->drotmg(&d1, &d2, &b1, 5.0, p);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += p[0] + d1 + d2 + b1;
        }
        write_result(writer, lib->name, "drotmg", n, iters, elapsed);
    }

    free(x);
    free(y);
    free(x0);
    free(y0);
}

static void bench_level2(BlasLib *lib, JsonWriter *writer, int n, int iters) {
    int packed_size = n * (n + 1) / 2;
    int kband = 1;
    int lda_band = kband + 1;

    double *a_gen = alloc_aligned((size_t)n * (size_t)n);
    double *a_sym = alloc_aligned((size_t)n * (size_t)n);
    double *a_tri = alloc_aligned((size_t)n * (size_t)n);
    double *x = alloc_aligned((size_t)n);
    double *y = alloc_aligned((size_t)n);
    double *x0 = alloc_aligned((size_t)n);
    double *y0 = alloc_aligned((size_t)n);
    double *ap_sym = alloc_aligned((size_t)packed_size);
    double *ap_tri = alloc_aligned((size_t)packed_size);
    double *ab_tri = alloc_aligned((size_t)lda_band * (size_t)n);

    if (!a_gen || !a_sym || !a_tri || !x || !y || !x0 || !y0 || !ap_sym || !ap_tri || !ab_tri) {
        fprintf(stderr, "allocation failed for level2 n=%d\n", n);
        free(a_gen);
        free(a_sym);
        free(a_tri);
        free(x);
        free(y);
        free(x0);
        free(y0);
        free(ap_sym);
        free(ap_tri);
        free(ab_tri);
        return;
    }

    make_general_matrix(a_gen, n);
    make_symmetric_matrix(a_sym, n);
    make_upper_triangular_matrix(a_tri, n);
    make_packed_upper_symmetric(ap_sym, n);
    make_packed_upper_triangular(ap_tri, n);
    make_upper_triangular_band(ab_tri, n, kband, lda_band);
    fill_random(x0, (size_t)n);
    fill_random(y0, (size_t)n);

    if (lib->dgemv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(y, y0, (size_t)n);
            lib->dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, a_gen, n, x0, 1, 0.0, y, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(y, y0, (size_t)n);
            double t0 = now_seconds();
            lib->dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, a_gen, n, x0, 1, 0.0, y, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += y[n - 1];
        }
        write_result(writer, lib->name, "dgemv", n, iters, elapsed);

        elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(y, y0, (size_t)n);
            double t0 = now_seconds();
            lib->dgemv(CblasColMajor, CblasTrans, n, n, 1.0, a_gen, n, x0, 1, 0.0, y, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += y[n - 1];
        }
        write_result(writer, lib->name, "dgemv_trans", n, iters, elapsed);
    }

    if (lib->dtrmv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            lib->dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a_tri, n, x, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            double t0 = now_seconds();
            lib->dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a_tri, n, x, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1];
        }
        write_result(writer, lib->name, "dtrmv", n, iters, elapsed);
    }

    if (lib->dtrsv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            lib->dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a_tri, n, x, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            double t0 = now_seconds();
            lib->dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a_tri, n, x, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1];
        }
        write_result(writer, lib->name, "dtrsv", n, iters, elapsed);
    }

    if (lib->dsymv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(y, y0, (size_t)n);
            lib->dsymv(CblasColMajor, CblasUpper, n, 1.0, a_sym, n, x0, 1, 0.0, y, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(y, y0, (size_t)n);
            double t0 = now_seconds();
            lib->dsymv(CblasColMajor, CblasUpper, n, 1.0, a_sym, n, x0, 1, 0.0, y, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += y[n - 1];
        }
        write_result(writer, lib->name, "dsymv", n, iters, elapsed);
    }

    if (lib->dsyr) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(a_sym, a_sym, (size_t)n * (size_t)n);
        }
        double *a_work = alloc_aligned((size_t)n * (size_t)n);
        if (a_work) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(a_work, a_sym, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dsyr(CblasColMajor, CblasUpper, n, 1.0, x0, 1, a_work, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += a_work[n - 1];
            }
            write_result(writer, lib->name, "dsyr", n, iters, elapsed);
            free(a_work);
        }
    }

    if (lib->dsyr2) {
        double *a_work = alloc_aligned((size_t)n * (size_t)n);
        if (a_work) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(a_work, a_sym, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dsyr2(CblasColMajor, CblasUpper, n, 1.0, x0, 1, y0, 1, a_work, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += a_work[n - 1];
            }
            write_result(writer, lib->name, "dsyr2", n, iters, elapsed);
            free(a_work);
        }
    }

    if (lib->dspmv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(y, y0, (size_t)n);
            lib->dspmv(CblasColMajor, CblasUpper, n, 1.0, ap_sym, x0, 1, 0.0, y, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(y, y0, (size_t)n);
            double t0 = now_seconds();
            lib->dspmv(CblasColMajor, CblasUpper, n, 1.0, ap_sym, x0, 1, 0.0, y, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += y[n - 1];
        }
        write_result(writer, lib->name, "dspmv", n, iters, elapsed);
    }

    if (lib->dspr) {
        double *ap_work = alloc_aligned((size_t)packed_size);
        if (ap_work) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(ap_work, ap_sym, (size_t)packed_size);
                double t0 = now_seconds();
                lib->dspr(CblasColMajor, CblasUpper, n, 1.0, x0, 1, ap_work);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += ap_work[packed_size - 1];
            }
            write_result(writer, lib->name, "dspr", n, iters, elapsed);
            free(ap_work);
        }
    }

    if (lib->dspr2) {
        double *ap_work = alloc_aligned((size_t)packed_size);
        if (ap_work) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(ap_work, ap_sym, (size_t)packed_size);
                double t0 = now_seconds();
                lib->dspr2(CblasColMajor, CblasUpper, n, 1.0, x0, 1, y0, 1, ap_work);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += ap_work[packed_size - 1];
            }
            write_result(writer, lib->name, "dspr2", n, iters, elapsed);
            free(ap_work);
        }
    }

    if (lib->dtpmv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            lib->dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap_tri, x, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            double t0 = now_seconds();
            lib->dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap_tri, x, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1];
        }
        write_result(writer, lib->name, "dtpmv", n, iters, elapsed);
    }

    if (lib->dtpsv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            lib->dtpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap_tri, x, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            double t0 = now_seconds();
            lib->dtpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap_tri, x, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1];
        }
        write_result(writer, lib->name, "dtpsv", n, iters, elapsed);
    }

    if (lib->dtbmv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            lib->dtbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, kband, ab_tri, lda_band, x, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            double t0 = now_seconds();
            lib->dtbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, kband, ab_tri, lda_band, x, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1];
        }
        write_result(writer, lib->name, "dtbmv", n, iters, elapsed);
    }

    if (lib->dtbsv) {
        for (int warm = 0; warm < 4; warm++) {
            copy_buf(x, x0, (size_t)n);
            lib->dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, kband, ab_tri, lda_band, x, 1);
        }
        double elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            double t0 = now_seconds();
            lib->dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, kband, ab_tri, lda_band, x, 1);
            double t1 = now_seconds();
            elapsed += t1 - t0;
            g_sink += x[n - 1];
        }
        write_result(writer, lib->name, "dtbsv", n, iters, elapsed);
    }

    free(a_gen);
    free(a_sym);
    free(a_tri);
    free(x);
    free(y);
    free(x0);
    free(y0);
    free(ap_sym);
    free(ap_tri);
    free(ab_tri);
}

static void bench_level3(BlasLib *lib, JsonWriter *writer, int n, int iters) {
    double *a_gen = alloc_aligned((size_t)n * (size_t)n);
    double *b_gen = alloc_aligned((size_t)n * (size_t)n);
    double *a_sym = alloc_aligned((size_t)n * (size_t)n);
    double *a_tri = alloc_aligned((size_t)n * (size_t)n);
    double *c0 = alloc_aligned((size_t)n * (size_t)n);
    double *b0 = alloc_aligned((size_t)n * (size_t)n);

    if (!a_gen || !b_gen || !a_sym || !a_tri || !c0 || !b0) {
        fprintf(stderr, "allocation failed for level3 n=%d\n", n);
        free(a_gen);
        free(b_gen);
        free(a_sym);
        free(a_tri);
        free(c0);
        free(b0);
        return;
    }

    make_general_matrix(a_gen, n);
    make_general_matrix(b_gen, n);
    make_symmetric_matrix(a_sym, n);
    make_upper_triangular_matrix(a_tri, n);
    fill_random(c0, (size_t)n * (size_t)n);
    fill_random(b0, (size_t)n * (size_t)n);

    if (lib->dgemm) {
        double *c = alloc_aligned((size_t)n * (size_t)n);
        if (c) {
            for (int warm = 0; warm < 4; warm++) {
                copy_buf(c, c0, (size_t)n * (size_t)n);
                lib->dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a_gen, n, b_gen, n, 0.0, c, n);
            }
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(c, c0, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a_gen, n, b_gen, n, 0.0, c, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += c[n - 1];
            }
            write_result(writer, lib->name, "dgemm", n, iters, elapsed);
            free(c);
        }
    }

    if (lib->dsyrk) {
        double *c = alloc_aligned((size_t)n * (size_t)n);
        if (c) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(c, c0, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, n, 1.0, a_gen, n, 0.0, c, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += c[n - 1];
            }
            write_result(writer, lib->name, "dsyrk", n, iters, elapsed);
            free(c);
        }
    }

    if (lib->dsyr2k) {
        double *c = alloc_aligned((size_t)n * (size_t)n);
        if (c) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(c, c0, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dsyr2k(CblasColMajor, CblasUpper, CblasNoTrans, n, n, 1.0, a_gen, n, b_gen, n, 0.0, c, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += c[n - 1];
            }
            write_result(writer, lib->name, "dsyr2k", n, iters, elapsed);
            free(c);
        }
    }

    if (lib->dsymm) {
        double *c = alloc_aligned((size_t)n * (size_t)n);
        if (c) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(c, c0, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dsymm(CblasColMajor, CblasLeft, CblasUpper, n, n, 1.0, a_sym, n, b_gen, n, 0.0, c, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += c[n - 1];
            }
            write_result(writer, lib->name, "dsymm", n, iters, elapsed);
            free(c);
        }
    }

    if (lib->dtrmm) {
        double *b = alloc_aligned((size_t)n * (size_t)n);
        if (b) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(b, b0, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, a_tri, n, b, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += b[n - 1];
            }
            write_result(writer, lib->name, "dtrmm", n, iters, elapsed);
            free(b);
        }
    }

    if (lib->dtrsm) {
        double *b = alloc_aligned((size_t)n * (size_t)n);
        if (b) {
            double elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(b, b0, (size_t)n * (size_t)n);
                double t0 = now_seconds();
                lib->dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, a_tri, n, b, n);
                double t1 = now_seconds();
                elapsed += t1 - t0;
                g_sink += b[n - 1];
            }
            write_result(writer, lib->name, "dtrsm", n, iters, elapsed);
            free(b);
        }
    }

    free(a_gen);
    free(b_gen);
    free(a_sym);
    free(a_tri);
    free(c0);
    free(b0);
}

int main(int argc, char **argv) {
    int min_n = 256;
    int max_n = 262144;
    double step = 2.0;
    int iters = 0;
    const char *json_path = "bench_results.json";
    const char *openblas_path = getenv("OPENBLAS_PATH");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min") == 0 && i + 1 < argc) {
            min_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            step = atof(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--openblas") == 0 && i + 1 < argc) {
            openblas_path = argv[++i];
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (min_n < 1 || max_n < min_n) {
        fprintf(stderr, "invalid size range\n");
        return 1;
    }
    if (!(step > 1.0)) {
        fprintf(stderr, "--step must be > 1.0\n");
        return 1;
    }
    if (!openblas_path) {
        openblas_path = "libopenblas.dylib";
    }
    const int threads = 1;
    set_max_thread_env(threads);
    srand(1U);

    BlasLib accel;
    BlasLib openblas;
    init_accelerate(&accel);
    int have_accel = accel.daxpy != NULL;
    int have_openblas = load_openblas(&openblas, openblas_path);
    if (have_openblas && openblas.set_num_threads) {
        openblas.set_num_threads(threads);
    }

    if (!have_accel) {
        fprintf(stderr, "Accelerate is not available in this build\n");
    }
    if (!have_openblas) {
        fprintf(stderr, "OpenBLAS could not be loaded from %s\n", openblas_path);
    }
    if (!have_accel && !have_openblas) {
        return 1;
    }

    FILE *out = fopen(json_path, "w");
    if (!out) {
        fprintf(stderr, "failed to open %s\n", json_path);
        unload_lib(&openblas);
        return 1;
    }

    JsonWriter writer;
    writer.out = out;
    writer.first = 1;
    json_begin(out, min_n, max_n, step, iters, openblas_path);

    for (int n = min_n; n <= max_n; n = next_geometric_n(n, step)) {
        int local_iters = pick_iters_level1(n, iters);
        if (have_accel) {
            bench_level1(&accel, &writer, n, local_iters);
        }
        if (have_openblas) {
            bench_level1(&openblas, &writer, n, local_iters);
        }
        if (n == max_n) {
            break;
        }
    }

    for (int n = 32; n <= 2048; n *= 2) {
        int local_iters = pick_iters_level2(n, iters);
        if (have_accel) {
            bench_level2(&accel, &writer, n, local_iters);
        }
        if (have_openblas) {
            bench_level2(&openblas, &writer, n, local_iters);
        }
    }

    for (int n = 32; n <= 2048; n *= 2) {
        int local_iters = pick_iters_level3(n, iters);
        if (have_accel) {
            bench_level3(&accel, &writer, n, local_iters);
        }
        if (have_openblas) {
            bench_level3(&openblas, &writer, n, local_iters);
        }
    }

    json_end(out);
    fclose(out);
    unload_lib(&openblas);

    if (fabs(g_sink) == DBL_MAX) {
        printf("%.17g\n", g_sink);
    }

    return 0;
}
