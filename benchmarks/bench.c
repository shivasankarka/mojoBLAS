#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

/* CBLAS enum values (ref: cblas.h) */
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

/* Level 1 */
typedef void (*cblas_daxpy_fn)(int N, double alpha, const double *X, int incX, double *Y, int incY);
typedef double (*cblas_dnrm2_fn)(int N, const double *X, int incX);
typedef double (*cblas_dasum_fn)(int N, const double *X, int incX);
typedef double (*cblas_ddot_fn)(int N, const double *X, int incX, const double *Y, int incY);
typedef void (*cblas_dscal_fn)(int N, double alpha, double *X, int incX);

/* Level 2 */
typedef void (*cblas_dgemv_fn)(int Order, int TransA, int M, int N,
    double alpha, const double *A, int lda, const double *X, int incX,
    double beta, double *Y, int incY);
typedef void (*cblas_dtrmv_fn)(int Order, int Uplo, int TransA, int Diag,
    int N, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dtrsv_fn)(int Order, int Uplo, int TransA, int Diag,
    int N, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dsymv_fn)(int Order, int Uplo, int N, double alpha,
    const double *A, int lda, const double *X, int incX,
    double beta, double *Y, int incY);
typedef void (*cblas_dsyr_fn)(int Order, int Uplo, int N, double alpha,
    const double *X, int incX, double *A, int lda);
typedef void (*cblas_dsyr2_fn)(int Order, int Uplo, int N, double alpha,
    const double *X, int incX, const double *Y, int incY,
    double *A, int lda);

/* Level 3 */
typedef void (*cblas_dgemm_fn)(int Order, int TransA, int TransB, int M, int N, int K,
    double alpha, const double *A, int lda, const double *B, int ldb,
    double beta, double *C, int ldc);
typedef void (*cblas_dsyrk_fn)(int Order, int Uplo, int Trans, int N, int K,
    double alpha, const double *A, int lda, double beta, double *C, int ldc);
typedef void (*cblas_dsyr2k_fn)(int Order, int Uplo, int Trans, int N, int K,
    double alpha, const double *A, int lda, const double *B, int ldb,
    double beta, double *C, int ldc);
typedef void (*cblas_dsymm_fn)(int Order, int Side, int Uplo, int M, int N,
    double alpha, const double *A, int lda, const double *B, int ldb,
    double beta, double *C, int ldc);
typedef void (*cblas_dtrmm_fn)(int Order, int Side, int Uplo, int TransA, int Diag,
    int M, int N, double alpha, const double *A, int lda, double *B, int ldb);
typedef void (*cblas_dtrsm_fn)(int Order, int Side, int Uplo, int TransA, int Diag,
    int M, int N, double alpha, const double *A, int lda, double *B, int ldb);

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
} BlasLib;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *checked_dlsym(void *handle, const char *symbol) {
    dlerror();
    void *p = dlsym(handle, symbol);
    const char *err = dlerror();
    if (err != NULL) {
        fprintf(stderr, "  dlsym(%s) failed: %s\n", symbol, err);
        return NULL;
    }
    return p;
}

static int load_lib(BlasLib *lib, const char *path) {
    lib->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib->handle) {
        fprintf(stderr, "  dlopen(%s) failed: %s\n", path, dlerror());
        return 0;
    }
    lib->daxpy = (cblas_daxpy_fn)checked_dlsym(lib->handle, "cblas_daxpy");
    lib->dnrm2 = (cblas_dnrm2_fn)checked_dlsym(lib->handle, "cblas_dnrm2");
    lib->dasum = (cblas_dasum_fn)checked_dlsym(lib->handle, "cblas_dasum");
    lib->ddot = (cblas_ddot_fn)checked_dlsym(lib->handle, "cblas_ddot");
    lib->dscal = (cblas_dscal_fn)checked_dlsym(lib->handle, "cblas_dscal");
    lib->dgemv = (cblas_dgemv_fn)checked_dlsym(lib->handle, "cblas_dgemv");
    lib->dtrmv = (cblas_dtrmv_fn)checked_dlsym(lib->handle, "cblas_dtrmv");
    lib->dtrsv = (cblas_dtrsv_fn)checked_dlsym(lib->handle, "cblas_dtrsv");
    lib->dsymv = (cblas_dsymv_fn)checked_dlsym(lib->handle, "cblas_dsymv");
    lib->dsyr = (cblas_dsyr_fn)checked_dlsym(lib->handle, "cblas_dsyr");
    lib->dsyr2 = (cblas_dsyr2_fn)checked_dlsym(lib->handle, "cblas_dsyr2");
    lib->dgemm = (cblas_dgemm_fn)checked_dlsym(lib->handle, "cblas_dgemm");
    lib->dsyrk = (cblas_dsyrk_fn)checked_dlsym(lib->handle, "cblas_dsyrk");
    lib->dsyr2k = (cblas_dsyr2k_fn)checked_dlsym(lib->handle, "cblas_dsyr2k");
    lib->dsymm = (cblas_dsymm_fn)checked_dlsym(lib->handle, "cblas_dsymm");
    lib->dtrmm = (cblas_dtrmm_fn)checked_dlsym(lib->handle, "cblas_dtrmm");
    lib->dtrsm = (cblas_dtrsm_fn)checked_dlsym(lib->handle, "cblas_dtrsm");
    if (!lib->daxpy || !lib->dnrm2 || !lib->dasum || !lib->ddot || !lib->dscal) {
        dlclose(lib->handle);
        lib->handle = NULL;
        return 0;
    }
    return 1;
}

static void unload_lib(BlasLib *lib) {
    if (lib->handle) {
        dlclose(lib->handle);
        lib->handle = NULL;
    }
}

static void fill_random(double *buf, int n) {
    for (int i = 0; i < n; i++) {
        buf[i] = (double)rand() / (double)RAND_MAX;
    }
}

static void write_json_header(FILE *f, int min_n, int max_n, double step, int iters, const char *openblas_path) {
    fprintf(f, "{\n");
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"min_n\": %d,\n", min_n);
    fprintf(f, "    \"max_n\": %d,\n", max_n);
    fprintf(f, "    \"step\": %.6g,\n", step);
    fprintf(f, "    \"iters\": %d,\n", iters);
    fprintf(f, "    \"openblas_path\": \"%s\"\n", openblas_path ? openblas_path : "");
    fprintf(f, "  },\n");
    fprintf(f, "  \"results\": [\n");
}

static void write_json_footer(FILE *f) {
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
}

#define WRITE_RESULT(lib_name, op_name, n_val, iters_val, elapsed) \
    do { \
        if (!*first) fprintf(out, ",\n"); \
        *first = 0; \
        fprintf(out, "    {\"lib\":\"%s\",\"op\":\"%s\",\"n\":%d,\"iters\":%d,\"avg_seconds\":%.9g}", \
                lib_name, op_name, n_val, iters_val, (elapsed) / (double)(iters_val)); \
    } while (0)

static void bench_level1(BlasLib *lib, const char *lib_name, FILE *out, int n, int iters, int *first) {
    double *x = NULL;
    double *y = NULL;
    if (posix_memalign((void **)&x, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&y, 64, sizeof(double) * (size_t)n) != 0) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(x); free(y);
        return;
    }
    fill_random(x, n);
    fill_random(y, n);

    for (int warm = 0; warm < 5; warm++) {
        lib->daxpy(n, 1.1, x, 1, y, 1);
        lib->dscal(n, 1.01, x, 1);
    }

    double t0, t1;

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->daxpy(n, 1.1, x, 1, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "axpy", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dscal(n, 1.01, x, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "scal", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->ddot(n, x, 1, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "dot", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dnrm2(n, x, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "nrm2", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dasum(n, x, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "sum", n, iters, t1 - t0);

    free(x); free(y);
}

static void bench_level2(BlasLib *lib, const char *lib_name, FILE *out, int n, int iters, int *first) {
    double *a = NULL, *x = NULL, *y = NULL;
    if (posix_memalign((void **)&a, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&x, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&y, 64, sizeof(double) * (size_t)n) != 0) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(a); free(x); free(y);
        return;
    }
    fill_random(a, n * n);
    fill_random(x, n);
    fill_random(y, n);

    double one_d = 1.0, zero_d = 0.0;

    for (int warm = 0; warm < 5; warm++) {
        lib->dgemv(CblasColMajor, CblasNoTrans, n, n, one_d, a, n, x, 1, zero_d, y, 1);
        lib->dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
        lib->dsymv(CblasColMajor, CblasUpper, n, one_d, a, n, x, 1, zero_d, y, 1);
    }

    double t0, t1;

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dgemv(CblasColMajor, CblasNoTrans, n, n, one_d, a, n, x, 1, zero_d, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "gemv", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dgemv(CblasColMajor, CblasTrans, n, n, one_d, a, n, x, 1, zero_d, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "gemv_trans", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "trmv", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "trsv", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsymv(CblasColMajor, CblasUpper, n, one_d, a, n, x, 1, zero_d, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "symv", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsyr(CblasColMajor, CblasUpper, n, one_d, x, 1, a, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "syr", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsyr2(CblasColMajor, CblasUpper, n, one_d, x, 1, y, 1, a, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "syr2", n, iters, t1 - t0);

    free(a); free(x); free(y);
}

static void bench_level3(BlasLib *lib, const char *lib_name, FILE *out, int n, int iters, int *first) {
    double *a = NULL, *b = NULL, *c = NULL;
    if (posix_memalign((void **)&a, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&b, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&c, 64, sizeof(double) * (size_t)n * n) != 0) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(a); free(b); free(c);
        return;
    }
    fill_random(a, n * n);
    fill_random(b, n * n);
    fill_random(c, n * n);

    double one_d = 1.0;

    for (int warm = 0; warm < 5; warm++) {
        lib->dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, one_d, a, n, b, n, one_d, c, n);
        lib->dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, n, one_d, a, n, one_d, c, n);
    }

    double t0, t1;

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, one_d, a, n, b, n, one_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "gemm", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, n, one_d, a, n, one_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "syrk", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsyr2k(CblasColMajor, CblasUpper, CblasNoTrans, n, n, one_d, a, n, b, n, one_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "syr2k", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsymm(CblasColMajor, CblasLeft, CblasUpper, n, n, one_d, a, n, b, n, one_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "symm", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, one_d, a, n, b, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "trmm", n, iters, t1 - t0);

    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, one_d, a, n, b, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "trsm", n, iters, t1 - t0);

    free(a); free(b); free(c);
}

int main(int argc, char **argv) {
    int min_n = 256;
    int max_n = 262144;
    double step = 2.0;
    int iters = 0;
    const char *json_path = "bench_results.json";
    const char *openblas_path = getenv("OPENBLAS_PATH");
    const char *accelerate_path = "/System/Library/Frameworks/Accelerate.framework/Accelerate";

    srand((unsigned)time(NULL));

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min") == 0 && i + 1 < argc) {
            min_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--openblas") == 0 && i + 1 < argc) {
            openblas_path = argv[++i];
        } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            step = atof(argv[++i]);
        }
    }

    if (!openblas_path) {
        openblas_path = "libopenblas.dylib";
    }

    BlasLib accel = { .name = "accelerate" };
    BlasLib openblas = { .name = "openblas" };

    fprintf(stderr, "Loading Accelerate from %s\n", accelerate_path);
    if (!load_lib(&accel, accelerate_path)) {
        fprintf(stderr, "Failed to load Accelerate\n");
        return 1;
    }

    fprintf(stderr, "Loading OpenBLAS from %s\n", openblas_path);
    if (!load_lib(&openblas, openblas_path)) {
        fprintf(stderr, "Failed to load OpenBLAS\n");
        unload_lib(&accel);
        return 1;
    }

    FILE *out = fopen(json_path, "w");
    if (!out) {
        fprintf(stderr, "Failed to open %s\n", json_path);
        unload_lib(&accel);
        unload_lib(&openblas);
        return 1;
    }

    write_json_header(out, min_n, max_n, step, iters, openblas_path);

    volatile double sink = 0.0;
    int first = 1;

    fprintf(stderr, "Benchmarking Level 1...\n");
    for (int n = min_n; n <= max_n; n = (int)((double)n * step)) {
        if (n < 1) break;

        int local_iters = iters;
        if (local_iters <= 0) {
            const double target_elems = 1e8;
            local_iters = (int)(target_elems / (double)n);
            if (local_iters < 1) local_iters = 1;
        }

        bench_level1(&accel, "accelerate", out, n, local_iters, &first);
        bench_level1(&openblas, "openblas", out, n, local_iters, &first);
    }

    fprintf(stderr, "Benchmarking Level 2...\n");
    int l2_min = 32, l2_max = 512;
    for (int n = l2_min; n <= l2_max; n *= 2) {
        int local_iters = iters;
        if (local_iters <= 0) {
            const double target_elems = 1e7;
            local_iters = (int)(target_elems / (double)(n * n));
            if (local_iters < 1) local_iters = 1;
        }

        bench_level2(&accel, "accelerate", out, n, local_iters, &first);
        bench_level2(&openblas, "openblas", out, n, local_iters, &first);
    }

    fprintf(stderr, "Benchmarking Level 3...\n");
    int l3_min = 32, l3_max = 512;
    for (int n = l3_min; n <= l3_max; n *= 2) {
        int local_iters = iters;
        if (local_iters <= 0) {
            const double target_ops = 1e7;
            local_iters = (int)(target_ops / (double)(n * n * n));
            if (local_iters < 1) local_iters = 1;
        }

        bench_level3(&accel, "accelerate", out, n, local_iters, &first);
        bench_level3(&openblas, "openblas", out, n, local_iters, &first);
    }

    write_json_footer(out);
    fclose(out);

    unload_lib(&accel);
    unload_lib(&openblas);

    if (sink == 0.123456789) {
        printf("%f\n", sink);
    }

    fprintf(stderr, "Done. Results written to %s\n", json_path);
    return 0;
}
