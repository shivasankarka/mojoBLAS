#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifndef __APPLE__
enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142};
#endif

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

/* Packed / band routines missing from original bench */
typedef void (*cblas_dtpmv_fn)(int Order, int Uplo, int TransA, int Diag,
    int N, const double *Ap, double *X, int incX);
typedef void (*cblas_dtpsv_fn)(int Order, int Uplo, int TransA, int Diag,
    int N, const double *Ap, double *X, int incX);
typedef void (*cblas_dtbmv_fn)(int Order, int Uplo, int TransA, int Diag,
    int N, int K, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dtbsv_fn)(int Order, int Uplo, int TransA, int Diag,
    int N, int K, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dspmv_fn)(int Order, int Uplo, int N, double alpha,
    const double *Ap, const double *X, int incX, double beta, double *Y, int incY);
typedef void (*cblas_dspr_fn)(int Order, int Uplo, int N, double alpha,
    const double *X, int incX, double *Ap);
typedef void (*cblas_dspr2_fn)(int Order, int Uplo, int N, double alpha,
    const double *X, int incX, const double *Y, int incY, double *Ap);
typedef void (*cblas_drotm_fn)(int N, double *X, int incX, double *Y, int incY,
    const double *P);
typedef void (*cblas_drotmg_fn)(double *d1, double *d2, double *b1,
    double b2, double *P);

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
    /* packed / band */
    cblas_dtpmv_fn  dtpmv;
    cblas_dtpsv_fn  dtpsv;
    cblas_dtbmv_fn  dtbmv;
    cblas_dtbsv_fn  dtbsv;
    cblas_dspmv_fn  dspmv;
    cblas_dspr_fn   dspr;
    cblas_dspr2_fn  dspr2;
    cblas_drotm_fn  drotm;
    cblas_drotmg_fn drotmg;
} BlasLib;

static void init_accelerate(BlasLib *lib) {
    lib->name   = "accelerate";
    lib->handle = NULL;
#ifdef __APPLE__
    lib->daxpy  = (cblas_daxpy_fn)  cblas_daxpy;
    lib->dnrm2  = (cblas_dnrm2_fn)  cblas_dnrm2;
    lib->dasum  = (cblas_dasum_fn)  cblas_dasum;
    lib->ddot   = (cblas_ddot_fn)   cblas_ddot;
    lib->dscal  = (cblas_dscal_fn)  cblas_dscal;
    lib->dgemv  = (cblas_dgemv_fn)  cblas_dgemv;
    lib->dtrmv  = (cblas_dtrmv_fn)  cblas_dtrmv;
    lib->dtrsv  = (cblas_dtrsv_fn)  cblas_dtrsv;
    lib->dsymv  = (cblas_dsymv_fn)  cblas_dsymv;
    lib->dsyr   = (cblas_dsyr_fn)   cblas_dsyr;
    lib->dsyr2  = (cblas_dsyr2_fn)  cblas_dsyr2;
    lib->dgemm  = (cblas_dgemm_fn)  cblas_dgemm;
    lib->dsyrk  = (cblas_dsyrk_fn)  cblas_dsyrk;
    lib->dsyr2k = (cblas_dsyr2k_fn) cblas_dsyr2k;
    lib->dsymm  = (cblas_dsymm_fn)  cblas_dsymm;
    lib->dtrmm  = (cblas_dtrmm_fn)  cblas_dtrmm;
    lib->dtrsm  = (cblas_dtrsm_fn)  cblas_dtrsm;
    lib->dtpmv  = (cblas_dtpmv_fn)  cblas_dtpmv;
    lib->dtpsv  = (cblas_dtpsv_fn)  cblas_dtpsv;
    lib->dtbmv  = (cblas_dtbmv_fn)  cblas_dtbmv;
    lib->dtbsv  = (cblas_dtbsv_fn)  cblas_dtbsv;
    lib->dspmv  = (cblas_dspmv_fn)  cblas_dspmv;
    lib->dspr   = (cblas_dspr_fn)   cblas_dspr;
    lib->dspr2  = (cblas_dspr2_fn)  cblas_dspr2;
    lib->drotm  = (cblas_drotm_fn)  cblas_drotm;
    lib->drotmg = (cblas_drotmg_fn) cblas_drotmg;
#endif
}

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
    lib->dtrmm  = (cblas_dtrmm_fn) checked_dlsym(lib->handle, "cblas_dtrmm");
    lib->dtrsm  = (cblas_dtrsm_fn) checked_dlsym(lib->handle, "cblas_dtrsm");
    lib->dtpmv  = (cblas_dtpmv_fn) checked_dlsym(lib->handle, "cblas_dtpmv");
    lib->dtpsv  = (cblas_dtpsv_fn) checked_dlsym(lib->handle, "cblas_dtpsv");
    lib->dtbmv  = (cblas_dtbmv_fn) checked_dlsym(lib->handle, "cblas_dtbmv");
    lib->dtbsv  = (cblas_dtbsv_fn) checked_dlsym(lib->handle, "cblas_dtbsv");
    lib->dspmv  = (cblas_dspmv_fn) checked_dlsym(lib->handle, "cblas_dspmv");
    lib->dspr   = (cblas_dspr_fn)  checked_dlsym(lib->handle, "cblas_dspr");
    lib->dspr2  = (cblas_dspr2_fn) checked_dlsym(lib->handle, "cblas_dspr2");
    lib->drotm  = (cblas_drotm_fn) checked_dlsym(lib->handle, "cblas_drotm");
    lib->drotmg = (cblas_drotmg_fn)checked_dlsym(lib->handle, "cblas_drotmg");
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

static void copy_buf(double *dst, const double *src, size_t n) {
    memcpy(dst, src, sizeof(double) * n);
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
    double *x0 = NULL;
    double *y0 = NULL;
    if (posix_memalign((void **)&x, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&y, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&x0, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&y0, 64, sizeof(double) * (size_t)n) != 0) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(x); free(y); free(x0); free(y0);
        return;
    }
    fill_random(x0, n);
    fill_random(y0, n);
    copy_buf(x, x0, (size_t)n);
    copy_buf(y, y0, (size_t)n);

    for (int warm = 0; warm < 5; warm++) {
        lib->daxpy(n, 1.1, x, 1, y, 1);
        lib->dscal(n, 1.01, x, 1);
    }

    double t0, t1;

    /* axpy: reset y each iteration. */
    double axpy_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(x, x0, (size_t)n);
        copy_buf(y, y0, (size_t)n);
        t0 = now_seconds();
        lib->daxpy(n, 1.25, x, 1, y, 1);
        t1 = now_seconds();
        axpy_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "axpy", n, iters, axpy_elapsed);

    /* scal: reset x each iteration. */
    double scal_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(x, x0, (size_t)n);
        t0 = now_seconds();
        lib->dscal(n, 2.0, x, 1);
        t1 = now_seconds();
        scal_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "scal", n, iters, scal_elapsed);

    copy_buf(x, x0, (size_t)n);
    copy_buf(y, y0, (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->ddot(n, x, 1, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "dot", n, iters, t1 - t0);

    copy_buf(x, x0, (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dnrm2(n, x, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "nrm2", n, iters, t1 - t0);

    copy_buf(x, x0, (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dasum(n, x, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "sum", n, iters, t1 - t0);

    if (lib->drotm) {
        /* rotm: reset x and y each iteration. */
        double param[5] = {-1.0, 1.0, 0.5, -0.5, 1.0};
        double rotm_elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            copy_buf(x, x0, (size_t)n);
            copy_buf(y, y0, (size_t)n);
            t0 = now_seconds();
            lib->drotm(n, x, 1, y, 1, param);
            t1 = now_seconds();
            rotm_elapsed += t1 - t0;
        }
        WRITE_RESULT(lib_name, "rotm", n, iters, rotm_elapsed);
    }

    if (lib->drotmg) {
        /* rotmg: reset inputs each iteration. */
        double rotmg_elapsed = 0.0;
        for (int it = 0; it < iters; it++) {
            double d1 = 2.0, d2 = 3.0, b1 = 4.0, b2 = 5.0, p[5] = {0};
            t0 = now_seconds();
            lib->drotmg(&d1, &d2, &b1, b2, p);
            t1 = now_seconds();
            rotmg_elapsed += t1 - t0;
        }
        WRITE_RESULT(lib_name, "rotmg", n, iters, rotmg_elapsed);
    }

    free(x); free(y); free(x0); free(y0);
}

static void bench_level2(BlasLib *lib, const char *lib_name, FILE *out, int n, int iters, int *first) {
    double *a = NULL, *x = NULL, *y = NULL;
    double *a0 = NULL, *x0 = NULL, *y0 = NULL;
    if (posix_memalign((void **)&a, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&x, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&y, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&a0, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&x0, 64, sizeof(double) * (size_t)n) != 0 ||
        posix_memalign((void **)&y0, 64, sizeof(double) * (size_t)n) != 0) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(a); free(x); free(y); free(a0); free(x0); free(y0);
        return;
    }
    fill_random(a0, n * n);
    fill_random(x0, n);
    fill_random(y0, n);
    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(x, x0, (size_t)n);
    copy_buf(y, y0, (size_t)n);

    double one_d = 1.0, zero_d = 0.0;

    for (int warm = 0; warm < 5; warm++) {
        lib->dgemv(CblasColMajor, CblasNoTrans, n, n, one_d, a, n, x, 1, zero_d, y, 1);
        lib->dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
        lib->dsymv(CblasColMajor, CblasUpper, n, one_d, a, n, x, 1, zero_d, y, 1);
    }

    double t0, t1;

    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(x, x0, (size_t)n);
    copy_buf(y, y0, (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dgemv(CblasColMajor, CblasNoTrans, n, n, one_d, a, n, x, 1, zero_d, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "gemv", n, iters, t1 - t0);

    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(x, x0, (size_t)n);
    copy_buf(y, y0, (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dgemv(CblasColMajor, CblasTrans, n, n, one_d, a, n, x, 1, zero_d, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "gemv_trans", n, iters, t1 - t0);

    /* dtrmv: reset x each iteration. */
    double trmv_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(x, x0, (size_t)n);
        t0 = now_seconds();
        lib->dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
        t1 = now_seconds();
        trmv_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "trmv", n, iters, trmv_elapsed);

    double trsv_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(x, x0, (size_t)n);
        t0 = now_seconds();
        lib->dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
        t1 = now_seconds();
        trsv_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "trsv", n, iters, trsv_elapsed);

    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(x, x0, (size_t)n);
    copy_buf(y, y0, (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsymv(CblasColMajor, CblasUpper, n, one_d, a, n, x, 1, zero_d, y, 1);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "symv", n, iters, t1 - t0);

    /* dsyr: reset a each iteration. */
    double syr_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(a, a0, (size_t)n * (size_t)n);
        copy_buf(x, x0, (size_t)n);
        t0 = now_seconds();
        lib->dsyr(CblasColMajor, CblasUpper, n, one_d, x, 1, a, n);
        t1 = now_seconds();
        syr_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "syr", n, iters, syr_elapsed);

    /* dsyr2: reset a each iteration. */
    double syr2_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(a, a0, (size_t)n * (size_t)n);
        copy_buf(x, x0, (size_t)n);
        copy_buf(y, y0, (size_t)n);
        t0 = now_seconds();
        lib->dsyr2(CblasColMajor, CblasUpper, n, one_d, x, 1, y, 1, a, n);
        t1 = now_seconds();
        syr2_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "syr2", n, iters, syr2_elapsed);

    /* --- Packed and band routines --- */
    int packed_size = n * (n + 1) / 2;
    int kband = 1;
    int lda_band = kband + 1;
    double *ap = NULL, *ap0 = NULL, *ab = NULL, *ab0 = NULL;
    if (posix_memalign((void **)&ap,  64, sizeof(double) * (size_t)packed_size) == 0 &&
        posix_memalign((void **)&ap0, 64, sizeof(double) * (size_t)packed_size) == 0 &&
        posix_memalign((void **)&ab,  64, sizeof(double) * (size_t)(lda_band * n)) == 0 &&
        posix_memalign((void **)&ab0, 64, sizeof(double) * (size_t)(lda_band * n)) == 0) {

        fill_random(ap0, packed_size);
        fill_random(ab0, lda_band * n);
        int k2 = 0;
        for (int j = 0; j < n; j++) { ap0[k2] += (double)(n + 1); k2 += j + 2; }
        for (int j = 0; j < n; j++) ab0[kband + j * lda_band] += (double)(n + 1);

        if (lib->dspmv) {
            copy_buf(ap, ap0, (size_t)packed_size);
            copy_buf(x, x0, (size_t)n);
            copy_buf(y, y0, (size_t)n);
            t0 = now_seconds();
            for (int it = 0; it < iters; it++)
                lib->dspmv(CblasColMajor, CblasUpper, n, one_d, ap, x, 1, zero_d, y, 1);
            t1 = now_seconds();
            WRITE_RESULT(lib_name, "spmv", n, iters, t1 - t0);
        }

        if (lib->dspr) {
            /* dspr: reset ap each iteration. */
            double spr_elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(ap, ap0, (size_t)packed_size);
                copy_buf(x, x0, (size_t)n);
                t0 = now_seconds();
                lib->dspr(CblasColMajor, CblasUpper, n, one_d, x, 1, ap);
                t1 = now_seconds();
                spr_elapsed += t1 - t0;
            }
            WRITE_RESULT(lib_name, "spr", n, iters, spr_elapsed);
        }

        if (lib->dspr2) {
            /* dspr2: reset ap each iteration. */
            double spr2_elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(ap, ap0, (size_t)packed_size);
                copy_buf(x, x0, (size_t)n);
                copy_buf(y, y0, (size_t)n);
                t0 = now_seconds();
                lib->dspr2(CblasColMajor, CblasUpper, n, one_d, x, 1, y, 1, ap);
                t1 = now_seconds();
                spr2_elapsed += t1 - t0;
            }
            WRITE_RESULT(lib_name, "spr2", n, iters, spr2_elapsed);
        }

        if (lib->dtpmv) {
            /* dtpmv: reset x each iteration. */
            double tpmv_elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(x, x0, (size_t)n);
                t0 = now_seconds();
                lib->dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap0, x, 1);
                t1 = now_seconds();
                tpmv_elapsed += t1 - t0;
            }
            WRITE_RESULT(lib_name, "tpmv", n, iters, tpmv_elapsed);
        }

        if (lib->dtpsv) {
            double tpsv_elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(x, x0, (size_t)n);
                t0 = now_seconds();
                lib->dtpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, ap0, x, 1);
                t1 = now_seconds();
                tpsv_elapsed += t1 - t0;
            }
            WRITE_RESULT(lib_name, "tpsv", n, iters, tpsv_elapsed);
        }

        if (lib->dtbmv) {
            /* dtbmv: reset x each iteration. */
            double tbmv_elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(x, x0, (size_t)n);
                t0 = now_seconds();
                lib->dtbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                           n, kband, ab0, lda_band, x, 1);
                t1 = now_seconds();
                tbmv_elapsed += t1 - t0;
            }
            WRITE_RESULT(lib_name, "tbmv", n, iters, tbmv_elapsed);
        }

        if (lib->dtbsv) {
            double tbsv_elapsed = 0.0;
            for (int it = 0; it < iters; it++) {
                copy_buf(x, x0, (size_t)n);
                t0 = now_seconds();
                lib->dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                           n, kband, ab0, lda_band, x, 1);
                t1 = now_seconds();
                tbsv_elapsed += t1 - t0;
            }
            WRITE_RESULT(lib_name, "tbsv", n, iters, tbsv_elapsed);
        }
    }
    free(ap); free(ap0); free(ab); free(ab0);

    free(a); free(x); free(y); free(a0); free(x0); free(y0);
}

static void bench_level3(BlasLib *lib, const char *lib_name, FILE *out, int n, int iters, int *first) {
    double *a = NULL, *b = NULL, *c = NULL;
    double *a0 = NULL, *b0 = NULL, *c0 = NULL;
    if (posix_memalign((void **)&a, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&b, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&c, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&a0, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&b0, 64, sizeof(double) * (size_t)n * n) != 0 ||
        posix_memalign((void **)&c0, 64, sizeof(double) * (size_t)n * n) != 0) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(a); free(b); free(c); free(a0); free(b0); free(c0);
        return;
    }
    fill_random(a0, n * n);
    fill_random(b0, n * n);
    fill_random(c0, n * n);
    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(b, b0, (size_t)n * (size_t)n);
    copy_buf(c, c0, (size_t)n * (size_t)n);

    double one_d = 1.0;
    double zero_d = 0.0;

    for (int warm = 0; warm < 5; warm++) {
        lib->dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, one_d, a, n, b, n, one_d, c, n);
        lib->dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, n, one_d, a, n, one_d, c, n);
    }

    double t0, t1;

    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(b, b0, (size_t)n * (size_t)n);
    copy_buf(c, c0, (size_t)n * (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, one_d, a, n, b, n, zero_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "gemm", n, iters, t1 - t0);

    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(c, c0, (size_t)n * (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, n, one_d, a, n, zero_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "syrk", n, iters, t1 - t0);

    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(b, b0, (size_t)n * (size_t)n);
    copy_buf(c, c0, (size_t)n * (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsyr2k(CblasColMajor, CblasUpper, CblasNoTrans, n, n, one_d, a, n, b, n, zero_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "syr2k", n, iters, t1 - t0);

    copy_buf(a, a0, (size_t)n * (size_t)n);
    copy_buf(b, b0, (size_t)n * (size_t)n);
    copy_buf(c, c0, (size_t)n * (size_t)n);
    t0 = now_seconds();
    for (int it = 0; it < iters; it++) lib->dsymm(CblasColMajor, CblasLeft, CblasUpper, n, n, one_d, a, n, b, n, zero_d, c, n);
    t1 = now_seconds();
    WRITE_RESULT(lib_name, "symm", n, iters, t1 - t0);

    /* trmm: reset b each iteration. */
    double trmm_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(a, a0, (size_t)n * (size_t)n);
        copy_buf(b, b0, (size_t)n * (size_t)n);
        t0 = now_seconds();
        lib->dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, one_d, a, n, b, n);
        t1 = now_seconds();
        trmm_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "trmm", n, iters, trmm_elapsed);

    double trsm_elapsed = 0.0;
    for (int it = 0; it < iters; it++) {
        copy_buf(b, b0, (size_t)n * (size_t)n);
        t0 = now_seconds();
        lib->dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, one_d, a, n, b, n);
        t1 = now_seconds();
        trsm_elapsed += t1 - t0;
    }
    WRITE_RESULT(lib_name, "trsm", n, iters, trsm_elapsed);

    free(a); free(b); free(c); free(a0); free(b0); free(c0);
}

int main(int argc, char **argv) {
    int min_n = 256;
    int max_n = 262144;
    double step = 2.0;
    int iters = 0;
    const char *json_path = "bench_results.json";
    const char *openblas_path = getenv("OPENBLAS_PATH");

    setenv("OPENBLAS_NUM_THREADS", "1", 0);
    setenv("OMP_NUM_THREADS", "1", 0);

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

    BlasLib accel   = {0};
    BlasLib openblas = {0};

    fprintf(stderr, "Initialising Accelerate (linked)\n");
    init_accelerate(&accel);

    fprintf(stderr, "Loading OpenBLAS from %s\n", openblas_path);
    if (!load_lib(&openblas, openblas_path)) {
        fprintf(stderr, "Failed to load OpenBLAS - continuing without it\n");
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
        if (openblas.daxpy) bench_level1(&openblas, "openblas", out, n, local_iters, &first);
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
        if (openblas.dgemv) bench_level2(&openblas, "openblas", out, n, local_iters, &first);
    }

    fprintf(stderr, "Benchmarking Level 3...\n");
    int l3_min = 32, l3_max = 512;
    for (int n = l3_min; n <= l3_max; n *= 2) {
        int local_iters = iters;
        if (local_iters <= 0) {
            const double target_ops = 1e7;
            local_iters = (int)(target_ops / (double)(n * n * n));
            if (local_iters < 1) local_iters = 1;
            if (local_iters < 3) local_iters = 3;
        }

        bench_level3(&accel, "accelerate", out, n, local_iters, &first);
        if (openblas.dgemm) bench_level3(&openblas, "openblas", out, n, local_iters, &first);
    }

    write_json_footer(out);
    fclose(out);

    unload_lib(&openblas);

    if (sink == 0.123456789) {
        printf("%f\n", sink);
    }

    fprintf(stderr, "Done. Results written to %s\n", json_path);
    return 0;
}
