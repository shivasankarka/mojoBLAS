#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* CBLAS enum values */
enum CBLAS_ORDER {CblasRowMajor = 101, CblasColMajor = 102};
enum CBLAS_TRANSPOSE {CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113};
enum CBLAS_UPLO {CblasUpper = 121, CblasLower = 122};
enum CBLAS_DIAG {CblasNonUnit = 131, CblasUnit = 132};
enum CBLAS_SIDE {CblasLeft = 141, CblasRight = 142};

typedef void (*cblas_dcopy_fn)(int N, const double *X, int incX, double *Y, int incY);
typedef void (*cblas_dscal_fn)(int N, double alpha, double *X, int incX);
typedef void (*cblas_daxpy_fn)(int N, double alpha, const double *X, int incX, double *Y, int incY);
typedef double (*cblas_ddot_fn)(int N, const double *X, int incX, const double *Y, int incY);
typedef double (*cblas_dnrm2_fn)(int N, const double *X, int incX);
typedef double (*cblas_dasum_fn)(int N, const double *X, int incX);
typedef void (*cblas_dswap_fn)(int N, double *X, int incX, double *Y, int incY);
typedef int (*cblas_idamax_fn)(int N, const double *X, int incX);
typedef void (*cblas_drotg_fn)(double *a, double *b, double *c, double *s);
typedef void (*cblas_drot_fn)(int N, double *X, int incX, double *Y, int incY, double c, double s);
typedef void (*cblas_drotm_fn)(int N, double *X, int incX, double *Y, int incY, const double *P);
typedef void (*cblas_drotmg_fn)(double *d1, double *d2, double *x1, const double y1, double *P);

typedef void (*cblas_dgemv_fn)(int Order, int TransA, int M, int N, double alpha, const double *A, int lda, const double *X, int incX, double beta, double *Y, int incY);
typedef void (*cblas_dtrmv_fn)(int Order, int Uplo, int TransA, int Diag, int N, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dtrsv_fn)(int Order, int Uplo, int TransA, int Diag, int N, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dsymv_fn)(int Order, int Uplo, int N, double alpha, const double *A, int lda, const double *X, int incX, double beta, double *Y, int incY);
typedef void (*cblas_dsyr_fn)(int Order, int Uplo, int N, double alpha, const double *X, int incX, double *A, int lda);
typedef void (*cblas_dsyr2_fn)(int Order, int Uplo, int N, double alpha, const double *X, int incX, const double *Y, int incY, double *A, int lda);
typedef void (*cblas_dspr_fn)(int Order, int Uplo, int N, double alpha, const double *X, int incX, double *AP);
typedef void (*cblas_dspr2_fn)(int Order, int Uplo, int N, double alpha, const double *X, int incX, const double *Y, int incY, double *AP);
typedef void (*cblas_dger_fn)(int Order, int M, int N, double alpha, const double *X, int incX, const double *Y, int incY, double *A, int lda);
typedef void (*cblas_dgbmv_fn)(int Order, int TransA, int M, int N, int KL, int KU, double alpha, const double *A, int lda, const double *X, int incX, double beta, double *Y, int incY);
typedef void (*cblas_dsbmv_fn)(int Order, int Uplo, int N, int K, double alpha, const double *A, int lda, const double *X, int incX, double beta, double *Y, int incY);
typedef void (*cblas_dspmv_fn)(int Order, int Uplo, int N, double alpha, const double *AP, const double *X, int incX, double beta, double *Y, int incY);
typedef void (*cblas_dtbmv_fn)(int Order, int Uplo, int TransA, int Diag, int N, int K, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dtbsv_fn)(int Order, int Uplo, int TransA, int Diag, int N, int K, const double *A, int lda, double *X, int incX);
typedef void (*cblas_dtpmv_fn)(int Order, int Uplo, int TransA, int Diag, int N, const double *AP, double *X, int incX);
typedef void (*cblas_dtpsv_fn)(int Order, int Uplo, int TransA, int Diag, int N, const double *AP, double *X, int incX);

typedef void (*cblas_dgemm_fn)(int Order, int TransA, int TransB, int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
typedef void (*cblas_dsyrk_fn)(int Order, int Uplo, int Trans, int N, int K, double alpha, const double *A, int lda, double beta, double *C, int ldc);
typedef void (*cblas_dsyr2k_fn)(int Order, int Uplo, int Trans, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
typedef void (*cblas_dsymm_fn)(int Order, int Side, int Uplo, int M, int N, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
typedef void (*cblas_dtrmm_fn)(int Order, int Side, int Uplo, int TransA, int Diag, int M, int N, double alpha, const double *A, int lda, double *B, int ldb);
typedef void (*cblas_dtrsm_fn)(int Order, int Side, int Uplo, int TransA, int Diag, int M, int N, double alpha, const double *A, int lda, double *B, int ldb);

typedef struct {
    void *handle;
    cblas_dcopy_fn dcopy;
    cblas_dscal_fn dscal;
    cblas_daxpy_fn daxpy;
    cblas_ddot_fn ddot;
    cblas_dnrm2_fn dnrm2;
    cblas_dasum_fn dasum;
    cblas_dswap_fn dswap;
    cblas_idamax_fn idamax;
    cblas_drotg_fn drotg;
    cblas_drot_fn drot;
    cblas_drotm_fn drotm;
    cblas_drotmg_fn drotmg;
    cblas_dgemv_fn dgemv;
    cblas_dtrmv_fn dtrmv;
    cblas_dtrsv_fn dtrsv;
    cblas_dsymv_fn dsymv;
    cblas_dsyr_fn dsyr;
    cblas_dsyr2_fn dsyr2;
    cblas_dspr_fn dspr;
    cblas_dspr2_fn dspr2;
    cblas_dger_fn dger;
    cblas_dgbmv_fn dgbmv;
    cblas_dsbmv_fn dsbmv;
    cblas_dspmv_fn dspmv;
    cblas_dtbmv_fn dtbmv;
    cblas_dtbsv_fn dtbsv;
    cblas_dtpmv_fn dtpmv;
    cblas_dtpsv_fn dtpsv;
    cblas_dgemm_fn dgemm;
    cblas_dsyrk_fn dsyrk;
    cblas_dsyr2k_fn dsyr2k;
    cblas_dsymm_fn dsymm;
    cblas_dtrmm_fn dtrmm;
    cblas_dtrsm_fn dtrsm;
} BlasLib;

static int load_blas(BlasLib *lib, const char *path) {
    lib->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib->handle) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror());
        return 0;
    }

    lib->dcopy = (cblas_dcopy_fn)dlsym(lib->handle, "cblas_dcopy");
    lib->dscal = (cblas_dscal_fn)dlsym(lib->handle, "cblas_dscal");
    lib->daxpy = (cblas_daxpy_fn)dlsym(lib->handle, "cblas_daxpy");
    lib->ddot = (cblas_ddot_fn)dlsym(lib->handle, "cblas_ddot");
    lib->dnrm2 = (cblas_dnrm2_fn)dlsym(lib->handle, "cblas_dnrm2");
    lib->dasum = (cblas_dasum_fn)dlsym(lib->handle, "cblas_dasum");
    lib->dswap = (cblas_dswap_fn)dlsym(lib->handle, "cblas_dswap");
    lib->idamax = (cblas_idamax_fn)dlsym(lib->handle, "cblas_idamax");
    lib->drotg = (cblas_drotg_fn)dlsym(lib->handle, "cblas_drotg");
    lib->drot = (cblas_drot_fn)dlsym(lib->handle, "cblas_drot");
    lib->drotm = (cblas_drotm_fn)dlsym(lib->handle, "cblas_drotm");
    lib->drotmg = (cblas_drotmg_fn)dlsym(lib->handle, "cblas_drotmg");
    lib->dgemv = (cblas_dgemv_fn)dlsym(lib->handle, "cblas_dgemv");
    lib->dtrmv = (cblas_dtrmv_fn)dlsym(lib->handle, "cblas_dtrmv");
    lib->dtrsv = (cblas_dtrsv_fn)dlsym(lib->handle, "cblas_dtrsv");
    lib->dsymv = (cblas_dsymv_fn)dlsym(lib->handle, "cblas_dsymv");
    lib->dsyr = (cblas_dsyr_fn)dlsym(lib->handle, "cblas_dsyr");
    lib->dsyr2 = (cblas_dsyr2_fn)dlsym(lib->handle, "cblas_dsyr2");
    lib->dspr = (cblas_dspr_fn)dlsym(lib->handle, "cblas_dspr");
    lib->dspr2 = (cblas_dspr2_fn)dlsym(lib->handle, "cblas_dspr2");
    lib->dger = (cblas_dger_fn)dlsym(lib->handle, "cblas_dger");
    lib->dgbmv = (cblas_dgbmv_fn)dlsym(lib->handle, "cblas_dgbmv");
    lib->dsbmv = (cblas_dsbmv_fn)dlsym(lib->handle, "cblas_dsbmv");
    lib->dspmv = (cblas_dspmv_fn)dlsym(lib->handle, "cblas_dspmv");
    lib->dtbmv = (cblas_dtbmv_fn)dlsym(lib->handle, "cblas_dtbmv");
    lib->dtbsv = (cblas_dtbsv_fn)dlsym(lib->handle, "cblas_dtbsv");
    lib->dtpmv = (cblas_dtpmv_fn)dlsym(lib->handle, "cblas_dtpmv");
    lib->dtpsv = (cblas_dtpsv_fn)dlsym(lib->handle, "cblas_dtpsv");
    lib->dgemm = (cblas_dgemm_fn)dlsym(lib->handle, "cblas_dgemm");
    lib->dsyrk = (cblas_dsyrk_fn)dlsym(lib->handle, "cblas_dsyrk");
    lib->dsyr2k = (cblas_dsyr2k_fn)dlsym(lib->handle, "cblas_dsyr2k");
    lib->dsymm = (cblas_dsymm_fn)dlsym(lib->handle, "cblas_dsymm");
    lib->dtrmm = (cblas_dtrmm_fn)dlsym(lib->handle, "cblas_dtrmm");
    lib->dtrsm = (cblas_dtrsm_fn)dlsym(lib->handle, "cblas_dtrsm");

    if (!lib->dcopy || !lib->dscal || !lib->daxpy || !lib->ddot) {
        fprintf(stderr, "Failed to load required BLAS functions\n");
        dlclose(lib->handle);
        return 0;
    }
    return 1;
}

static void write_array(FILE *f, const char *name, double *arr, int n) {
    fprintf(f, "%s: ", name);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%.1f", arr[i]);
        if (i < n - 1) fprintf(f, ", ");
    }
    fprintf(f, "\n");
}

int main(int argc, char **argv) {
    const char *openblas_path = getenv("OPENBLAS_PATH");
    if (!openblas_path) openblas_path = "libopenblas.dylib";

    fprintf(stderr, "Loading OpenBLAS from %s\n", openblas_path);

    BlasLib lib = {0};
    if (!load_blas(&lib, openblas_path)) {
        fprintf(stderr, "Failed to load OpenBLAS\n");
        return 1;
    }

    FILE *f1 = fopen("tests/reference_level1.txt", "w");
    FILE *f2 = fopen("tests/reference_level2.txt", "w");
    FILE *f3 = fopen("tests/reference_level3.txt", "w");

    if (!f1 || !f2 || !f3) {
        fprintf(stderr, "Failed to open output files\n");
        return 1;
    }

    fprintf(f1, "# Level 1 Reference Values (OpenBLAS double precision)\n");
    fprintf(f2, "# Level 2 Reference Values (OpenBLAS double precision)\n");
    fprintf(f3, "# Level 3 Reference Values (OpenBLAS double precision)\n");

    double *x, *y, *a, *b, *c;

    fprintf(stderr, "Computing Level 1 reference values...\n");

    /* === LEVEL 1 === */

    /* copy */
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    x[0]=1; x[1]=2; x[2]=3;
    y[0]=0; y[1]=0; y[2]=0;
    lib.dcopy(3, x, 1, y, 1);
    write_array(f1, "copy", y, 3);
    free(x); free(y);

    /* copy_with_increment */
    x = (double*)malloc(sizeof(double) * 6);
    y = (double*)malloc(sizeof(double) * 6);
    x[0]=1; x[2]=2; x[4]=3;
    y[0]=0; y[1]=0; y[2]=0;
    lib.dcopy(3, x, 2, y, 2);
    write_array(f1, "copy_with_increment", y, 6);
    free(x); free(y);

    /* copy_float64 */
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    x[0]=1.5; x[1]=2.5; x[2]=3.5;
    y[0]=0; y[1]=0; y[2]=0;
    lib.dcopy(3, x, 1, y, 1);
    write_array(f1, "copy_float64", y, 3);
    free(x); free(y);

    /* scal */
    x = (double*)malloc(sizeof(double) * 3);
    x[0]=1; x[1]=2; x[2]=3;
    lib.dscal(3, 2.0, x, 1);
    write_array(f1, "scal", x, 3);
    free(x);

    /* axpy */
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    x[0]=1; x[1]=2; x[2]=3;
    y[0]=4; y[1]=5; y[2]=6;
    lib.daxpy(3, 2.0, x, 1, y, 1);
    write_array(f1, "axpy", y, 3);
    free(x); free(y);

    /* dot */
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    x[0]=1; x[1]=2; x[2]=3;
    y[0]=4; y[1]=5; y[2]=6;
    double dot_result = lib.ddot(3, x, 1, y, 1);
    fprintf(f1, "dot: %.1f\n", dot_result);
    free(x); free(y);

    /* dot_with_increment */
    x = (double*)malloc(sizeof(double) * 6);
    y = (double*)malloc(sizeof(double) * 6);
    x[0]=1; x[2]=2; x[4]=3;
    y[0]=4; y[2]=5; y[4]=6;
    dot_result = lib.ddot(3, x, 2, y, 2);
    fprintf(f1, "dot_with_increment: %.1f\n", dot_result);
    free(x); free(y);

    /* dot_float64 */
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    x[0]=1; x[1]=2; x[2]=3;
    y[0]=4; y[1]=5; y[2]=6;
    dot_result = lib.ddot(3, x, 1, y, 1);
    fprintf(f1, "dot_float64: %.1f\n", dot_result);
    free(x); free(y);

    /* dot_orthogonal */
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    x[0]=1; x[1]=0; x[2]=0;
    y[0]=0; y[1]=1; y[2]=0;
    dot_result = lib.ddot(3, x, 1, y, 1);
    fprintf(f1, "dot_orthogonal: %.1f\n", dot_result);
    free(x); free(y);

    /* nrm2 */
    x = (double*)malloc(sizeof(double) * 3);
    x[0]=3; x[1]=4; x[2]=0;
    double nrm2_result = lib.dnrm2(3, x, 1);
    fprintf(f1, "nrm2: %.1f\n", nrm2_result);
    free(x);

    /* asum */
    x = (double*)malloc(sizeof(double) * 4);
    x[0]=1; x[1]=-2; x[2]=3; x[3]=-4;
    double asum_result = lib.dasum(4, x, 1);
    fprintf(f1, "asum: %.1f\n", asum_result);
    free(x);

    /* swap */
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    x[0]=1; x[1]=2; x[2]=3;
    y[0]=4; y[1]=5; y[2]=6;
    lib.dswap(3, x, 1, y, 1);
    fprintf(f1, "swap_x: %.1f, %.1f, %.1f\n", x[0], x[1], x[2]);
    fprintf(f1, "swap_y: %.1f, %.1f, %.1f\n", y[0], y[1], y[2]);
    free(x); free(y);

    /* iamax */
    x = (double*)malloc(sizeof(double) * 5);
    x[0]=1; x[1]=-5; x[2]=3; x[3]=2; x[4]=-4;
    int iamax_result = lib.idamax(5, x, 1);
    fprintf(f1, "iamax: %d\n", iamax_result);
    free(x);

    /* rotg */
    double a_rot = 3.0, b_rot = 4.0, c_rot, s_rot;
    lib.drotg(&a_rot, &b_rot, &c_rot, &s_rot);
    fprintf(f1, "rotg_a: %.1f\n", a_rot);
    fprintf(f1, "rotg_c: %.1f\n", c_rot);
    fprintf(f1, "rotg_s: %.1f\n", s_rot);

    /* rot */
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 2);
    x[0]=1; x[1]=2;
    y[0]=3; y[1]=4;
    lib.drot(2, x, 1, y, 1, 0.0, 1.0);
    fprintf(f1, "rot_x: %.1f, %.1f\n", x[0], x[1]);
    fprintf(f1, "rot_y: %.1f, %.1f\n", y[0], y[1]);
    free(x); free(y);

    /* rotm */
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 2);
    double *p = (double*)malloc(sizeof(double) * 5);
    x[0]=1; x[1]=2;
    y[0]=3; y[1]=4;
    p[0]=-1; p[1]=1; p[2]=2; p[3]=3; p[4]=4;
    lib.drotm(2, x, 1, y, 1, p);
    fprintf(f1, "rotm_x: %.1f, %.1f\n", x[0], x[1]);
    fprintf(f1, "rotm_y: %.1f, %.1f\n", y[0], y[1]);
    free(x); free(y); free(p);

    /* rotmg */
    double d1 = 2.0, d2 = 3.0, x1 = 4.0, y1 = 5.0;
    double pmg[5] = {0, 0, 0, 0, 0};
    lib.drotmg(&d1, &d2, &x1, y1, pmg);
    fprintf(f1, "rotmg_d1: %.7f\n", d1);
    fprintf(f1, "rotmg_d2: %.7f\n", d2);
    fprintf(f1, "rotmg_x1: %.7f\n", x1);
    fprintf(f1, "rotmg_p: %.7f, %.7f, %.7f, %.7f, %.7f\n",
            pmg[0], pmg[1], pmg[2], pmg[3], pmg[4]);

    fprintf(stderr, "Computing Level 2 reference values...\n");

    /* === LEVEL 2 === */

    /* gemv_no_transpose: m=2, n=3, A=[1,4,2,5,3,6], x=[1,1,1], y=[0,0] */
    a = (double*)malloc(sizeof(double) * 6);
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 2);
    a[0]=1; a[1]=4; a[2]=2; a[3]=5; a[4]=3; a[5]=6;
    x[0]=1; x[1]=1; x[2]=1;
    y[0]=0; y[1]=0;
    lib.dgemv(CblasColMajor, CblasNoTrans, 2, 3, 1.0, a, 2, x, 1, 0.0, y, 1);
    write_array(f2, "gemv_no_transpose", y, 2);
    free(a); free(x); free(y);

    /* gemv_transpose: m=2, n=3, A=[1,4,2,5,3,6], x=[1,2] */
    a = (double*)malloc(sizeof(double) * 6);
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 3);
    a[0]=1; a[1]=4; a[2]=2; a[3]=5; a[4]=3; a[5]=6;
    x[0]=1; x[1]=2;
    y[0]=0; y[1]=0; y[2]=0;
    lib.dgemv(CblasColMajor, CblasTrans, 2, 3, 1.0, a, 2, x, 1, 0.0, y, 1);
    write_array(f2, "gemv_transpose", y, 3);
    free(a); free(x); free(y);

    /* gemv_with_beta: m=2, n=2, A=[1,3,2,4], x=[1,1], y=[1,1] */
    a = (double*)malloc(sizeof(double) * 4);
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 2);
    a[0]=1; a[1]=3; a[2]=2; a[3]=4;
    x[0]=1; x[1]=1;
    y[0]=1; y[1]=1;
    lib.dgemv(CblasColMajor, CblasNoTrans, 2, 2, 1.0, a, 2, x, 1, 1.0, y, 1);
    write_array(f2, "gemv_with_beta", y, 2);
    free(a); free(x); free(y);

    /* trmv_upper: n=3, A=[1,0,0,2,4,0,3,5,6], x=[1,1,1] */
    a = (double*)malloc(sizeof(double) * 9);
    x = (double*)malloc(sizeof(double) * 3);
    a[0]=1; a[1]=0; a[2]=0; a[3]=2; a[4]=4; a[5]=0; a[6]=3; a[7]=5; a[8]=6;
    x[0]=1; x[1]=1; x[2]=1;
    lib.dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, a, 3, x, 1);
    write_array(f2, "trmv_upper", x, 3);
    free(a); free(x);

    /* symv_lower: n=2, A=[1,2,2,3], x=[1,1], y=[0,0] */
    a = (double*)malloc(sizeof(double) * 4);
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 2);
    a[0]=1; a[1]=2; a[2]=2; a[3]=3;
    x[0]=1; x[1]=1;
    y[0]=0; y[1]=0;
    lib.dsymv(CblasColMajor, CblasLower, 2, 1.0, a, 2, x, 1, 0.0, y, 1);
    write_array(f2, "symv_lower", y, 2);
    free(a); free(x); free(y);

    /* trsv_solution - solve A*x = b where b is trmv result */
    a = (double*)malloc(sizeof(double) * 9);
    x = (double*)malloc(sizeof(double) * 3);
    a[0]=1; a[1]=0; a[2]=0; a[3]=2; a[4]=4; a[5]=0; a[6]=3; a[7]=5; a[8]=6;
    x[0]=6; x[1]=9; x[2]=6;
    double *x_trsv = (double*)malloc(sizeof(double) * 3);
    x_trsv[0]=6; x_trsv[1]=9; x_trsv[2]=6;
    lib.dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, a, 3, x_trsv, 1);
    write_array(f2, "trsv_solution", x_trsv, 3);
    free(a); free(x_trsv);

    /* ger: A += alpha*x*y' */
    a = (double*)malloc(sizeof(double) * 4);
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 2);
    a[0]=0; a[1]=0; a[2]=0; a[3]=0;
    x[0]=1; x[1]=2;
    y[0]=3; y[1]=4;
    lib.dger(CblasColMajor, 2, 2, 1.0, x, 1, y, 1, a, 2);
    write_array(f2, "ger_result", a, 4);
    free(a); free(x); free(y);

    /* syr: symmetric rank-1 update */
    a = (double*)malloc(sizeof(double) * 4);
    x = (double*)malloc(sizeof(double) * 2);
    a[0]=0; a[1]=0; a[2]=0; a[3]=0;
    x[0]=1; x[1]=2;
    lib.dsyr(CblasColMajor, CblasUpper, 2, 1.0, x, 1, a, 2);
    write_array(f2, "syr_upper", a, 4);
    free(a); free(x);

    /* syr2: symmetric rank-2 update */
    a = (double*)malloc(sizeof(double) * 4);
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 2);
    a[0]=0; a[1]=0; a[2]=0; a[3]=0;
    x[0]=1; x[1]=2;
    y[0]=3; y[1]=4;
    lib.dsyr2(CblasColMajor, CblasUpper, 2, 1.0, x, 1, y, 1, a, 2);
    write_array(f2, "syr2_upper", a, 4);
    free(a); free(x); free(y);

    /* spr: symmetric packed rank-1 update (upper packed) */
    a = (double*)malloc(sizeof(double) * 3);
    x = (double*)malloc(sizeof(double) * 2);
    a[0]=0; a[1]=0; a[2]=0;
    x[0]=1; x[1]=2;
    lib.dspr(CblasColMajor, CblasUpper, 2, 1.0, x, 1, a);
    write_array(f2, "spr_upper", a, 3);
    free(a); free(x);

    /* spr2: symmetric packed rank-2 update (upper packed) */
    a = (double*)malloc(sizeof(double) * 3);
    x = (double*)malloc(sizeof(double) * 2);
    y = (double*)malloc(sizeof(double) * 2);
    a[0]=0; a[1]=0; a[2]=0;
    x[0]=1; x[1]=2;
    y[0]=3; y[1]=4;
    lib.dspr2(CblasColMajor, CblasUpper, 2, 1.0, x, 1, y, 1, a);
    write_array(f2, "spr2_upper", a, 3);
    free(a); free(x); free(y);

    /* gbmv */
    a = (double*)malloc(sizeof(double) * 16);
    x = (double*)malloc(sizeof(double) * 4);
    y = (double*)malloc(sizeof(double) * 4);
    for (int i = 0; i < 16; i++) a[i] = 0;
    a[0]=1; a[1]=2; a[4]=3; a[5]=4; a[6]=5; a[9]=6; a[10]=7; a[14]=8;
    for(int i=0;i<4;i++) { x[i] = 1; y[i] = 0; }
    lib.dgbmv(CblasColMajor, CblasNoTrans, 4, 4, 1, 1, 1.0, a, 4, x, 1, 0.0, y, 1);
    write_array(f2, "gbmv_result", y, 4);
    free(a); free(x); free(y);

    /* sbmv */
    a = (double*)malloc(sizeof(double) * 12);
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    for(int i=0;i<12;i++) a[i] = 0;
    a[0]=1; a[2]=2; a[3]=3; a[5]=4; a[6]=5; a[8]=6;
    x[0]=1; x[1]=1; x[2]=1;
    y[0]=0; y[1]=0; y[2]=0;
    lib.dsbmv(CblasColMajor, CblasLower, 3, 1, 1.0, a, 3, x, 1, 0.0, y, 1);
    write_array(f2, "sbmv_lower", y, 3);
    free(a); free(x); free(y);

    /* spmv */
    a = (double*)malloc(sizeof(double) * 6);
    x = (double*)malloc(sizeof(double) * 3);
    y = (double*)malloc(sizeof(double) * 3);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6;
    x[0]=1; x[1]=1; x[2]=1;
    y[0]=0; y[1]=0; y[2]=0;
    lib.dspmv(CblasColMajor, CblasLower, 3, 1.0, a, x, 1, 0.0, y, 1);
    write_array(f2, "spmv_lower", y, 3);
    free(a); free(x); free(y);

    /* tbmv */
    a = (double*)malloc(sizeof(double) * 6);
    x = (double*)malloc(sizeof(double) * 3);
    for(int i=0;i<6;i++) a[i] = 0;
    a[0]=1; a[1]=2; a[2]=4; a[3]=3; a[4]=5; a[5]=6;
    x[0]=1; x[1]=1; x[2]=1;
    lib.dtbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, 1, a, 2, x, 1);
    write_array(f2, "tbmv_upper", x, 3);
    free(a); free(x);

    /* tbsv */
    a = (double*)malloc(sizeof(double) * 6);
    x = (double*)malloc(sizeof(double) * 3);
    for(int i=0;i<6;i++) a[i] = 0;
    a[0]=1; a[1]=2; a[2]=4; a[3]=3; a[4]=5; a[5]=6;
    x[0]=3; x[1]=9; x[2]=6;
    lib.dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, 1, a, 2, x, 1);
    write_array(f2, "tbsv_upper", x, 3);
    fprintf(f2, "tbsv_upper_prec: %.7f, %.7f, %.7f\n", x[0], x[1], x[2]);
    free(a); free(x);

    /* tpmv */
    a = (double*)malloc(sizeof(double) * 6);
    x = (double*)malloc(sizeof(double) * 3);
    a[0]=1; a[1]=2; a[2]=4; a[3]=3; a[4]=5; a[5]=6;
    x[0]=1; x[1]=1; x[2]=1;
    lib.dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, a, x, 1);
    write_array(f2, "tpmv_upper", x, 3);
    free(a); free(x);

    /* tpsv */
    a = (double*)malloc(sizeof(double) * 6);
    x = (double*)malloc(sizeof(double) * 3);
    a[0]=1; a[1]=2; a[2]=4; a[3]=3; a[4]=5; a[5]=6;
    x[0]=6; x[1]=9; x[2]=6;
    lib.dtpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, a, x, 1);
    write_array(f2, "tpsv_upper", x, 3);
    free(a); free(x);

    fprintf(stderr, "Computing Level 3 reference values...\n");

    /* === LEVEL 3 === */

    /* gemm_no_transpose */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4;
    b[0]=5; b[1]=6; b[2]=7; b[3]=8;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
    write_array(f3, "gemm_no_transpose", c, 4);
    free(a); free(b); free(c);

    /* gemm_transpose_b */
    a = (double*)malloc(sizeof(double) * 6);
    b = (double*)malloc(sizeof(double) * 6);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6;
    b[0]=1; b[1]=3; b[2]=5; b[3]=2; b[4]=4; b[5]=6;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 2, 2, 3, 1.0, a, 2, b, 2, 0.0, c, 2);
    write_array(f3, "gemm_transpose_b", c, 4);
    free(a); free(b); free(c);

    /* gemm_transpose_both */
    a = (double*)malloc(sizeof(double) * 6);
    b = (double*)malloc(sizeof(double) * 6);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6;
    b[0]=1; b[1]=3; b[2]=5; b[3]=2; b[4]=4; b[5]=6;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dgemm(CblasColMajor, CblasTrans, CblasTrans, 2, 2, 3, 1.0, a, 3, b, 2, 0.0, c, 2);
    write_array(f3, "gemm_transpose_both", c, 4);
    free(a); free(b); free(c);

    /* gemm_with_beta */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4;
    b[0]=5; b[1]=6; b[2]=7; b[3]=8;
    c[0]=1; c[1]=1; c[2]=1; c[3]=1;
    lib.dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, a, 2, b, 2, 1.0, c, 2);
    write_array(f3, "gemm_with_beta", c, 4);
    free(a); free(b); free(c);

    /* syrk_upper */
    a = (double*)malloc(sizeof(double) * 6);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, 2, 3, 1.0, a, 2, 0.0, c, 2);
    write_array(f3, "syrk_upper", c, 4);
    free(a); free(c);

    /* syrk_lower */
    a = (double*)malloc(sizeof(double) * 6);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dsyrk(CblasColMajor, CblasLower, CblasNoTrans, 2, 3, 1.0, a, 2, 0.0, c, 2);
    write_array(f3, "syrk_lower", c, 4);
    free(a); free(c);

    /* syrk_transpose */
    a = (double*)malloc(sizeof(double) * 6);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=3; a[2]=5; a[3]=2; a[4]=4; a[5]=6;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dsyrk(CblasColMajor, CblasUpper, CblasTrans, 2, 3, 1.0, a, 3, 0.0, c, 2);
    write_array(f3, "syrk_transpose", c, 4);
    free(a); free(c);

    /* syr2k_upper */
    a = (double*)malloc(sizeof(double) * 6);
    b = (double*)malloc(sizeof(double) * 6);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=4; a[2]=2; a[3]=5; a[4]=3; a[5]=6;
    b[0]=7; b[1]=10; b[2]=8; b[3]=11; b[4]=9; b[5]=12;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dsyr2k(CblasColMajor, CblasUpper, CblasNoTrans, 2, 3, 1.0, a, 2, b, 2, 0.0, c, 2);
    write_array(f3, "syr2k_upper", c, 4);
    free(a); free(b); free(c);

    /* symm_left_upper */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=2; a[3]=3;
    b[0]=1; b[1]=2; b[2]=3; b[3]=4;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dsymm(CblasColMajor, CblasLeft, CblasUpper, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
    write_array(f3, "symm_left_upper", c, 4);
    free(a); free(b); free(c);

    /* symm_right_upper */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=0; a[2]=2; a[3]=3;
    b[0]=1; b[1]=2; b[2]=3; b[3]=4;
    c[0]=0; c[1]=0; c[2]=0; c[3]=0;
    lib.dsymm(CblasColMajor, CblasRight, CblasUpper, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
    write_array(f3, "symm_right_upper", c, 4);
    free(a); free(b); free(c);

    /* trmm_left_upper */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=0; a[2]=2; a[3]=3;
    b[0]=1; b[1]=2; b[2]=3; b[3]=4;
    lib.dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0, a, 2, b, 2);
    write_array(f3, "trmm_left_upper", b, 4);
    free(a); free(b);

    /* trmm_no_unit_diagonal - same as above */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=0; a[2]=2; a[3]=3;
    b[0]=1; b[1]=2; b[2]=3; b[3]=4;
    lib.dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0, a, 2, b, 2);
    write_array(f3, "trmm_no_unit_diagonal", b, 4);
    free(a); free(b);

    /* trsm_left_upper */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=0; a[2]=2; a[3]=3;
    b[0]=5; b[1]=6; b[2]=11; b[3]=12;
    lib.dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0, a, 2, b, 2);
    write_array(f3, "trsm_left_upper", b, 4);
    free(a); free(b);

    /* trsm_no_unit_diagonal - same as above */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=0; a[2]=2; a[3]=3;
    b[0]=5; b[1]=6; b[2]=11; b[3]=12;
    lib.dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0, a, 2, b, 2);
    write_array(f3, "trsm_no_unit_diagonal", b, 4);
    free(a); free(b);

    /* gemm_alpha_zero */
    a = (double*)malloc(sizeof(double) * 4);
    b = (double*)malloc(sizeof(double) * 4);
    c = (double*)malloc(sizeof(double) * 4);
    a[0]=1; a[1]=2; a[2]=3; a[3]=4;
    b[0]=5; b[1]=6; b[2]=7; b[3]=8;
    c[0]=1; c[1]=2; c[2]=3; c[3]=4;
    lib.dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 0.0, a, 2, b, 2, 1.0, c, 2);
    write_array(f3, "gemm_alpha_zero", c, 4);
    free(a); free(b); free(c);

    fclose(f1);
    fclose(f2);
    fclose(f3);

    dlclose(lib.handle);

    fprintf(stderr, "Done. Reference files written:\n");
    fprintf(stderr, "  - tests/reference_level1.txt\n");
    fprintf(stderr, "  - tests/reference_level2.txt\n");
    fprintf(stderr, "  - tests/reference_level3.txt\n");

    return 0;
}
