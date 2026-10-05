#pragma once
// LAPACK 符号声明（OpenBLAS LP64, Fortran 调用约定）
#include <cstdint>

extern "C" {

// 广义对称特征问题 A x = λ B x 全部特征对（分治）
void dsygvd_(const int32_t* itype, const char* jobz, const char* uplo,
             const int32_t* n, double* a, const int32_t* lda, double* b,
             const int32_t* ldb, double* w, double* work, const int32_t* lwork,
             int32_t* iwork, const int32_t* liwork, int32_t* info);

// 广义对称特征问题 区间/索引 特征对
void dsygvx_(const int32_t* itype, const char* jobz, const char* range,
             const char* uplo, const int32_t* n, double* a, const int32_t* lda,
             double* b, const int32_t* ldb, const double* vl, const double* vu,
             const int32_t* il, const int32_t* iu, const double* abstol,
             int32_t* m, double* w, double* z, const int32_t* ldz, double* work,
             const int32_t* lwork, int32_t* iwork, int32_t* ifail, int32_t* info);

// 对称特征问题全部特征对（分治）,标准型 A x = λ x
void dsyevd_(const char* jobz, const char* uplo, const int32_t* n, double* a,
             const int32_t* lda, double* w, double* work, const int32_t* lwork,
             int32_t* iwork, const int32_t* liwork, int32_t* info);

} // extern "C"
