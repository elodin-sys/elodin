/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

#ifndef _LINEARALGEBRA_H_
#define _LINEARALGEBRA_H_

#include <stdio.h>
#include <architecture/utilities/bskLogging.h>

/* Divide by zero epsilon value */
#define DB0_EPS 1e-30

/* define a maximum array size for the functions that need
 to allocate memory within their routine */
#define LINEAR_ALGEBRA_MAX_ARRAY_SIZE (64*64)

#define MXINDEX(dim2, row, col) ((row)*(dim2) + (col))

/* General vectors */
#ifdef __cplusplus
extern "C" {
#endif

    /* N element vectors */
    void    vElementwiseMult(double *v1, size_t dim, double *v2, double *result);
    void    vCopy(double *v, size_t dim, double *result);
    void    vSetZero(double *v, size_t dim);
    void    vSetOnes(double *v, size_t dim);
    void    vAdd(double *v1, size_t dim, double *v2, double *result);
    void    vSubtract(double *v1, size_t dim, double *v2, double *result);
    void    vScale(double scaleFactor, double *v, size_t dim, double *result);
    double  vDot(double *v1, size_t dim, double *v2);
    void    vOuterProduct(double *v1, size_t dim1, double *v2, size_t dim2, void *result);
    void    vtMultM(double *v, void *mx, size_t dim1, size_t dim2, void *result);
    void    vtMultMt(double *v, void *mx, size_t dim1, size_t dim2, void *result);
    double  vNorm(double *v, size_t dim);
    double  vMax(double *v, size_t dim); /* Non-sorted, non-optimized algorithm for finding the max of a small 1-D array*/
    double  vMaxAbs(double *v, size_t dim); /* Non-sorted, non-optimized algorithm for finding the max of the absolute values of the elements of a small 1-D array*/
    void    vNormalize(double *v, size_t dim, double *result);
    int     vIsEqual(double *v1, size_t dim, double *v2, double accuracy);
    int     vIsZero(double *v, size_t dim, double accuracy);
    void    vPrint(FILE *pFile, const char *name, double *v, size_t dim);
    void    vSort(double *Input, double *Output, size_t dim);

    /* 2 element vectors */
    void    v2Set(double v0, double v1, double result[2]);
    void    v2Copy(double v[2], double result[2]);
    void    v2Scale(double scaleFactor, double v[2], double result[2]);
    void    v2SetZero(double v[2]);
    double  v2Dot(double v1[2], double v2[2]);
    int     v2IsEqual(double v1[2], double v2[2], double accuracy);
    int     v2IsZero(double v[2], double accuracy);
    void    v2Add(double v1[2], double v2[2], double result[2]);
    void    v2Subtract(double v1[2], double v2[2], double result[2]);
    double  v2Norm(double v1[2]);
    void    v2Normalize(double v1[2], double result[2]);

    /* 3 element vectors */
    void    v3Set(double v0, double v1, double v2, double result[3]);
    void    v3Copy(double v[3], double result[3]);
    void    v3SetZero(double v[3]);
    void    v3Add(double v1[3], double v2[3], double result[3]);
    void    v3Subtract(double v1[3], double v2[3], double result[3]);
    void    v3Scale(double scaleFactor, double v[3], double result[3]);
    double  v3Dot(double v1[3], double v2[3]);
    void    v3OuterProduct(double v1[3], double v2[3], double result[3][3]);
    void    v3tMultM33(double v[3], double mx[3][3], double result[3]);
    void    v3tMultM33t(double v[3], double mx[3][3], double result[3]);
    double  v3Norm(double v[3]);
    void    v3Normalize(double v[3], double result[3]);
    int     v3IsEqual(double v1[3], double v2[3], double accuracy);
    int     v3IsEqualRel(double v1[3], double v2[3], double accuracy);
    int     v3IsZero(double v[3], double accuracy);
    void    v3Print(FILE *pFile, const char *name, double v[3]);
    void    v3Cross(double v1[3], double v2[3], double result[3]);
    void    v3Perpendicular(double v[3], double result[3]);
    void    v3Tilde(double v[3], double result[3][3]);
    void    v3Sort(double v[3], double result[3]);
    void    v3PrintScreen(const char *name, double v[3]);

    /* 4 element vectors */
    void    v4Set(double v0, double v1, double v2, double v3, double result[4]);
    void    v4Copy(double v[4], double result[4]);
    void    v4SetZero(double v[4]);
    double  v4Dot(double v1[4], double v2[4]);
    double  v4Norm(double v[4]);
    int     v4IsEqual(double v1[4], double v2[4], double accuracy);
    int     v4IsZero(double v[4], double accuracy);

    /* 6 element vectors */
    void    v6Set(double v0, double v1, double v2, double v3, double v4, double v5, double result[6]);
    void    v6Copy(double v[6], double result[6]);
    double  v6Dot(double v1[6], double v2[6]);
    void    v6Scale(double scaleFactor, double v[6], double result[6]);
    void    v6OuterProduct(double v1[6], double v2[6], double result[6][6]);
    int     v6IsEqual(double v1[6], double v2[6], double accuracy);

    /* NxM matrices */
    void    mLeastSquaresInverse(void *mx, size_t dim1, size_t dim2, void *result);
    void    mMinimumNormInverse(void *mx, size_t dim1, size_t dim2, void *result);
    void    mCopy(void *mx, size_t dim1, size_t dim2, void *result);
    void    mSetZero(void *result, size_t dim1, size_t dim2);
    void    mSetIdentity(void *result, size_t dim1, size_t dim2);
    void    mDiag(void *v, size_t dim, void *result);
    void    mTranspose(void *mx, size_t dim1, size_t dim2, void *result);
    void    mAdd(void *mx1, size_t dim1, size_t dim2, void *mx2, void *result);
    void    mSubtract(void *mx1, size_t dim1, size_t dim2, void *mx2, void *result);
    void    mScale(double scaleFactor, void *mx, size_t dim1, size_t dim2, void *result);
    void    mMultM(void *mx1, size_t dim11, size_t dim12,
                   void *mx2, size_t dim21, size_t dim22,
                   void *result);
    void    mtMultM(void *mx1, size_t dim11, size_t dim12,
                    void *mx2, size_t dim21, size_t dim22,
                    void *result);
    void    mMultMt(void *mx1, size_t dim11, size_t dim12,
                    void *mx2, size_t dim21, size_t dim22,
                    void *result);
    void    mtMultMt(void *mx1, size_t dim11, size_t dim12,
                     void *mx2, size_t dim21, size_t dim22,
                     void *result);
    void    mMultV(void *mx, size_t dim1, size_t dim2,
                   void *v,
                   void *result);
    void    mtMultV(void *mx, size_t dim1, size_t dim2,
                    void *v,
                    void *result);
    double  mTrace(void *mx, size_t dim);
    double  mDeterminant(void *mx, size_t dim);
    void    mCofactor(void *mx, size_t dim, void *result);
    int     mInverse(void *mx, size_t dim, void *result);
    int     mIsEqual(void *mx1, size_t dim1, size_t dim2, void *mx2, double accuracy);
    int     mIsZero(void *mx, size_t dim1, size_t dim2, double accuracy);
    void mPrintScreen(const char *name, void *mx, size_t dim1, size_t dim2);
    void    mPrint(FILE *pFile, const char *name, void *mx, size_t dim1, size_t dim2);
    void    mGetSubMatrix(void *mx, size_t dim1, size_t dim2,
                          size_t dim1Start, size_t dim2Start,
                          size_t dim1Result, size_t dim2Result, void *result);
    void    mSetSubMatrix(void *mx, size_t dim1, size_t dim2,
                          void *result, size_t dim1Result, size_t dim2Result,
                          size_t dim1Start, size_t dim2Start);

    /* 2x2 matrices */
    void    m22Set(double m00, double m01,
                   double m10, double m11,
                   double m[2][2]);
    void    m22Copy(double mx[2][2], double result[2][2]);
    void    m22SetZero(double result[2][2]);
    void    m22SetIdentity(double result[2][2]);
    void    m22Transpose(double mx[2][2], double result[2][2]);
    void    m22Add(double mx1[2][2], double mx2[2][2], double result[2][2]);
    void    m22Subtract(double mx1[2][2], double mx2[2][2], double result[2][2]);
    void    m22Scale(double scaleFactor, double mx[2][2], double result[2][2]);
    void    m22MultM22(double mx1[2][2], double mx2[2][2], double result[2][2]);
    void    m22tMultM22(double mx1[2][2], double mx2[2][2], double result[2][2]);
    void    m22MultM22t(double mx1[2][2], double mx2[2][2], double result[2][2]);
    void    m22MultV2(double mx[2][2], double v[2], double result[2]);
    void    m22tMultV2(double mx[2][2], double v[2], double result[2]);
    double  m22Trace(double mx[2][2]);
    double  m22Determinant(double mx[2][2]);
    int     m22IsEqual(double mx1[2][2], double mx2[2][2], double accuracy);
    int     m22IsZero(double mx[2][2], double accuracy);
    void    m22Print(FILE *pFile, const char *name, double mx[2][2]);
    int     m22Inverse(double mx[2][2], double result[2][2]);
    void    m22PrintScreen(const char *name, double mx[2][2]);

    /* 3x3 matrices */
    void    m33Set(double m00, double m01, double m02,
                   double m10, double m11, double m12,
                   double m20, double m21, double m22,
                   double m[3][3]);
    void    m33Copy(double mx[3][3], double result[3][3]);
    void    m33SetZero(double result[3][3]);
    void    m33SetIdentity(double result[3][3]);
    void    m33Transpose(double mx[3][3], double result[3][3]);
    void    m33Add(double mx1[3][3], double mx2[3][3], double result[3][3]);
    void    m33Subtract(double mx1[3][3], double mx2[3][3], double result[3][3]);
    void    m33Scale(double scaleFactor, double mx[3][3], double result[3][3]);
    void    m33MultM33(double mx1[3][3], double mx2[3][3], double result[3][3]);
    void    m33tMultM33(double mx1[3][3], double mx2[3][3], double result[3][3]);
    void    m33MultM33t(double mx1[3][3], double mx2[3][3], double result[3][3]);
    void    m33MultV3(double mx[3][3], double v[3], double result[3]);
    void    m33tMultV3(double mx[3][3], double v[3], double result[3]);
    double  m33Trace(double mx[3][3]);
    double  m33Determinant(double mx[3][3]);
    int     m33IsEqual(double mx1[3][3], double mx2[3][3], double accuracy);
    int     m33IsZero(double mx[3][3], double accuracy);
    void    m33Print(FILE *pfile, const char *name, double mx[3][3]);
    int     m33Inverse(double mx[3][3], double result[3][3]);
    void    m33SingularValues(double mx[3][3], double result[3]);
    void    m33EigenValues(double mx[3][3], double result[3]);
    double  m33ConditionNumber(double mx[3][3]);
    void    m33PrintScreen(const char *name, double mx[3][3]);

    /* 4x4 matrices */
    void    m44Set(double m00, double m01, double m02, double m03,
                   double m10, double m11, double m12, double m13,
                   double m20, double m21, double m22, double m23,
                   double m30, double m31, double m32, double m33,
                   double m[4][4]);
    void    m44Copy(double mx[4][4], double result[4][4]);
    void    m44SetZero(double result[4][4]);
    void    m44MultV4(double mx[4][4], double v[4], double result[4]);
    double  m44Determinant(double mx[4][4]);
    int     m44IsEqual(double mx1[4][4], double mx2[4][4], double accuracy);
    int     m44Inverse(double mx[4][4], double result[4][4]);

    /* 6x6 matrices */
    void    m66Set(double m00, double m01, double m02, double m03, double m04, double m05,
                   double m10, double m11, double m12, double m13, double m14, double m15,
                   double m20, double m21, double m22, double m23, double m24, double m25,
                   double m30, double m31, double m32, double m33, double m34, double m35,
                   double m40, double m41, double m42, double m43, double m44, double m45,
                   double m50, double m51, double m52, double m53, double m54, double m55,
                   double m[6][6]);
    void    m66Copy(double mx[6][6], double result[6][6]);
    void    m66SetZero(double result[6][6]);
    void    m66SetIdentity(double result[6][6]);
    void    m66Transpose(double mx[6][6], double result[6][6]);
    void    m66Get33Matrix(size_t row, size_t col, double m[6][6], double mij[3][3]);
    void    m66Set33Matrix(size_t row, size_t col, double mij[3][3], double m[6][6]);
    void    m66Scale(double scaleFactor, double mx[6][6], double result[6][6]);
    void    m66Add(double mx1[6][6], double mx2[6][6], double result[6][6]);
    void    m66Subtract(double mx1[6][6], double mx2[6][6], double result[6][6]);
    void    m66MultM66(double mx1[6][6], double mx2[6][6], double result[6][6]);
    void    m66tMultM66(double mx1[6][6], double mx2[6][6], double result[6][6]);
    void    m66MultM66t(double mx1[6][6], double mx2[6][6], double result[6][6]);
    void    m66MultV6(double mx[6][6], double v[6], double result[6]);
    void    m66tMultV6(double mx[6][6], double v[6], double result[6]);
    int     m66IsEqual(double mx1[6][6], double mx2[6][6], double accuracy);
    int     m66IsZero(double mx[6][6], double accuracy);

    /* 9x9 matrices */
    void    m99SetZero(double result[9][9]);

    /* additional routines */
    /* Solve cubic formula for x^3 + a[2]*x^2 + a[1]*x + a[0] = 0 */
    void    cubicRoots(double a[3], double result[3]);

    double safeAcos(double x);
    double safeAsin(double x);
    double safeSqrt(double x);

#ifdef __cplusplus
}
#endif

#endif

