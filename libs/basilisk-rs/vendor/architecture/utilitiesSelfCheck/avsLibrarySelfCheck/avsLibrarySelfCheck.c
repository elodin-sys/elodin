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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/tests/unitTestComparators.h"
#include "avsLibrarySelfCheck.h"

int testLinearAlgebra(double accuracy)
{
    int errorCount = 0;

    double v2_0[2];
    double v2_1[2];
    double v2_2[2];

    double v3_0[3];
    double v3_1[3];
    double v3_2[3];

    double v4_0[4];
    double v4_1[4];
    double v4_2[4];

    double v6_0[6];
    double v6_1[6];

    double m24_0[2][4];
    double m24_1[2][4];
    double m42_0[4][2];
    double m42_1[4][2];
    double m34_0[3][4];
    double m43_0[4][3];
    double m43_1[4][3];
    double m23_0[2][3];
    double m23_1[2][3];
    double m32_0[3][2];

    double m22_0[2][2];
    double m22_1[2][2];
    double m22_2[2][2];

    double m33_0[3][3];
    double m33_1[3][3];
    double m33_2[3][3];

    double m44_0[4][4];
    double m44_1[4][4];

    double m66_0[6][6];
    double m66_1[6][6];
    double m66_2[6][6];

    double a;

    printf("--testLinearAlgebra, accuracy = %g\n", accuracy);

    /*-----------------------------------------------------------------------*/
    /* generally sized vector checks */


    v3Set(4, 5, 16, v3_0);
    vCopy(v3_0, 3, v3_1);
    if(!vIsEqual(v3_0, 3, v3_1, accuracy)) {
        printf("vCopy failed\n");
        errorCount++;
    }

    v3Set(0, 0, 0, v3_0);
    vSetZero(v3_1, 3);
    if(!vIsEqual(v3_0, 3, v3_1, accuracy)) {
        printf("vSetZero failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(4, 5, 6, v3_1);
    v3Set(5, 7, 9, v3_2);
    vAdd(v3_0, 3, v3_1, v3_0);
    if(!vIsEqual(v3_0, 3, v3_2, accuracy)) {
        printf("vAdd failed\n");
        errorCount++;
    }

    v3Set(4, 6, 8, v3_0);
    v3Set(1, 2, 3, v3_1);
    v3Set(3, 4, 5, v3_2);
    vSubtract(v3_0, 3, v3_1, v3_0);
    if(!vIsEqual(v3_0, 3, v3_2, accuracy)) {
        printf("vSubtract failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(3, 6, 9, v3_2);
    vScale(3, v3_0, 3, v3_0);
    if(!vIsEqual(v3_0, 3, v3_2, accuracy)) {
        printf("vScale failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_1);
    v3Set(4, 5, 6, v3_2);
    a = vDot(v3_1, 3, v3_2);
    if(!isEqual(a, 32.0, accuracy)) {
        printf("vDot failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(4, 5, 6, v3_1);
    m33Set(4, 5, 6,
           8, 10, 12,
           12, 15, 18,
           m33_1);
    vOuterProduct(v3_0, 3, v3_1, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("vOuterProduct failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    m33Set(4, 5, 6,
           7, 8, 9,
           10, 11, 12,
           m33_0);
    v3Set(48, 54, 60, v3_1);
    vtMultM(v3_0, m33_0, 3, 3, v3_0);
    if(!vIsEqual(v3_0, 3, v3_1, accuracy)) {
        printf("vtMultM failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    m33Set(4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        m33_0);
    v3Set(32, 50, 68, v3_1);
    vtMultMt(v3_0, m33_0, 3, 3, v3_0);
    if(!vIsEqual(v3_0, 3, v3_1, accuracy)) {
        printf("vtMultMt failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    a = vNorm(v3_0, 3);
    if(!isEqual(a, 3.74165738677394, accuracy)) {
        printf("vNorm failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(0.267261241912424,
          0.534522483824849,
          0.801783725737273,
          v3_1);
    vNormalize(v3_0, 3, v3_0);
    if(!vIsEqual(v3_0, 3, v3_1, accuracy)) {
        printf("vNormalize failed\n");
        errorCount++;
    }

    //---------

    v2_0[0] = 1;
    v2_0[1] = 2;
    v2Set(1, 2, v2_1);
    if(!v2IsEqual(v2_0, v2_1, accuracy)) {
        printf("v2IsEqual failed\n");
        errorCount++;
    }

    v2Set(1, 2, v2_0);
    v2Copy(v2_0, v2_1);
    if(!v2IsEqual(v2_0, v2_1, accuracy)) {
        printf("v2Copy failed\n");
        errorCount++;
    }

    v2Set(0, 0, v2_0);
    v2SetZero(v2_1);
    if(!v2IsEqual(v2_0, v2_1, accuracy)) {
        printf("v2SetZero failed\n");
        errorCount++;
    }

    v2Set(1, 2, v2_0);
    v2Set(4, 5, v2_1);
    a = v2Dot(v2_0, v2_1);
    if(!isEqual(a, 14.0, accuracy)) {
        printf("v2Dot failed\n");
        errorCount++;
    }

    v2Set(1, 2, v2_0);
    v2Set(4, 5, v2_1);
    v2Set(5, 7, v2_2);
    v2Add(v2_0, v2_1, v2_0);
    if(!v2IsEqual(v2_0, v2_2, accuracy)) {
        printf("v2Add failed\n");
        errorCount++;
    }

    v2Set(4, 6, v2_0);
    v2Set(1, 2, v2_1);
    v2Set(3, 4, v2_2);
    v2Subtract(v2_0, v2_1, v2_0);
    if(!v2IsEqual(v2_0, v2_2, accuracy)) {
        printf("v2Subtract failed\n");
        errorCount++;
    }

    v2Set(3, 4, v2_0);
    a = v2Norm(v2_0);
    if(!isEqual(a, 5.0, accuracy)) {
        printf("v2Norm failed\n");
        errorCount++;
    }

    v2Set(1, 2, v2_0);
    v2Set(3, 6, v2_2);
    v2Scale(3, v2_0, v2_0);
    if(!v2IsEqual(v2_0, v2_2, accuracy)) {
        printf("v2Scale failed\n");
        errorCount++;
    }

    v2Set(1, 1, v2_0);
    v2Set(1./sqrt(2), 1./(sqrt(2)), v2_2);
    v2Normalize(v2_0, v2_0);
    if(!v2IsEqual(v2_0, v2_2, accuracy)) {
        printf("v2Normalize failed\n");
        errorCount++;
    }



    //---------

    v3_0[0] = 1;
    v3_0[1] = 2;
    v3_0[2] = 3;
    v3Set(1, 2, 3, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Set failed\n");
        errorCount++;
    }

    v3Set(4, 5, 16, v3_0);
    v3Copy(v3_0, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Copy failed\n");
        errorCount++;
    }

    v3Set(0, 0, 0, v3_0);
    v3SetZero(v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3SetZero failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(4, 5, 6, v3_1);
    v3Set(5, 7, 9, v3_2);
    v3Add(v3_0, v3_1, v3_0);
    if(!v3IsEqual(v3_0, v3_2, accuracy)) {
        printf("v3Add failed\n");
        errorCount++;
    }

    v3Set(4, 6, 8, v3_0);
    v3Set(1, 2, 3, v3_1);
    v3Set(3, 4, 5, v3_2);
    vSubtract(v3_0, 3, v3_1, v3_0);
    if(!vIsEqual(v3_0, 3, v3_2, accuracy)) {
        printf("vSubtract failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(3, 6, 9, v3_2);
    v3Scale(3, v3_0, v3_0);
    if(!v3IsEqual(v3_0, v3_2, accuracy)) {
        printf("v3Scale failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_1);
    v3Set(4, 5, 6, v3_2);
    a = v3Dot(v3_1, v3_2);
    if(!isEqual(a, 32.0, accuracy)) {
        printf("v3Dot failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(4, 5, 6, v3_1);
    m33Set(4, 5, 6,
           8, 10, 12,
           12, 15, 18,
           m33_1);
    v3OuterProduct(v3_0, v3_1, m33_0);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("v3OuterProduct failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    m33Set(4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        m33_0);
    v3Set(48, 54, 60, v3_1);
    v3tMultM33(v3_0, m33_0, v3_0);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3tMultM33 failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    m33Set(4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        m33_0);
    v3Set(32, 50, 68, v3_1);
    v3tMultM33t(v3_0, m33_0, v3_0);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3tMultM33t failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    a = v3Norm(v3_0);
    if(!isEqual(a, 3.74165738677394, accuracy)) {
        printf("v3Norm failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(0.267261241912424,
          0.534522483824849,
          0.801783725737273,
          v3_1);
    v3Normalize(v3_0, v3_0);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Normalize failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Set(4, 5, 6, v3_1);
    v3Set(-3, 6, -3, v3_2);
    v3Cross(v3_0, v3_1, v3_0);
    if(!v3IsEqual(v3_0, v3_2, accuracy)) {
        printf("v3Cross failed\n");
        errorCount++;
    }

    v3Set(2, 1, 1, v3_0);
    v3Set(-1, 1, 1, v3_1);
    v3Normalize(v3_1, v3_1);
    v3Perpendicular(v3_0, v3_2);
    if(!v3IsEqual(v3_2, v3_1, accuracy)) {
        printf("v3Perpendicular failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Tilde(v3_0, m33_0);
    m33Set(0, -3, 2,
           3, 0, -1,
           -2, 1, 0,
           m33_1);
    if(!m33IsEqual(m33_1, m33_0, accuracy)) {
        printf("v3Tilde failed\n");
        errorCount++;
    }

    v3Set(1, 2, 3, v3_0);
    v3Sort(v3_0, v3_0);
    v3Set(3, 2, 1, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Sort 1 failed\n");
        errorCount++;
    }

    v3Set(1, 3, 2, v3_0);
    v3Sort(v3_0, v3_0);
    v3Set(3, 2, 1, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Sort 2 failed\n");
        errorCount++;
    }

    v3Set(2, 1, 3, v3_0);
    v3Sort(v3_0, v3_0);
    v3Set(3, 2, 1, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Sort 3 failed\n");
        errorCount++;
    }

    v3Set(2, 3, 1, v3_0);
    v3Sort(v3_0, v3_0);
    v3Set(3, 2, 1, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Sort 4 failed\n");
        errorCount++;
    }

    v3Set(3, 1, 2, v3_0);
    v3Sort(v3_0, v3_0);
    v3Set(3, 2, 1, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Sort 5 failed\n");
        errorCount++;
    }

    v3Set(3, 2, 1, v3_0);
    v3Sort(v3_0, v3_0);
    v3Set(3, 2, 1, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("v3Sort 6 failed\n");
        errorCount++;
    }

    //----------

    v4_0[0] = 1;
    v4_0[1] = 2;
    v4_0[2] = 3;
    v4_0[3] = 4;
    v4Set(1, 2, 3, 4, v4_1);
    if(!v4IsEqual(v4_0, v4_1, accuracy)) {
        printf("v4IsEqual failed\n");
        errorCount++;
    }

    v4Set(4, 5, 16, 22, v4_0);
    v4Copy(v4_0, v4_1);
    if(!v4IsEqual(v4_0, v4_1, accuracy)) {
        printf("v4Copy failed\n");
        errorCount++;
    }

    v4Set(0, 0, 0, 0, v4_0);
    v4SetZero(v4_1);
    if(!v4IsEqual(v4_0, v4_1, accuracy)) {
        printf("v4SetZero failed\n");
        errorCount++;
    }

    v4Set(1, 2, 3, 4, v4_1);
    v4Set(4, 5, 6, 7, v4_2);
    a = v4Dot(v4_1, v4_2);
    if(!isEqual(a, 60.0, accuracy)) {
        printf("v4Dot failed\n");
        errorCount++;
    }

    v4Set(1, 2, 3, 4, v4_0);
    a = v4Norm(v4_0);
    if(!isEqual(a, 5.47722557505166, accuracy)) {
        printf("v4Norm failed\n");
        errorCount++;
    }

    /*-----------------------------------------------------------------------*/
    /* Matrix checks */

    m33_0[0][0] = 1;
    m33_0[0][1] = 2;
    m33_0[0][2] = 3;
    m33_0[1][0] = 4;
    m33_0[1][1] = 5;
    m33_0[1][2] = 6;
    m33_0[2][0] = 7;
    m33_0[2][1] = 8;
    m33_0[2][2] = 9;
    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_1);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mIsEqual failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    mCopy(m33_0, 3, 3, m33_1);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mCopy failed\n");
        errorCount++;
    }

    m33Set(0, 0, 0, 0, 0, 0, 0, 0, 0, m33_0);
    mSetZero(m33_1, 3, 3);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mSetZero failed\n");
        errorCount++;
    }

    m33Set(1, 0, 0, 0, 1, 0, 0, 0, 1, m33_0);
    mSetIdentity(m33_1, 3, 3);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mSetIdentity failed\n");
        errorCount++;
    }

    m33Set(1, 0, 0, 0, 2, 0, 0, 0, 3, m33_0);
    v3Set(1, 2, 3, v3_0);
    mDiag(v3_0, 3, m33_1);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mDiag failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(1, 4, 7, 2, 5, 8, 3, 6, 9, m33_1);
    mTranspose(m33_0, 3, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mTranspose failed\n");
        errorCount++;
    }

    m24_0[0][0] = 1;
    m24_0[0][1] = 2;
    m24_0[0][2] = 3;
    m24_0[0][3] = 4;
    m24_0[1][0] = 5;
    m24_0[1][1] = 6;
    m24_0[1][2] = 7;
    m24_0[1][3] = 8;

    m42_0[0][0] = 1;
    m42_0[0][1] = 5;
    m42_0[1][0] = 2;
    m42_0[1][1] = 6;
    m42_0[2][0] = 3;
    m42_0[2][1] = 7;
    m42_0[3][0] = 4;
    m42_0[3][1] = 8;
    mTranspose(m24_0, 2, 4, m42_1);
    if(!mIsEqual(m42_0, 4, 2, m42_1, accuracy)) {
        printf("mTranspose failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(11, 13, 15, 17, 19, 21, 23, 25, 27, m33_2);
    mAdd(m33_0, 3, 3, m33_1, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_2, accuracy)) {
        printf("mAdd failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(-9, -9, -9, -9, -9, -9, -9, -9, -9, m33_2);
    mSubtract(m33_0, 3, 3, m33_1, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_2, accuracy)) {
        printf("mSubtract failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(2, 4, 6, 8, 10, 12, 14, 16, 18, m33_1);
    mScale(2, m33_0, 3, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mScale failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(84, 90, 96, 201, 216, 231, 318, 342, 366, m33_2);
    mMultM(m33_0, 3, 3, m33_1, 3, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_2, accuracy)) {
        printf("mMultM failed\n");
        errorCount++;
    }

    m24_0[0][0] = 1;
    m24_0[0][1] = 2;
    m24_0[0][2] = 3;
    m24_0[0][3] = 4;
    m24_0[1][0] = 5;
    m24_0[1][1] = 6;
    m24_0[1][2] = 7;
    m24_0[1][3] = 8;

    m43_0[0][0] = 1;
    m43_0[0][1] = 5;
    m43_0[0][2] = 9;
    m43_0[1][0] = 2;
    m43_0[1][1] = 6;
    m43_0[1][2] = 10;
    m43_0[2][0] = 3;
    m43_0[2][1] = 7;
    m43_0[2][2] = 11;
    m43_0[3][0] = 4;
    m43_0[3][1] = 8;
    m43_0[3][2] = 12;

    m23_0[0][0] = 30;
    m23_0[0][1] = 70;
    m23_0[0][2] = 110;
    m23_0[1][0] = 70;
    m23_0[1][1] = 174;
    m23_0[1][2] = 278;
    mMultM(m24_0, 2, 4, m43_0, 4, 3, m23_1);
    if(!mIsEqual(m23_1, 2, 3, m23_0, accuracy)) {
        printf("mMultM failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(174, 186, 198, 213, 228, 243, 252, 270, 288, m33_2);
    mtMultM(m33_0, 3, 3, m33_1, 3, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_2, accuracy)) {
        printf("mtMultM failed\n");
        errorCount++;
    }

    m24_0[0][0] = 1;
    m24_0[0][1] = 2;
    m24_0[0][2] = 3;
    m24_0[0][3] = 4;
    m24_0[1][0] = 5;
    m24_0[1][1] = 6;
    m24_0[1][2] = 7;
    m24_0[1][3] = 8;

    m23_0[0][0] = 1;
    m23_0[0][1] = 2;
    m23_0[0][2] = 3;
    m23_0[1][0] = 4;
    m23_0[1][1] = 5;
    m23_0[1][2] = 6;

    m43_0[0][0] = 21;
    m43_0[0][1] = 27;
    m43_0[0][2] = 33;
    m43_0[1][0] = 26;
    m43_0[1][1] = 34;
    m43_0[1][2] = 42;
    m43_0[2][0] = 31;
    m43_0[2][1] = 41;
    m43_0[2][2] = 51;
    m43_0[3][0] = 36;
    m43_0[3][1] = 48;
    m43_0[3][2] = 60;
    mtMultM(m24_0, 2, 4, m23_0, 2, 3, m43_1);
    if(!mIsEqual(m43_0, 4, 3, m43_1, accuracy)) {
        printf("mtMult failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(68, 86, 104, 167, 212, 257, 266, 338, 410, m33_2);
    mMultMt(m33_0, 3, 3, m33_1, 3, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_2, accuracy)) {
        printf("mMultMt failed\n");
        errorCount++;
    }

    m23_0[0][0] = 1;
    m23_0[0][1] = 2;
    m23_0[0][2] = 3;
    m23_0[1][0] = 4;
    m23_0[1][1] = 5;
    m23_0[1][2] = 6;

    m43_0[0][0] = 1;
    m43_0[0][1] = 5;
    m43_0[0][2] = 9;
    m43_0[1][0] = 2;
    m43_0[1][1] = 6;
    m43_0[1][2] = 10;
    m43_0[2][0] = 3;
    m43_0[2][1] = 7;
    m43_0[2][2] = 11;
    m43_0[3][0] = 4;
    m43_0[3][1] = 8;
    m43_0[3][2] = 12;

    m24_0[0][0] = 38;
    m24_0[0][1] = 44;
    m24_0[0][2] = 50;
    m24_0[0][3] = 56;
    m24_0[1][0] = 83;
    m24_0[1][1] = 98;
    m24_0[1][2] = 113;
    m24_0[1][3] = 128;

    mMultMt(m23_0, 2, 3, m43_0, 4, 3, m24_1);
    if(!mIsEqual(m24_0, 2, 4, m24_1, accuracy)) {
        printf("mMultMt failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(138, 174, 210, 171, 216, 261, 204, 258, 312, m33_2);
    mtMultMt(m33_0, 3, 3, m33_1, 3, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_2, accuracy)) {
        printf("mtMultMt failed\n");
        errorCount++;
    }

    m32_0[0][0] = 1;
    m32_0[0][1] = 2;
    m32_0[1][0] = 3;
    m32_0[1][1] = 4;
    m32_0[2][0] = 5;
    m32_0[2][1] = 6;

    m43_0[0][0] = 1;
    m43_0[0][1] = 2;
    m43_0[0][2] = 3;
    m43_0[1][0] = 4;
    m43_0[1][1] = 5;
    m43_0[1][2] = 6;
    m43_0[2][0] = 7;
    m43_0[2][1] = 8;
    m43_0[2][2] = 9;
    m43_0[3][0] = 10;
    m43_0[3][1] = 11;
    m43_0[3][2] = 12;

    m24_0[0][0] = 22;
    m24_0[0][1] = 49;
    m24_0[0][2] = 76;
    m24_0[0][3] = 103;
    m24_0[1][0] = 28;
    m24_0[1][1] = 64;
    m24_0[1][2] = 100;
    m24_0[1][3] = 136;

    mtMultMt(m32_0, 3, 2, m43_0, 4, 3, m24_1);
    if(!mIsEqual(m24_0, 2, 4, m24_1, accuracy)) {
        printf("mtMultMt failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    v3Set(2, 3, 4, v3_0);
    v3Set(20, 47, 74, v3_1);
    mMultV(m33_0, 3, 3, v3_0, v3_0);
    if(!vIsEqual(v3_0, 3, v3_1, accuracy)) {
        printf("mMultV failed\n");
        errorCount++;
    }

    m23_0[0][0] = 1;
    m23_0[0][1] = 2;
    m23_0[0][2] = 3;
    m23_0[1][0] = 4;
    m23_0[1][1] = 5;
    m23_0[1][2] = 6;
    v3Set(2, 3, 4, v3_0);
    v2_1[0] = 20;
    v2_1[1] = 47;
    mMultV(m23_0, 2, 3, v3_0, v2_0);
    if(!vIsEqual(v2_0, 2, v2_1, accuracy)) {
        printf("mMultV failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    v3Set(2, 3, 4, v3_0);
    v3Set(42, 51, 60, v3_1);
    mtMultV(m33_0, 3, 3, v3_0, v3_0);
    if(!vIsEqual(v3_0, 3, v3_1, accuracy)) {
        printf("mtMultV failed\n");
        errorCount++;
    }

    m34_0[0][0] = 1;
    m34_0[0][1] = 2;
    m34_0[0][2] = 3;
    m34_0[0][3] = 4;
    m34_0[1][0] = 5;
    m34_0[1][1] = 6;
    m34_0[1][2] = 7;
    m34_0[1][3] = 8;
    m34_0[2][0] = 9;
    m34_0[2][1] = 10;
    m34_0[2][2] = 11;
    m34_0[2][3] = 12;
    v3Set(2, 3, 4, v3_0);
    v4Set(53, 62, 71, 80, v4_0);
    mtMultV(m34_0, 3, 4, v3_0, v4_1);
    if(!vIsEqual(v4_0, 4, v4_1, accuracy)) {
        printf("mtMultV failed\n");
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    a = mTrace(m33_0, 3);
    if(!isEqual(a, 32.0, accuracy)) {
        printf("mTrace failed %f\n", a);
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    a = mDeterminant(m33_0, 3);
    if(!isEqual(a, 500.0, accuracy)) {
        printf("mDeterminant failed %f\n", a);
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    m33Set(-0.3, 0.0, 0.1, 0.68, -0.12, -0.08, -0.2, 0.1, 0.0, m33_1);
    mInverse(m33_0, 3, m33_0);
    if(!mIsEqual(m33_0, 3, 3, m33_1, accuracy)) {
        printf("mInverse failed\n");
        errorCount++;
    }

    m44_0[0][0] = 4;
    m44_0[0][1] = 5;
    m44_0[0][2] = 6;
    m44_0[0][3] = 7;
    m44_0[1][0] = 8;
    m44_0[1][1] = 10;
    m44_0[1][2] = 22;
    m44_0[1][3] = 36;
    m44_0[2][0] = 22;
    m44_0[2][1] = 15;
    m44_0[2][2] = 18;
    m44_0[2][3] = 15;
    m44_0[3][0] = 1;
    m44_0[3][1] = 2;
    m44_0[3][2] = 3;
    m44_0[3][3] = 4;

    m44_1[0][0] = 11.0 / 40.0;
    m44_1[0][1] = 3.0 / 40.0;
    m44_1[0][2] = 1.0 / 40.0;
    m44_1[0][3] = -5.0 / 4.0;
    m44_1[1][0] = 169.0 / 120.0;
    m44_1[1][1] = -1.0 / 40.0;
    m44_1[1][2] = -7.0 / 40.0;
    m44_1[1][3] = -19.0 / 12.0;
    m44_1[2][0] = -277.0 / 120.0;
    m44_1[2][1] = -7.0 / 40.0;
    m44_1[2][2] = 11.0 / 40.0;
    m44_1[2][3] = 55.0 / 12.0;
    m44_1[3][0] = 23.0 / 24.0;
    m44_1[3][1] = 1.0 / 8.0;
    m44_1[3][2] = -1.0 / 8.0;
    m44_1[3][3] = -25.0 / 12.0;

    mInverse(m44_0, 4, m44_0);
    if(!mIsEqual(m44_0, 4, 4, m44_1, accuracy)) {
        printf("mInverse failed\n");
        errorCount++;
    }

    //----------

    m22_0[0][0] = 1;
    m22_0[0][1] = 2;
    m22_0[1][0] = 3;
    m22_0[1][1] = 4;
    m22Set(1, 2, 3, 4, m22_1);
    if(!m22IsEqual(m22_0, m22_1, accuracy)) {
        printf("m22IsEqual failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Copy(m22_0, m22_1);
    if(!m22IsEqual(m22_0, m22_1, accuracy)) {
        printf("m22Copy failed\n");
        errorCount++;
    }

    m22Set(0, 0, 0, 0, m22_0);
    m22SetZero(m22_1);
    if(!m22IsEqual(m22_0, m22_1, accuracy)) {
        printf("m22SetZero failed\n");
        errorCount++;
    }

    m22Set(1, 0, 0, 1, m22_0);
    m22SetIdentity(m22_1);
    if(!m22IsEqual(m22_0, m22_1, accuracy)) {
        printf("m22SetIdentity failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(1, 3, 2, 4, m22_1);
    m22Transpose(m22_0, m22_0);
    if(!m22IsEqual(m22_0, m22_1, accuracy)) {
        printf("m22Transpose failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(10, 11, 12, 13, m22_1);
    m22Set(11, 13, 15, 17, m22_2);
    m22Add(m22_0, m22_1, m22_0);
    if(!m22IsEqual(m22_0, m22_2, accuracy)) {
        printf("m22Add failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(10, 11, 12, 13, m22_1);
    m22Set(-9, -9, -9, -9, m22_2);
    m22Subtract(m22_0, m22_1, m22_0);
    if(!m22IsEqual(m22_0, m22_2, accuracy)) {
        printf("m22Subtract failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(2, 4, 6, 8, m22_1);
    m22Scale(2, m22_0, m22_0);
    if(!m22IsEqual(m22_0, m22_1, accuracy)) {
        printf("m22Scale failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(10, 11, 12, 13, m22_1);
    m22Set(34, 37, 78, 85, m22_2);
    m22MultM22(m22_0, m22_1, m22_0);
    if(!m22IsEqual(m22_0, m22_2, accuracy)) {
        printf("m22MultM22 failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(10, 11, 12, 13, m22_1);
    m22Set(46, 50, 68, 74, m22_2);
    m22tMultM22(m22_0, m22_1, m22_0);
    if(!m22IsEqual(m22_0, m22_2, accuracy)) {
        printf("m22tMultM22 failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(10, 11, 12, 13, m22_1);
    m22Set(32, 38, 74, 88, m22_2);
    m22MultM22t(m22_0, m22_1, m22_0);
    if(!m22IsEqual(m22_0, m22_2, accuracy)) {
        printf("m22MultM22t failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    v2Set(2, 3, v2_0);
    v2Set(8, 18, v2_1);
    m22MultV2(m22_0, v2_0, v2_0);
    if(!v2IsEqual(v2_0, v2_1, accuracy)) {
        printf("m22MultV2 failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    v2Set(2, 3, v2_0);
    v2Set(11, 16, v2_1);
    m22tMultV2(m22_0, v2_0, v2_0);
    if(!v2IsEqual(v2_0, v2_1, accuracy)) {
        printf("m22tMultV2 failed\n");
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    a = m22Trace(m22_0);
    if(!isEqual(a, 5.0, accuracy)) {
        printf("m22Trace failed %f\n", a);
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    a = m22Determinant(m22_0);
    if(!isEqual(a, -2.0, accuracy)) {
        printf("m22Determinant failed %f\n", a);
        errorCount++;
    }

    m22Set(1, 2, 3, 4, m22_0);
    m22Set(-2.0, 1.0, 1.5, -0.5, m22_1);
    m22Inverse(m22_0, m22_0);
    if(!m22IsEqual(m22_0, m22_1, accuracy)) {
        printf("m22Inverse failed\n");
        errorCount++;
    }

    //----------

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Copy(m33_0, m33_1);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("m33Copy failed\n");
        errorCount++;
    }

    m33Set(0, 0, 0, 0, 0, 0, 0, 0, 0, m33_0);
    m33SetZero(m33_1);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("m33SetZero failed\n");
        errorCount++;
    }

    m33Set(1, 0, 0, 0, 1, 0, 0, 0, 1, m33_0);
    m33SetIdentity(m33_1);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("m33SetIdentity failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(1, 4, 7, 2, 5, 8, 3, 6, 9, m33_1);
    m33Transpose(m33_0, m33_0);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("m33Transpose failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(11, 13, 15, 17, 19, 21, 23, 25, 27, m33_2);
    m33Add(m33_0, m33_1, m33_0);
    if(!m33IsEqual(m33_0, m33_2, accuracy)) {
        printf("m33Add failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(-9, -9, -9, -9, -9, -9, -9, -9, -9, m33_2);
    m33Subtract(m33_0, m33_1, m33_0);
    if(!m33IsEqual(m33_0, m33_2, accuracy)) {
        printf("m33Subtract failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(2, 4, 6, 8, 10, 12, 14, 16, 18, m33_1);
    m33Scale(2, m33_0, m33_0);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("m33Scale failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(84, 90, 96, 201, 216, 231, 318, 342, 366, m33_2);
    m33MultM33(m33_0, m33_1, m33_0);
    if(!m33IsEqual(m33_0, m33_2, accuracy)) {
        printf("m33MultM33 failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(174, 186, 198, 213, 228, 243, 252, 270, 288, m33_2);
    m33tMultM33(m33_0, m33_1, m33_0);
    if(!m33IsEqual(m33_0, m33_2, accuracy)) {
        printf("m33tMultM33 failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    m33Set(10, 11, 12, 13, 14, 15, 16, 17, 18, m33_1);
    m33Set(68, 86, 104, 167, 212, 257, 266, 338, 410, m33_2);
    m33MultM33t(m33_0, m33_1, m33_0);
    if(!m33IsEqual(m33_0, m33_2, accuracy)) {
        printf("m33MultM33t failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    v3Set(2, 3, 4, v3_0);
    v3Set(20, 47, 74, v3_1);
    m33MultV3(m33_0, v3_0, v3_0);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("m33MultV3 failed\n");
        errorCount++;
    }

    m33Set(1, 2, 3, 4, 5, 6, 7, 8, 9, m33_0);
    v3Set(2, 3, 4, v3_0);
    v3Set(42, 51, 60, v3_1);
    m33tMultV3(m33_0, v3_0, v3_0);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("m33tMultV3 failed\n");
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    a = m33Trace(m33_0);
    if(!isEqual(a, 32.0, accuracy)) {
        printf("m33Trace failed %f\n", a);
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    a = m33Determinant(m33_0);
    if(!isEqual(a, 500.0, accuracy)) {
        printf("m33Determinant failed %f\n", a);
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    v3Set(40.7786093479462, 9.66938737160798, 1.26805658603441, v3_0);
    m33SingularValues(m33_0, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("m33SingularValues failed\n");
        errorCount++;
    }

    m33Set(4, 5, 6, 7, 8, 9, 22, 15, 20, m33_0);
    v3Set(32.879131511069, 0.695395691157217, -1.5745272022262, v3_0);
    m33EigenValues(m33_0, v3_1);
    if(!v3IsEqual(v3_0, v3_1, accuracy)) {
        printf("m33EigenValues failed\n");
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    a = m33ConditionNumber(m33_0);
    if(!isEqual(a, 32.1583514466598, accuracy)) {
        printf("m33ConditionNumberFailed\n");
        errorCount++;
    }

    m33Set(4, 5, 6, 8, 10, 22, 22, 15, 18, m33_0);
    m33Set(-0.3, 0.0, 0.1, 0.68, -0.12, -0.08, -0.2, 0.1, 0.0, m33_1);
    m33Inverse(m33_0, m33_0);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("m33Inverse failed\n");
        errorCount++;
    }

    //----------

    m44Set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, m44_0);
    m44Copy(m44_0, m44_1);
    if(!m44IsEqual(m44_0, m44_1, accuracy)) {
        printf("m44Copy failed\n");
        errorCount++;
    }

    m44Set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m44_0);
    m44SetZero(m44_1);
    if(!m44IsEqual(m44_0, m44_1, accuracy)) {
        printf("m44SetZero failed\n");
        errorCount++;
    }

    m44Set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, m44_0);
    v4Set(2, 3, 4, 5, v4_0);
    v4Set(40, 96, 152, 208, v4_1);
    m44MultV4(m44_0, v4_0, v4_0);
    if(!v4IsEqual(v4_0, v4_1, accuracy)) {
        printf("m44MultV4 failed\n");
        errorCount++;
    }

    m44Set(4, 5, 6, 7, 8, 10, 22, 24, 22, 15, 18, 19, 1, 4, 9, 3, m44_0);
    a = m44Determinant(m44_0);
    if(!isEqual(a, -3620.0, accuracy)) {
        printf("m44Determinant failed %f\n", a);
        errorCount++;
    }

    m44Set(4, 5, 6, 7, 8, 10, 22, 24, 22, 15, 18, 19, 1, 4, 9, 3, m44_0);
    m44Set(-0.282872928176796,
           0.0116022099447514,
           0.0939226519337017,
           -0.0276243093922652,
           0.649171270718232,
           -0.140883977900553,
           -0.069060773480663,
           0.0497237569060773,
           -0.285635359116022,
           0.0419889502762431,
           0.0303867403314917,
           0.138121546961326,
           0.0856353591160222,
           0.0580110497237569,
           -0.0303867403314917,
           -0.138121546961326, m44_1);
    m44Inverse(m44_0, m44_0);
    if(!m44IsEqual(m44_0, m44_1, accuracy)) {
        printf("m44Inverse failed\n");
        errorCount++;
    }

    //----------

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m66Copy(m66_0, m66_1);
    if(!m66IsEqual(m66_0, m66_1, accuracy)) {
        printf("m66Copy failed\n");
        errorCount++;
    }

    m66Set(0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0,
           m66_0);
    m66SetZero(m66_1);
    if(!m66IsEqual(m66_0, m66_1, accuracy)) {
        printf("m66SetZero failed\n");
        errorCount++;
    }

    m66Set(
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1,
        m66_0);
    m66SetIdentity(m66_1);
    if(!m66IsEqual(m66_0, m66_1, accuracy)) {
        printf("m66SetIdentity failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36,
        m66_0);
    m66Set(1, 7, 13, 19, 25, 31,
        2, 8, 14, 20, 26, 32,
        3, 9, 15, 21, 27, 33,
        4, 10, 16, 22, 28, 34,
        5, 11, 17, 23, 29, 35,
        6, 12, 18, 24, 30, 36,
        m66_1);
    m66Transpose(m66_0, m66_0);
    if(!m66IsEqual(m66_0, m66_1, accuracy)) {
        printf("m66Transpose failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m33Set(4, 5, 6,
           10, 11, 12,
           16, 17, 18,
           m33_1);
    m66Get33Matrix(0, 1, m66_0, m33_0);
    if(!m33IsEqual(m33_0, m33_1, accuracy)) {
        printf("m66Get33Matrix failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m33Set(54, 55, 56,
           57, 58, 59,
           60, 61, 62,
           m33_0);
    m66Set(1, 2, 3, 54, 55, 56,
           7, 8, 9, 57, 58, 59,
           13, 14, 15, 60, 61, 62,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_1);
    m66Set33Matrix(0, 1, m33_0, m66_0);
    if(!m66IsEqual(m66_0, m66_1, accuracy)) {
        printf("m66Set33Matrix failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m66Set(2,  4,  6,  8, 10, 12,
           14, 16, 18, 20, 22, 24,
           26, 28, 30, 32, 34, 36,
           38, 40, 42, 44, 46, 48,
           50, 52, 54, 56, 58, 60,
           62, 64, 66, 68, 70, 72,
           m66_1);
    m66Scale(2.0, m66_0, m66_0);
    if(!m66IsEqual(m66_0, m66_1, accuracy)) {
        printf("m66Scale failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m66Set(10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27,
           28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39,
           40, 41, 42, 43, 44, 45,
           m66_1);
    m66Set(11, 13, 15, 17, 19, 21,
           23, 25, 27, 29, 31, 33,
           35, 37, 39, 41, 43, 45,
           47, 49, 51, 53, 55, 57,
           59, 61, 63, 65, 67, 69,
           71, 73, 75, 77, 79, 81,
           m66_2);
    m66Add(m66_0, m66_1, m66_0);
    if(!m66IsEqual(m66_0, m66_2, accuracy)) {
        printf("m66Add failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m66Set(10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27,
           28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39,
           40, 41, 42, 43, 44, 45,
           m66_1);
    m66Set(-9, -9, -9, -9, -9, -9,
           -9, -9, -9, -9, -9, -9,
           -9, -9, -9, -9, -9, -9,
           -9, -9, -9, -9, -9, -9,
           -9, -9, -9, -9, -9, -9,
           -9, -9, -9, -9, -9, -9,
           m66_2);
    m66Subtract(m66_0, m66_1, m66_0);
    if(!m66IsEqual(m66_0, m66_2, accuracy)) {
        printf("m66Subtract failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m66Set(10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27,
           28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39,
           40, 41, 42, 43, 44, 45,
           m66_1);
    m66Set(630, 651, 672, 693, 714, 735,
           1530, 1587, 1644, 1701, 1758, 1815,
           2430, 2523, 2616, 2709, 2802, 2895,
           3330, 3459, 3588, 3717, 3846, 3975,
           4230, 4395, 4560, 4725, 4890, 5055,
           5130, 5331, 5532, 5733, 5934, 6135,
           m66_2);
    m66MultM66(m66_0, m66_1, m66_0);
    if(!m66IsEqual(m66_0, m66_2, accuracy)) {
        printf("m66MultM66 failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m66Set(10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27,
           28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39,
           40, 41, 42, 43, 44, 45,
           m66_1);
    m66Set(3030, 3126, 3222, 3318, 3414, 3510,
           3180, 3282, 3384, 3486, 3588, 3690,
           3330, 3438, 3546, 3654, 3762, 3870,
           3480, 3594, 3708, 3822, 3936, 4050,
           3630, 3750, 3870, 3990, 4110, 4230,
           3780, 3906, 4032, 4158, 4284, 4410,
           m66_2);
    m66tMultM66(m66_0, m66_1, m66_0);
    if(!m66IsEqual(m66_0, m66_2, accuracy)) {
        printf("m66tMultM66 failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    m66Set(10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27,
           28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39,
           40, 41, 42, 43, 44, 45,
           m66_1);
    m66Set(280, 406, 532, 658, 784, 910,
           730, 1072, 1414, 1756, 2098, 2440,
           1180, 1738, 2296, 2854, 3412, 3970,
           1630, 2404, 3178, 3952, 4726, 5500,
           2080, 3070, 4060, 5050, 6040, 7030,
           2530, 3736, 4942, 6148, 7354, 8560,
           m66_2);
    m66MultM66t(m66_0, m66_1, m66_0);
    if(!m66IsEqual(m66_0, m66_2, accuracy)) {
        printf("m66MultM66t failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    v6Set(10, 11, 12, 13, 14, 15, v6_0);
    v6Set(280, 730, 1180, 1630, 2080, 2530, v6_1);
    m66MultV6(m66_0, v6_0, v6_0);
    if(!v6IsEqual(v6_0, v6_1, accuracy)) {
        printf("m66MultV6 failed\n");
        errorCount++;
    }

    m66Set(1, 2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36,
           m66_0);
    v6Set(10, 11, 12, 13, 14, 15, v6_0);
    v6Set(1305, 1380, 1455, 1530, 1605, 1680, v6_1);
    m66tMultV6(m66_0, v6_0, v6_0);
    if(!v6IsEqual(v6_0, v6_1, accuracy)) {
        printf("m66tMultV6 failed\n");
        errorCount++;
    }

    //----------

    v3Set(-27, -72, -6, v3_0);
    v3Set(12.1228937846324, -5.73450994222507, -0.38838384240732, v3_1);
    cubicRoots(v3_0, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("cubicRoots failed\n");
        errorCount++;
    }

    return errorCount;
}

int testOrbitalAnomalies(double accuracy)
{
    int errorCount = 0;

    double Ecc;
    double e;
    double f;
    double M;
    double N;
    double H;

    printf("--testOrbitalAnomalies, accuracy = %g\n", accuracy);

    Ecc = 0.3;
    e = .1;
    f = E2f(Ecc, e);
    if(!isEqual(f, 0.33111382522243943, accuracy)) {
        printf("E2f(%g, %g) failed\n", Ecc, e);
        errorCount++;
    }
    Ecc = 0.3 + M_PI;
    e = .1;
    f = E2f(Ecc, e);
    if(!isEqual(f, 3.413322139966247, accuracy)) {
        printf("E2f(%g, %g) failed\n", Ecc, e);
        errorCount++;
    }
    Ecc = 0.3 + M_PI;
    e = .1;
    M = E2M(Ecc, e);
    if(!isEqual(M, 3.471144674255927, accuracy)) {
        printf("E2M(%g, %g) failed\n", Ecc, e);
        errorCount++;
    }
    f = 0.3;
    e = .1;
    Ecc = f2E(f, e);
    if(!isEqual(Ecc, 0.2717294863764543, accuracy)) {
        printf("f2E(%g, %g) failed\n", f, e);
        errorCount++;
    }
    f = 0.3;
    e = 2.1;
    H = f2H(f, e);
    if(!isEqual(H, 0.18054632550895094, accuracy)) {
        printf("f2H(%g, %g) failed\n", f, e);
        errorCount++;
    }
    H = 0.3;
    e = 2.1;
    f = H2f(H, e);
    if(!isEqual(f, 0.4898441475256363, accuracy)) {
        printf("H2f(%g, %g) failed\n", H, e);
        errorCount++;
    }
    H = 0.3;
    e = 2.1;
    N = H2N(H, e);
    if(!isEqual(N, 0.33949261623899946, accuracy)) {
        printf("H2N(%g, %g) failed\n", H, e);
        errorCount++;
    }
    M = 2.0;
    e = 0.3;
    Ecc = M2E(M, e);
    if(!isEqual(Ecc, 2.2360314951724365, accuracy)) {
        printf("M2E(%g, %g) failed with %g \n", M, e, Ecc);
        errorCount++;
    }
    N = 2.0;
    e = 2.3;
    H = N2H(N, e);
    if(!isEqual(H, 1.1098189302932016, accuracy)) {
        printf("N2H(%g, %g) failed\n", N, e);
        errorCount++;
    }

    return errorCount;
}

int testOrbitalHill(double accuracy)
{
    int errorCount = 0;

    double rc_N[3];
    double vc_N[3];
    double rd_N[3];
    double vd_N[3];
    double rho_H[3];
    double rhoPrime_H[3];
    double HN[3][3];
    double trueVector[3];

    printf("--testOrbitalHill, accuracy = %g\n", accuracy);

    /* test hillFrame()*/
    v3Set(353.38362479494975, 6494.841478640714, 2507.239669788398, rc_N);
    v3Set(-7.073840333019544, -0.5666429544308719, 2.6565522055197555, vc_N);
    hillFrame(rc_N, vc_N, HN);
    double HNtrue[3][3];
    m33Set(0.0506938, 0.931702, 0.35967, -0.93404, -0.0832604,
           0.347329, 0.353553, -0.353553, 0.866025, HNtrue);
    for (int i = 0; i<3; i++) {
        if(!v3IsEqualRel(HN[i], HNtrue[i], accuracy)) {
            printf("orbitalMotion:hillFrame failed case %d\n", i);
            errorCount++;
        }
    }

    /* test hill2rv() */
    v3Set(353.38362479494975, 6494.841478640714, 2507.239669788398, rc_N);
    v3Set(-7.073840333019544, -0.5666429544308719, 2.6565522055197555, vc_N);
    v3Set(-0.286371, 0.012113, 0.875157, rho_H);
    v3Set(0.000689358, 0.000620362, 0.000927434, rhoPrime_H);
    hill2rv(rc_N, vc_N, rho_H, rhoPrime_H, rd_N, vd_N);
    v3Set(353.6672082996106, 6494.264242564805, 2507.898786238764, trueVector);
    if(!v3IsEqualRel(rd_N, trueVector, accuracy)) {
        printf("orbitalMotion:hill2rv failed case rd_N\n");
        errorCount++;
    }
    v3Set(-7.073766857682589, -0.5663665778237081, 2.65770594819381, trueVector);
    if(!v3IsEqualRel(vd_N, trueVector, accuracy)) {
        printf("orbitalMotion:hill2rv failed case vd_N\n");
        errorCount++;
    }

    /* test rv2hill() */
    v3Set(353.38362479494975, 6494.841478640714, 2507.239669788398, rc_N);
    v3Set(-7.073840333019544, -0.5666429544308719, 2.6565522055197555, vc_N);
    v3Set(353.6672082996106, 6494.264242564805, 2507.898786238764, rd_N);
    v3Set(-7.073766857682589, -0.5663665778237081, 2.65770594819381, vd_N);
    rv2hill(rc_N, vc_N, rd_N, vd_N, rho_H, rhoPrime_H);
    v3Set(-0.286371, 0.012113, 0.875157, trueVector);
    if(!v3IsEqualRel(rho_H, trueVector, accuracy)) {
        printf("orbitalMotion:rv2hill failed case rho_H\n");
        errorCount++;
    }
    v3Set(0.000689358, 0.000620362, 0.000927434, trueVector);
    if(!v3IsEqualRel(rhoPrime_H, trueVector, accuracy)) {
        printf("orbitalMotion:rv2hill failed case rhoPrime_H\n");
        errorCount++;
    }


    return errorCount;
}

int testRigidBodyKinematics(double accuracy)
{
    int errorCount = 0;

    double C[3][3];
    double C2[3][3];
    double v3_1[4];
    double v3_2[4];
    double v3[4];
    double om[3];

    printf("--testRigidBodyKinematics, accuracy = %g\n", accuracy);

    v4Set(0.45226701686665, 0.75377836144441, 0.15075567228888, 0.45226701686665, v3_1);
    v4Set(-0.18663083698528, 0.46657709246321, 0.83983876643378, -0.20529392068381, v3_2);
    addEP(v3_1, v3_2, v3);
    v4Set(-0.46986547690254, -0.34044145332460, 0.71745926113861, 0.38545850500388, v3_1);
    if(!vIsEqual(v3, 4, v3_1, accuracy)) {
        printf("addEP failed\n");
        errorCount++;
    }
    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler121(v3_1, v3_2, v3);
    v3Set(-2.96705972839036, 2.44346095279206, 1.41371669411541, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler121 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler123(v3_1, v3_2, v3);
    v3Set(2.65556257351773, -0.34257634487528, -2.38843896474589, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler123 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler131(v3_1, v3_2, v3);
    v3Set(-2.96705972839036, 2.44346095279206, 1.41371669411541, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler123 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler132(v3_1, v3_2, v3);
    v3Set(2.93168877067466, -0.89056295435594, -2.11231276758895, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler132 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler212(v3_1, v3_2, v3);
    v3Set(-2.96705972839036, 2.44346095279206, 1.41371669411541, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler212 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler213(v3_1, v3_2, v3);
    v3Set(2.93168877067466, -0.89056295435594, -2.11231276758895, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler213 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler231(v3_1, v3_2, v3);
    v3Set(2.65556257351773, -0.34257634487528, -2.38843896474589, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler231 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler232(v3_1, v3_2, v3);
    v3Set(-2.96705972839036, 2.44346095279206, 1.41371669411541, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler232 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler312(v3_1, v3_2, v3);
    v3Set(2.65556257351773, -0.34257634487528, -2.38843896474589, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler312 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler313(v3_1, v3_2, v3);
    v3Set(-2.96705972839036, 2.44346095279206, 1.41371669411541, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler313 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler321(v3_1, v3_2, v3);
    v3Set(2.93168877067466, -0.89056295435594, -2.11231276758895, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler321 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    addEuler323(v3_1, v3_2, v3);
    v3Set(-2.96705972839036, 2.44346095279206, 1.41371669411541, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addEuler323 failed\n");
        errorCount++;
    }

    v3Set(1.5, 0.5, 0.5, v3_1);
    v3Set(-0.5, 0.25, 0.15, v3_2);
    addGibbs(v3_1, v3_2, v3);
    v3Set(0.61290322580645, 0.17741935483871, 0.82258064516129, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addGibbs failed\n");
        errorCount++;
    }

    v3Set(1.5, 0.5, 0.5, v3_1);
    v3Set(-0.5, 0.25, 0.15, v3_2);
    addMRP(v3_1, v3_2, v3);
    v3Set(0.58667769962764, -0.34919321472900, 0.43690525444766, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addMRP failed\n");
        errorCount++;
    }

    v3Set(0.0, 0.0, 1.0, v3_1);
    v3Set(0.0, 0.0, 1.0, v3_2);
    addMRP(v3_1, v3_2, v3);
    v3Set(0.0, 0.0, 0.0, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addMRP 360 addition test failed\n");
        errorCount++;
    }


    v3Set(1.5, 0.5, 0.5, v3_1);
    v3Set(-0.5, 0.25, 0.15, v3_2);
    addPRV(v3_1, v3_2, v3);
    v3Set(1.00227389370983, 0.41720669426711, 0.86837149207759, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("addPRV failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler121(v3_1, C);
    v3Set(0.76604444311898, 0.0, 1.0, C2[0]);
    v3Set(-0.16636567534280, 0.96592582628907, 0., C2[1]);
    v3Set(-0.62088515301485, -0.25881904510252, 0., C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler121 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler123(v3_1, C);
    v3Set(0.73994211169385, 0.25881904510252, 0, C2[0]);
    v3Set(-0.19826689127415, 0.96592582628907, 0, C2[1]);
    v3Set(-0.64278760968654, 0, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler123 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler131(v3_1, C);
    v3Set(0.76604444311898, 0, 1.00000000000000, C2[0]);
    v3Set(0.62088515301485, 0.25881904510252, 0, C2[1]);
    v3Set(-0.16636567534280, 0.96592582628907, 0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler131 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler132(v3_1, C);
    v3Set(0.73994211169385, -0.25881904510252, 0, C2[0]);
    v3Set(0.64278760968654, 0, 1.00000000000000, C2[1]);
    v3Set(0.19826689127415, 0.96592582628907, 0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler132 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler212(v3_1, C);
    v3Set(-0.16636567534280, 0.96592582628907, 0, C2[0]);
    v3Set(0.76604444311898, 0, 1.00000000000000, C2[1]);
    v3Set(0.62088515301485, 0.25881904510252, 0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler212 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler213(v3_1, C);
    v3Set(0.19826689127415, 0.96592582628907, 0, C2[0]);
    v3Set(0.73994211169385, -0.25881904510252, 0, C2[1]);
    v3Set(0.64278760968654, 0, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler213 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler231(v3_1, C);
    v3Set(-0.64278760968654, 0, 1.00000000000000, C2[0]);
    v3Set(0.73994211169385, 0.25881904510252, 0, C2[1]);
    v3Set(-0.19826689127415, 0.96592582628907, 0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler231 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler232(v3_1, C);
    v3Set(-0.62088515301485, -0.25881904510252, 0, C2[0]);
    v3Set(0.76604444311898, 0, 1.00000000000000, C2[1]);
    v3Set(-0.16636567534280, 0.96592582628907, 0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler232 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler312(v3_1, C);
    v3Set(-0.19826689127415, 0.96592582628907, 0, C2[0]);
    v3Set(-0.64278760968654, 0, 1.00000000000000, C2[1]);
    v3Set(0.73994211169385, 0.25881904510252, 0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler312 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler313(v3_1, C);
    v3Set(-0.16636567534280, 0.96592582628907, 0, C2[0]);
    v3Set(-0.62088515301485, -0.25881904510252, 0, C2[1]);
    v3Set(0.76604444311898, 0, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler313 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler321(v3_1, C);
    v3Set(0.64278760968654, 0, 1.00000000000000, C2[0]);
    v3Set(0.19826689127415, 0.96592582628907, 0, C2[1]);
    v3Set(0.73994211169385, -0.25881904510252, 0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler321 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BinvEuler323(v3_1, C);
    v3Set(0.62088515301485, 0.25881904510252, 0, C2[0]);
    v3Set(-0.16636567534280, 0.96592582628907, 0, C2[1]);
    v3Set(0.76604444311898, 0, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvEuler323 failed\n");
        errorCount++;
    }

    v3Set(0.25, 0.5, -0.5, v3_1);
    BinvGibbs(v3_1, C);
    v3Set(0.64, -0.32, -0.32, C2[0]);
    v3Set(0.32, 0.64, 0.16, C2[1]);
    v3Set(0.32, -0.16, 0.64, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvGibbs failed\n");
        errorCount++;
    }

    v3Set(0.25, 0.5, -0.5, v3_1);
    BinvMRP(v3_1, C);
    v3Set(0.2304, -0.3072, -0.512, C2[0]);
    v3Set(0.512, 0.384, 0, C2[1]);
    v3Set(0.3072, -0.4096, 0.3840, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvMRP failed\n");
        errorCount++;
    }

    v3Set(0.25, 0.5, -0.5, v3_1);
    BinvPRV(v3_1, C);
    v3Set(0.91897927113877, -0.21824360100796, -0.25875396543858, C2[0]);
    v3Set(0.25875396543858, 0.94936204446173, 0.07873902718102, C2[1]);
    v3Set(0.21824360100796, -0.15975975604225, 0.94936204446173, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BinvPRV failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler121(v3_1, C);
    v3Set(0, -0.40265095531125, -1.50271382293774, C2[0]);
    v3Set(0, 0.96592582628907, -0.25881904510252, C2[1]);
    v3Set(1.00000000000000, 0.30844852683273, 1.15114557365953, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler121 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler123(v3_1, C);
    v3Set(1.26092661459205, -0.33786426809485, 0, C2[0]);
    v3Set(0.25881904510252, 0.96592582628907, 0, C2[1]);
    v3Set(0.81050800458377, -0.21717496528718, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler123 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler131(v3_1, C);
    v3Set(0, 1.50271382293774, -0.40265095531125, C2[0]);
    v3Set(0, 0.25881904510252, 0.96592582628907, C2[1]);
    v3Set(1.00000000000000, -1.15114557365953, 0.30844852683273, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler131 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler132(v3_1, C);
    v3Set(1.26092661459205, 0, 0.33786426809485, C2[0]);
    v3Set(-0.25881904510252, 0, 0.96592582628907, C2[1]);
    v3Set(-0.81050800458377, 1.00000000000000, -0.21717496528718, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler132 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler212(v3_1, C);
    v3Set(-0.40265095531125, 0, 1.50271382293774, C2[0]);
    v3Set(0.96592582628907, 0, 0.25881904510252, C2[1]);
    v3Set(0.30844852683273, 1.00000000000000, -1.15114557365953, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler212 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler213(v3_1, C);
    v3Set(0.33786426809485, 1.26092661459205, 0, C2[0]);
    v3Set(0.96592582628907, -0.25881904510252, 0, C2[1]);
    v3Set(-0.21717496528718, -0.81050800458377, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler213 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler231(v3_1, C);
    v3Set(0, 1.26092661459205, -0.33786426809485, C2[0]);
    v3Set(0, 0.25881904510252, 0.96592582628907, C2[1]);
    v3Set(1.00000000000000, 0.81050800458377, -0.21717496528718, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler231 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler232(v3_1, C);
    v3Set(-1.50271382293774, 0, -0.40265095531125, C2[0]);
    v3Set(-0.25881904510252, 0, 0.96592582628907, C2[1]);
    v3Set(1.15114557365953, 1.00000000000000, 0.30844852683273, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler232 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler312(v3_1, C);
    v3Set(-0.33786426809485, 0, 1.26092661459205, C2[0]);
    v3Set(0.96592582628907, 0, 0.25881904510252, C2[1]);
    v3Set(-0.21717496528718, 1.00000000000000, 0.81050800458377, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler312 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler313(v3_1, C);
    v3Set(-0.40265095531125, -1.50271382293774, 0, C2[0]);
    v3Set(0.96592582628907, -0.25881904510252, 0, C2[1]);
    v3Set(0.30844852683273, 1.15114557365953, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler313 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler321(v3_1, C);
    v3Set(0, 0.33786426809485, 1.26092661459205, C2[0]);
    v3Set(0, 0.96592582628907, -0.25881904510252, C2[1]);
    v3Set(1.00000000000000, -0.21717496528718, -0.81050800458377, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler321 failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    BmatEuler323(v3_1, C);
    v3Set(1.50271382293774, -0.40265095531125, 0, C2[0]);
    v3Set(0.25881904510252, 0.96592582628907, 0, C2[1]);
    v3Set(-1.15114557365953, 0.30844852683273, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatEuler323 failed\n");
        errorCount++;
    }

    v3Set(0.25, 0.5, -0.5, v3_1);
    BmatGibbs(v3_1, C);
    v3Set(1.06250000000000, 0.62500000000000, 0.37500000000000, C2[0]);
    v3Set(-0.37500000000000, 1.25000000000000, -0.50000000000000, C2[1]);
    v3Set(-0.62500000000000, 0, 1.25000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatGibbs failed\n");
        errorCount++;
    }

    v3Set(0.25, 0.5, -0.5, v3_1);
    BmatMRP(v3_1, C);
    v3Set(0.56250000000000, 1.25000000000000, 0.75000000000000, C2[0]);
    v3Set(-0.75000000000000, 0.93750000000000, -1.00000000000000, C2[1]);
    v3Set(-1.25000000000000, 0, 0.93750000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatMRP failed\n");
        errorCount++;
    }

    v3Set(0.2, 0.1, -0.5, v3_1);
    v3Set(0.015, 0.045, -0.005, v3_2);
    v3Scale(1 / D2R, v3_2, v3_2);
    BdotmatMRP(v3_1, v3_2, C);
    v3Set(-0.4583662361046585,  1.7761691649055522,  4.1825919044550091, C2[0]);
    v3Set( 0.6302535746439056, -0.1145915590261646, -4.3544792429942563, C2[1]);
    v3Set(-6.1306484078998089, -0.9167324722093173, -0.5729577951308232, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BdotmatMRP failed\n");
        errorCount++;
    }

    v3Set(0.25, 0.5, -0.5, v3_1);
    BmatPRV(v3_1, C);
    v3Set(0.95793740211924, 0.26051564947019, 0.23948435052981, C2[0]);
    v3Set(-0.23948435052981, 0.97371087632453, -0.14603129894038, C2[1]);
    v3Set(-0.26051564947019, 0.10396870105962, 0.97371087632453, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("BmatPRV failed\n");
        errorCount++;
    }

    v3Set(-0.506611258027956, -0.05213449187759728, 0.860596902153381, C[0]);
    v3Set(-0.7789950887797505, -0.4000755572346052, -0.4828107291273137, C[1]);
    v3Set(0.3694748772194938, -0.9149981110691346, 0.1620702682281828, C[2]);
    C2EP(C, v3_1);
    v4Set(0.2526773896521122, 0.4276078901804977, -0.4859180570232927, 0.7191587243944733, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2EP failed\n");
        errorCount++;
    }

    v3Set(-0.506611258027956, -0.05213449187759728, 0.860596902153381, C[0]);
    v3Set(-0.7789950887797505, -0.4000755572346052, -0.4828107291273137, C[1]);
    v3Set(0.3694748772194938, -0.9149981110691346, 0.1620702682281828, C[2]);
    C2Euler121(C, v3_1);
    v3Set(-3.081087141428621, 2.102046098550739, -1.127921895439695, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler121 failed\n");
        errorCount++;
    }
    C2Euler123(C, v3_1);
    v3Set(1.395488250243478, 0.3784438476398376, 2.147410157986089, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler123 failed\n");
        errorCount++;
    }
    C2Euler131(C, v3_1);
    v3Set(1.631301838956069, 2.102046098550739, 0.4428744313552013, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler131 failed\n");
        errorCount++;
    }
    C2Euler132(C, v3_1);
    v3Set(-2.262757475208626, 0.8930615653924096, 2.511467464302149, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler132 failed\n");
        errorCount++;
    }
    C2Euler212(C, v3_1);
    v3Set(-2.125637903992466, 1.982395614047245, -0.05691616561213509, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler212 failed\n");
        errorCount++;
    }
    C2Euler213(C, v3_1);
    v3Set(1.157420789791818, 1.155503238813826, -3.012011225795042, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler213 failed\n");
        errorCount++;
    }
    C2Euler231(C, v3_1);
    v3Set(-2.102846464319881, -0.05215813778076988, 1.982990154077466, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler231 failed\n");
        errorCount++;
    }
    C2Euler232(C, v3_1);
    v3Set(-0.5548415771975691, 1.982395614047245, -1.627712492407032, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler232 failed\n");
        errorCount++;
    }
    C2Euler312(C, v3_1);
    v3Set(2.045248068737305, -0.5038614866151004, -1.384653359078797, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler312 failed\n");
        errorCount++;
    }
    C2Euler313(C, v3_1);
    v3Set(0.3837766626244829, 1.408008028147626, 2.082059614484753, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler313 failed\n");
        errorCount++;
    }
    C2Euler321(C, v3_1);
    v3Set(-3.039045355374235, -1.036440549977791, -1.246934586231547, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler321 failed\n");
        errorCount++;
    }
    C2Euler323(C, v3_1);
    v3Set(-1.187019664170414, 1.408008028147626, -2.630329365899936, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Euler323 failed\n");
        errorCount++;
    }

    v3Set(0.25, 0.5, -0.5, v3_1);
    C2Gibbs(C, v3_1);
    v3Set(1.692307692307693, -1.923076923076923, 2.846153846153846, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2Gibbs failed\n");
        errorCount++;
    }
    C2MRP(C, v3_1);
    v3Set(0.3413551595269481, -0.3879035903715318, 0.5740973137498672, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2MRP failed\n");
        errorCount++;
    }
    C2PRV(C, v3_1);
    v3Set(1.162634795241009, -1.321175903682964, 1.955340337450788, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2PRV failed\n");
        errorCount++;
    }
    m33SetIdentity(C);
    C2PRV(C, v3_1);
    v3Set(0.0, 0.0, 0.0, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2PRV failed\n");
        errorCount++;
    }
    m33SetIdentity(C);
    C[0][0] = -1.0;
    C[1][1] = -1.0;
    C2PRV(C, v3_1);
    v3Set(0.0, 0.0, M_PI, v3_2);
    if(!v3IsEqual(v3_1, v3_2, accuracy)) {
        printf("C2PRV failed\n");
        errorCount++;
    }

    v4Set(0.2526773896521122, 0.4276078901804977, -0.4859180570232927, 0.7191587243944733, v3_1);
    v3Set(0.2, 0.1, -0.5, om);
    dEP(v3_1, om, v3);
    v4Set(0.1613247949317332, 0.1107893170013107, 0.1914517144671774, 0.006802852798326098, v3_1);
    if(!vIsEqual(v3_1, 4, v3, accuracy)) {
        printf("dEP failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, -40 * D2R, 15 * D2R, v3_1);
    v3Set(0.2, 0.1, -0.5, om);
    dEuler121(v3_1, om, v3);
    v3Set(0.7110918159377425, 0.2260021051801672, -0.3447279341464908, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler121 failed\n");
        errorCount++;
    }
    dEuler123(v3_1, om, v3);
    v3Set(0.2183988961089258, 0.148356391649411, -0.3596158956119647, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler123 failed\n");
        errorCount++;
    }
    dEuler131(v3_1, om, v3);
    v3Set(0.3515968599493992, -0.4570810086342821, -0.06933882078231876, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler131 failed\n");
        errorCount++;
    }
    dEuler132(v3_1, om, v3);
    v3Set(0.08325318887098565, -0.5347267221650382, 0.04648588172683711, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler132 failed\n");
        errorCount++;
    }
    dEuler212(v3_1, om, v3);
    v3Set(-0.8318871025311179, 0.06377564270655334, 0.7372624921963103, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler212 failed\n");
        errorCount++;
    }
    dEuler213(v3_1, om, v3);
    v3Set(0.1936655150781755, 0.1673032607475616, -0.6244857935158128, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler213 failed\n");
        errorCount++;
    }
    dEuler231(v3_1, om, v3);
    v3Set(0.2950247955066306, -0.4570810086342821, 0.3896382831019671, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler231 failed\n");
        errorCount++;
    }
    dEuler232(v3_1, om, v3);
    v3Set(-0.09921728693192147, -0.5347267221650384, 0.1760048513155397, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler232 failed\n");
        errorCount++;
    }
    dEuler312(v3_1, om, v3);
    v3Set(-0.6980361609149971, 0.06377564270655331, -0.3486889953493196, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler312 failed\n");
        errorCount++;
    }
    dEuler313(v3_1, om, v3);
    v3Set(-0.2308015733560238, 0.1673032607475616, -0.3231957372675008, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler312 failed\n");
        errorCount++;
    }
    dEuler321(v3_1, om, v3);
    v3Set(-0.596676880486542, 0.2260021051801672, 0.5835365057631652, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler312 failed\n");
        errorCount++;
    }
    dEuler323(v3_1, om, v3);
    v3Set(0.260277669056422, 0.148356391649411, -0.6993842620486324, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dEuler312 failed\n");
        errorCount++;
    }

    dGibbs(v3_1, om, v3);
    v3Set(0.236312018677072, 0.2405875488560276, -0.1665723597065136, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dGibbs failed\n");
        errorCount++;
    }
    dMRP(v3_1, om, v3);
    v3Set(0.144807895231133, 0.1948354871330581, 0.062187948908334, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dMRP failed\n");
        errorCount++;
    }
    dPRV(v3_1, om, v3);
    v3Set(0.34316538031149, 0.255728121815202, -0.3710557691157747, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("dPRV failed\n");
        errorCount++;
    }
    
    double w[3];
    v3Set(0.0124791041517595, 0.0042760566673861, -0.0043633231299858, v3_1);
    dMRP2Omega(om, v3_1, v3);
    v3Set(0.0174532925199433, 0.0349065850398866, -0.0174532925199433, w);
    if(!v3IsEqual(v3, w, accuracy)) {
        printf("dMRP2Omega failed\n");
        errorCount++;
    }
    
    double dw[3];
    v3Set(0.0022991473184427, 0.0035194052312667, -0.0070466757773158, dw);
    ddMRP(om, v3_1, w, dw, v3);
    v3Set(0.0015, 0.0010, -0.0020, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("ddMRP failed\n");
        errorCount++;
    }

    ddMRP2dOmega(om, v3_1, v3_2, v3);
    if(!v3IsEqual(v3, dw, accuracy)) {
        printf("ddMRP2dOmega failed\n");
        errorCount++;
    }

    v4Set(0.9110886174894189, 0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    elem2PRV(v3_1, v3);
    v3Set(0.5235987755982988, -0.6981317007977318, 0.2617993877991494, v3_2);
    if(!v3IsEqual(v3, v3_2, accuracy)) {
        printf("elem2PRV failed\n");
        errorCount++;
    }

    v4Set(0.2526773896521122, 0.4276078901804977, -0.4859180570232927, 0.7191587243944733, v3_1);
    EP2C(v3_1, C);
    v3Set(-0.506611258027956, -0.05213449187759728, 0.860596902153381, C2[0]);
    v3Set(-0.7789950887797505, -0.4000755572346052, -0.4828107291273137, C2[1]);
    v3Set(0.3694748772194938, -0.9149981110691346, 0.1620702682281828, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("EP2C failed\n");
        errorCount++;
    }
    EP2Euler121(v3_1, v3_2);
    v3Set(3.202098165750965, 2.102046098550739, -1.127921895439695, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler121 failed\n");
        errorCount++;
    }
    EP2Euler123(v3_1, v3_2);
    v3Set(1.395488250243478, 0.3784438476398376, 2.147410157986089, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler123 failed\n");
        errorCount++;
    }
    EP2Euler131(v3_1, v3_2);
    v3Set(1.631301838956069, 2.102046098550739, 0.4428744313552013, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler131 failed\n");
        errorCount++;
    }
    EP2Euler132(v3_1, v3_2);
    v3Set(-2.262757475208626, 0.8930615653924096, 2.511467464302149, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler132 failed\n");
        errorCount++;
    }
    EP2Euler212(v3_1, v3_2);
    v3Set(-2.125637903992466, 1.982395614047245, -0.05691616561213508, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler212 failed\n");
        errorCount++;
    }
    EP2Euler213(v3_1, v3_2);
    v3Set(1.157420789791818, 1.155503238813826, -3.012011225795042, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler213 failed\n");
        errorCount++;
    }
    EP2Euler231(v3_1, v3_2);
    v3Set(-2.102846464319881, -0.05215813778076988, 1.982990154077466, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler231 failed\n");
        errorCount++;
    }
    EP2Euler232(v3_1, v3_2);
    v3Set(-0.5548415771975691, 1.982395614047245, -1.627712492407032, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler232 failed\n");
        errorCount++;
    }
    EP2Euler312(v3_1, v3_2);
    v3Set(2.045248068737305, -0.5038614866151004, -1.384653359078797, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler312 failed\n");
        errorCount++;
    }
    EP2Euler313(v3_1, v3_2);
    v3Set(0.3837766626244828, 1.408008028147627, 2.082059614484753, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler313 failed\n");
        errorCount++;
    }
    EP2Euler321(v3_1, v3_2);
    v3Set(-3.039045355374235, -1.036440549977791, -1.246934586231547, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler321 failed\n");
        errorCount++;
    }
    EP2Euler323(v3_1, v3_2);
    v3Set(-1.187019664170414, 1.408008028147627, 3.65285594127965, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Euler323 failed\n");
        errorCount++;
    }
    EP2Gibbs(v3_1, v3_2);
    v3Set(1.692307692307693, -1.923076923076923, 2.846153846153846, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2Gibbs failed\n");
        errorCount++;
    }
    EP2MRP(v3_1, v3_2);
    v3Set(0.3413551595269481, -0.3879035903715319, 0.5740973137498672, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2MRP failed\n");
        errorCount++;
    }
    EP2PRV(v3_1, v3_2);
    v3Set(1.162634795241009, -1.321175903682965, 1.955340337450788, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2PRV failed\n");
        errorCount++;
    }

    v4Set(1.0, 0.0, 0.0, 0.0, v3_1);
    EP2PRV(v3_1, v3_2);
    v3Set(0.0, 0.0, 0.0, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2PRV failed\n");
        errorCount++;
    }
    v4Set(-1.0, 0.0, 0.0, 0.0, v3_1);
    EP2PRV(v3_1, v3_2);
    v3Set(0.0, 0.0, 0.0, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2PRV failed\n");
        errorCount++;
    }
    v4Set(0.0, 1.0, 0.0, 0.0, v3_1);
    EP2PRV(v3_1, v3_2);
    v3Set(M_PI, 0.0, 0.0, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("EP2PRV failed\n");
        errorCount++;
    }



    Euler1(1.3, C);
    v3Set(1, 0, 0, C2[0]);
    v3Set(0, 0.2674988286245874, 0.963558185417193, C2[1]);
    v3Set(0, -0.963558185417193, 0.2674988286245874, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler1 failed\n");
        errorCount++;
    }
    Euler2(1.3, C);
    v3Set(0.2674988286245874, 0, -0.963558185417193, C2[0]);
    v3Set(0, 1, 0, C2[1]);
    v3Set(0.963558185417193, 0, 0.2674988286245874, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler2 failed\n");
        errorCount++;
    }
    Euler3(1.3, C);
    v3Set(0.2674988286245874, 0.963558185417193, 0, C2[0]);
    v3Set(-0.963558185417193, 0.2674988286245874, 0, C2[1]);
    v3Set(0, 0, 1, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler3 failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler1212C(v3_1, C);
    v3Set(0.7205084754311385, -0.3769430728235922, 0.5820493593177511, C2[0]);
    v3Set(-0.1965294640304305, 0.6939446195986547, 0.692688266609151, C2[1]);
    v3Set(-0.6650140649638986, -0.6134776155495705, 0.4259125598286639, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler1212C failed\n");
        errorCount++;
    }
    Euler1212EP(v3_1, v3_2);
    v4Set(0.8426692196316502, 0.3875084824890354, -0.3699741829975614, -0.05352444488005169, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler1212EP failed\n");
        errorCount++;
    }
    Euler1212Gibbs(v3_1, v3_2);
    v3Set(0.4598583565902931, -0.4390503110571495, -0.06351774057138154, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1212Gibbs failed\n");
        errorCount++;
    }
    Euler1212MRP(v3_1, v3_2);
    v3Set(0.2102973655610845, -0.2007816590497557, -0.02904723447366817, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1212MRP failed\n");
        errorCount++;
    }
    Euler1212PRV(v3_1, v3_2);
    v3Set(0.8184049632304388, -0.7813731087574279, -0.1130418386266624, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1212PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler1232C(v3_1, C);
    v3Set(0.6909668228739537, -0.1236057418710468, 0.7122404581768593, C2[0]);
    v3Set(-0.2041991989591971, 0.9117724894309838, 0.3563335721781613, C2[1]);
    v3Set(-0.6934461311680212, -0.391653607277317, 0.6047643467291773, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler1232C failed\n");
        errorCount++;
    }
    Euler1232EP(v3_1, v3_2);
    v4Set(0.8954752451958283, 0.2088240806958052, -0.3924414987701519, 0.02250019124496444, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler1232EP failed\n");
        errorCount++;
    }
    Euler1232Gibbs(v3_1, v3_2);
    v3Set(0.2331991663824702, -0.4382494109977661, 0.02512653628972619, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1232Gibbs failed\n");
        errorCount++;
    }
    Euler1232MRP(v3_1, v3_2);
    v3Set(0.1101697746911123, -0.2070412155288303, 0.01187047485953311, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1232MRP failed\n");
        errorCount++;
    }
    Euler1232PRV(v3_1, v3_2);
    v3Set(0.4328366663508259, -0.8134266388215754, 0.04663690000825693, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1232PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler1312C(v3_1, C);
    v3Set(0.7205084754311385, -0.5820493593177511, -0.3769430728235922, C2[0]);
    v3Set(0.6650140649638986, 0.4259125598286639, 0.6134776155495705, C2[1]);
    v3Set(-0.1965294640304305, -0.692688266609151, 0.6939446195986547, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler1312C failed\n");
        errorCount++;
    }
    Euler1312EP(v3_1, v3_2);
    v4Set(0.8426692196316502, 0.3875084824890354, 0.05352444488005169, -0.3699741829975614, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler1312EP failed\n");
        errorCount++;
    }
    Euler1312Gibbs(v3_1, v3_2);
    v3Set(0.4598583565902931, 0.06351774057138154, -0.4390503110571495, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1312Gibbs failed\n");
        errorCount++;
    }
    Euler1312MRP(v3_1, v3_2);
    v3Set(0.2102973655610845, 0.02904723447366817, -0.2007816590497557, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1312MRP failed\n");
        errorCount++;
    }
    Euler1312PRV(v3_1, v3_2);
    v3Set(0.8184049632304388, 0.1130418386266624, -0.7813731087574279, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1312PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler1322C(v3_1, C);
    v3Set(0.6909668228739537, -0.404128912281835, -0.5993702294453531, C2[0]);
    v3Set(0.6934461311680212, 0.6047643467291773, 0.391653607277317, C2[1]);
    v3Set(0.2041991989591971, -0.6862506154337003, 0.6981137299618809, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler1322C failed\n");
        errorCount++;
    }
    Euler1322EP(v3_1, v3_2);
    v4Set(0.8651365354042408, 0.3114838463640192, 0.2322088466732818, -0.3171681574333834, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler1322EP failed\n");
        errorCount++;
    }
    Euler1322Gibbs(v3_1, v3_2);
    v3Set(0.3600401018996109, 0.2684071671586273, -0.3666105226791566, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1322Gibbs failed\n");
        errorCount++;
    }
    Euler1322MRP(v3_1, v3_2);
    v3Set(0.1670032410235906, 0.1244996504360223, -0.1700509058789317, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1322MRP failed\n");
        errorCount++;
    }
    Euler1322PRV(v3_1, v3_2);
    v3Set(0.6525765328552258, 0.4864908592507521, -0.6644854907437873, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler1322PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler2122C(v3_1, C);
    v3Set(0.6939446195986547, -0.1965294640304305, -0.692688266609151, C2[0]);
    v3Set(-0.3769430728235922, 0.7205084754311385, -0.5820493593177511, C2[1]);
    v3Set(0.6134776155495705, 0.6650140649638986, 0.4259125598286639, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler2122C failed\n");
        errorCount++;
    }
    Euler2122EP(v3_1, v3_2);
    v4Set(0.8426692196316502, -0.3699741829975614, 0.3875084824890354, 0.05352444488005169, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler2122EP failed\n");
        errorCount++;
    }
    Euler2122Gibbs(v3_1, v3_2);
    v3Set(-0.4390503110571495, 0.4598583565902931, 0.06351774057138154, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2122Gibbs failed\n");
        errorCount++;
    }
    Euler2122MRP(v3_1, v3_2);
    v3Set(-0.2007816590497557, 0.2102973655610845, 0.02904723447366817, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2122MRP failed\n");
        errorCount++;
    }
    Euler2122PRV(v3_1, v3_2);
    v3Set(-0.7813731087574279, 0.8184049632304388, 0.1130418386266624, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2122PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler2132C(v3_1, C);
    v3Set(0.6981137299618809, 0.2041991989591971, -0.6862506154337003, C2[0]);
    v3Set(-0.5993702294453531, 0.6909668228739537, -0.404128912281835, C2[1]);
    v3Set(0.391653607277317, 0.6934461311680212, 0.6047643467291773, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler2132C failed\n");
        errorCount++;
    }
    Euler2132EP(v3_1, v3_2);
    v4Set(0.8651365354042408, -0.3171681574333834, 0.3114838463640192, 0.2322088466732818, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler2132EP failed\n");
        errorCount++;
    }
    Euler2132Gibbs(v3_1, v3_2);
    v3Set(-0.3666105226791566, 0.3600401018996109, 0.2684071671586273, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2132Gibbs failed\n");
        errorCount++;
    }
    Euler2132MRP(v3_1, v3_2);
    v3Set(-0.1700509058789317, 0.1670032410235906, 0.1244996504360223, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2132MRP failed\n");
        errorCount++;
    }
    Euler2132PRV(v3_1, v3_2);
    v3Set(-0.6644854907437873, 0.6525765328552258, 0.4864908592507521, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2132PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler2312C(v3_1, C);
    v3Set(0.6047643467291773, -0.6934461311680212, -0.391653607277317, C2[0]);
    v3Set(0.7122404581768593, 0.6909668228739537, -0.1236057418710468, C2[1]);
    v3Set(0.3563335721781613, -0.2041991989591971, 0.9117724894309838, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler2312C failed\n");
        errorCount++;
    }
    Euler2312EP(v3_1, v3_2);
    v4Set(0.8954752451958283, 0.02250019124496444, 0.2088240806958052, -0.3924414987701519, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler2312EP failed\n");
        errorCount++;
    }
    Euler2312Gibbs(v3_1, v3_2);
    v3Set(0.02512653628972619, 0.2331991663824702, -0.4382494109977661, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2312Gibbs failed\n");
        errorCount++;
    }
    Euler2312MRP(v3_1, v3_2);
    v3Set(0.01187047485953311, 0.1101697746911123, -0.2070412155288303, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2312MRP failed\n");
        errorCount++;
    }
    Euler2312PRV(v3_1, v3_2);
    v3Set(0.04663690000825693, 0.4328366663508259, -0.8134266388215754, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2312PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler2322C(v3_1, C);
    v3Set(0.4259125598286639, -0.6650140649638986, -0.6134776155495705, C2[0]);
    v3Set(0.5820493593177511, 0.7205084754311385, -0.3769430728235922, C2[1]);
    v3Set(0.692688266609151, -0.1965294640304305, 0.6939446195986547, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler2322C failed\n");
        errorCount++;
    }
    Euler2322EP(v3_1, v3_2);
    v4Set(0.8426692196316502, -0.05352444488005169, 0.3875084824890354, -0.3699741829975614, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler2322EP failed\n");
        errorCount++;
    }
    Euler2322Gibbs(v3_1, v3_2);
    v3Set(-0.06351774057138154, 0.4598583565902931, -0.4390503110571495, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2322Gibbs failed\n");
        errorCount++;
    }
    Euler2322MRP(v3_1, v3_2);
    v3Set(-0.02904723447366817, 0.2102973655610845, -0.2007816590497557, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2322MRP failed\n");
        errorCount++;
    }
    Euler2322PRV(v3_1, v3_2);
    v3Set(-0.1130418386266624, 0.8184049632304388, -0.7813731087574279, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler2322PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler3122C(v3_1, C);
    v3Set(0.9117724894309838, 0.3563335721781613, -0.2041991989591971, C2[0]);
    v3Set(-0.391653607277317, 0.6047643467291773, -0.6934461311680212, C2[1]);
    v3Set(-0.1236057418710468, 0.7122404581768593, 0.6909668228739537, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler3122C failed\n");
        errorCount++;
    }
    Euler3122EP(v3_1, v3_2);
    v4Set(0.8954752451958283, -0.3924414987701519, 0.02250019124496444, 0.2088240806958052, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler3122EP failed\n");
        errorCount++;
    }
    Euler3122Gibbs(v3_1, v3_2);
    v3Set(-0.4382494109977661, 0.02512653628972619, 0.2331991663824702, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3122Gibbs failed\n");
        errorCount++;
    }
    Euler3122MRP(v3_1, v3_2);
    v3Set(-0.2070412155288303, 0.01187047485953311, 0.1101697746911123, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3122MRP failed\n");
        errorCount++;
    }
    Euler3122PRV(v3_1, v3_2);
    v3Set(-0.8134266388215754, 0.04663690000825693, 0.4328366663508259, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3122PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler3132C(v3_1, C);
    v3Set(0.6939446195986547, 0.692688266609151, -0.1965294640304305, C2[0]);
    v3Set(-0.6134776155495705, 0.4259125598286639, -0.6650140649638986, C2[1]);
    v3Set(-0.3769430728235922, 0.5820493593177511, 0.7205084754311385, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler3132C failed\n");
        errorCount++;
    }
    Euler3132EP(v3_1, v3_2);
    v4Set(0.8426692196316502, -0.3699741829975614, -0.05352444488005169, 0.3875084824890354, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler3132EP failed\n");
        errorCount++;
    }
    Euler3132Gibbs(v3_1, v3_2);
    v3Set(-0.4390503110571495, -0.06351774057138154, 0.4598583565902931, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3132Gibbs failed\n");
        errorCount++;
    }
    Euler3132MRP(v3_1, v3_2);
    v3Set(-0.2007816590497557, -0.02904723447366817, 0.2102973655610845, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3132MRP failed\n");
        errorCount++;
    }
    Euler3132PRV(v3_1, v3_2);
    v3Set(-0.7813731087574279, -0.1130418386266624, 0.8184049632304388, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3132PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler3212C(v3_1, C);
    v3Set(0.6047643467291773, 0.391653607277317, 0.6934461311680212, C2[0]);
    v3Set(-0.6862506154337003, 0.6981137299618809, 0.2041991989591971, C2[1]);
    v3Set(-0.404128912281835, -0.5993702294453531, 0.6909668228739537, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler3212C failed\n");
        errorCount++;
    }
    Euler3212EP(v3_1, v3_2);
    v4Set(0.8651365354042408, 0.2322088466732818, -0.3171681574333834, 0.3114838463640192, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler3212EP failed\n");
        errorCount++;
    }
    Euler3212Gibbs(v3_1, v3_2);
    v3Set(0.2684071671586273, -0.3666105226791566, 0.3600401018996109, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3212Gibbs failed\n");
        errorCount++;
    }
    Euler3212MRP(v3_1, v3_2);
    v3Set(0.1244996504360223, -0.1700509058789317, 0.1670032410235906, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3212MRP failed\n");
        errorCount++;
    }
    Euler3212PRV(v3_1, v3_2);
    v3Set(0.4864908592507521, -0.6644854907437873, 0.6525765328552258, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3212PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Euler3232C(v3_1, C);
    v3Set(0.4259125598286639, 0.6134776155495705, 0.6650140649638986, C2[0]);
    v3Set(-0.692688266609151, 0.6939446195986547, -0.1965294640304305, C2[1]);
    v3Set(-0.5820493593177511, -0.3769430728235922, 0.7205084754311385, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler3232C failed\n");
        errorCount++;
    }
    Euler3232EP(v3_1, v3_2);
    v4Set(0.8426692196316502, 0.05352444488005169, -0.3699741829975614, 0.3875084824890354, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Euler3232EP failed\n");
        errorCount++;
    }
    Euler3232Gibbs(v3_1, v3_2);
    v3Set(0.06351774057138154, -0.4390503110571495, 0.4598583565902931, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3232Gibbs failed\n");
        errorCount++;
    }
    Euler3232MRP(v3_1, v3_2);
    v3Set(0.02904723447366817, -0.2007816590497557, 0.2102973655610845, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3232MRP failed\n");
        errorCount++;
    }
    Euler3232PRV(v3_1, v3_2);
    v3Set(0.1130418386266624, -0.7813731087574279, 0.8184049632304388, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Euler3232PRV failed\n");
        errorCount++;
    }

    v3Set(0.5746957711326909, -0.7662610281769212, 0.2873478855663454, v3_1);
    Gibbs2C(v3_1, C);
    v3Set(0.3302752293577981, -0.1530190869107189, 0.9313986428558203, C2[0]);
    v3Set(-0.7277148580434096, 0.5871559633027522, 0.3545122848941588, C2[1]);
    v3Set(-0.6011234134980221, -0.794879257371223, 0.08256880733944938, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Gibbs2C failed\n");
        errorCount++;
    }
    Gibbs2EP(v3_1, v3_2);
    v4Set(0.7071067811865475, 0.4063712768871578, -0.5418283691828771, 0.2031856384435789, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("Gibbs2EP failed\n");
        errorCount++;
    }
    Gibbs2Euler121(v3_1, v3_2);
    v3Set(3.304427597008361, 1.234201174364066, -2.26121636963008, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler121 failed\n");
        errorCount++;
    }
    Gibbs2Euler123(v3_1, v3_2);
    v3Set(1.467291629150036, -0.6449061163953342, 1.144743256726005, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler123 failed\n");
        errorCount++;
    }
    Gibbs2Euler131(v3_1, v3_2);
    v3Set(1.733631270213465, 1.234201174364066, -0.6904200428351842, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler131 failed\n");
        errorCount++;
    }
    Gibbs2Euler132(v3_1, v3_2);
    v3Set(0.54319335066115, 0.8149843403384446, -1.068390851022488, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler132 failed\n");
        errorCount++;
    }
    Gibbs2Euler212(v3_1, v3_2);
    v3Set(-1.117474807766432, 0.9432554204540935, -0.1901795897648197, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler212 failed\n");
        errorCount++;
    }
    Gibbs2Euler213(v3_1, v3_2);
    v3Set(-1.434293025994105, 0.9188085603647974, -0.2549399408440935, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler213 failed\n");
        errorCount++;
    }
    Gibbs2Euler231(v3_1, v3_2);
    v3Set(-1.230028192223063, -0.1536226209659692, 0.9345839026955233, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler231 failed\n");
        errorCount++;
    }
    Gibbs2Euler232(v3_1, v3_2);
    v3Set(0.4533215190284649, 0.9432554204540935, -1.760975916559716, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler232 failed\n");
        errorCount++;
    }
    Gibbs2Euler312(v3_1, v3_2);
    v3Set(0.8918931304028546, 0.3623924238788913, -1.482377127697951, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler312 failed\n");
        errorCount++;
    }
    Gibbs2Euler313(v3_1, v3_2);
    v3Set(-0.6474859022891233, 1.488133410155628, 1.207104533714101, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler313 failed\n");
        errorCount++;
    }
    Gibbs2Euler321(v3_1, v3_2);
    v3Set(-0.4338654111289937, -1.198236565236741, 1.341967642658489, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler321 failed\n");
        errorCount++;
    }
    Gibbs2Euler323(v3_1, v3_2);
    v3Set(-2.21828222908402, 1.488133410155628, 2.777900860508998, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2Euler321 failed\n");
        errorCount++;
    }
    Gibbs2MRP(v3_1, v3_2);
    v3Set(0.2380467826416248, -0.3173957101888331, 0.1190233913208124, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2MRP failed\n");
        errorCount++;
    }
    Gibbs2PRV(v3_1, v3_2);
    v3Set(0.9027300063197914, -1.203640008426389, 0.4513650031598956, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2PRV failed\n");
        errorCount++;
    }
    v3Set(0.0, 0.0, 0.0, v3_1);
    Gibbs2PRV(v3_1, v3_2);
    v3Set(0.0, 0.0, 0.0, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("Gibbs2PRV failed\n");
        errorCount++;
    }

    v3Set(0.2, -0.25, 0.3, v3_1);
    MRP2C(v3_1, C);
    v3Set(0.1420873822677549, 0.4001248192538094, 0.9053790945330048, C2[0]);
    v3Set(-0.9626904702257736, 0.2686646537364468, 0.03234752493088797, C2[1]);
    v3Set(-0.2303003133666478, -0.876196001388834, 0.4233702077537369, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("MRP2C failed\n");
        errorCount++;
    }
    MRP2EP(v3_1, v3_2);
    v4Set(0.6771488469601677, 0.3354297693920336, -0.419287211740042, 0.5031446540880503, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("MRP2EP failed\n");
        errorCount++;
    }
    MRP2Euler121(v3_1, v3_2);
    v3Set(2.725460144813494, 1.428226451915784, -1.805609061169705, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler121 failed\n");
        errorCount++;
    }
    MRP2Euler123(v3_1, v3_2);
    v3Set(1.120685944613971, -0.2323862804943196, 1.424260216144192, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler123 failed\n");
        errorCount++;
    }
    MRP2Euler131(v3_1, v3_2);
    v3Set(1.154663818018597, 1.428226451915784, -0.2348127343748092, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler131 failed\n");
        errorCount++;
    }
    MRP2Euler132(v3_1, v3_2);
    v3Set(0.1198243320629901, 1.296774918090265, -1.017995395279125, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler132 failed\n");
        errorCount++;
    }
    MRP2Euler212(v3_1, v3_2);
    v3Set(-1.537207795170527, 1.298789879764913, 0.4283796513241308, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler212 failed\n");
        errorCount++;
    }
    MRP2Euler213(v3_1, v3_2);
    v3Set(-0.4982011776145131, 1.067911809027856, 0.979488037955722, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler213 failed\n");
        errorCount++;
    }
    MRP2Euler231(v3_1, v3_2);
    v3Set(-1.415129132201094, 0.4116530390866675, 1.273271587093173, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler231 failed\n");
        errorCount++;
    }
    MRP2Euler232(v3_1, v3_2);
    v3Set(0.03358853162436948, 1.298789879764913, -1.142416675470766, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler232 failed\n");
        errorCount++;
    }
    MRP2Euler312(v3_1, v3_2);
    v3Set(1.298643836753137, 0.03235316879424937, -1.133389474325039, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler312 failed\n");
        errorCount++;
    }
    MRP2Euler313(v3_1, v3_2);
    v3Set(-0.257027406977469, 1.133634172515794, 1.535083362165219, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler313 failed\n");
        errorCount++;
    }
    MRP2Euler321(v3_1, v3_2);
    v3Set(1.22957853325386, -1.13227169191098, 0.0762566635156139, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler321 failed\n");
        errorCount++;
    }
    MRP2Euler323(v3_1, v3_2);
    v3Set(-1.827823733772366, 1.133634172515794, 3.105879688960115, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Euler321 failed\n");
        errorCount++;
    }
    MRP2Gibbs(v3_1, v3_2);
    v3Set(0.4953560371517029, -0.6191950464396285, 0.7430340557275542, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2Gibbs failed\n");
        errorCount++;
    }
    MRP2PRV(v3_1, v3_2);
    v3Set(0.7538859486650076, -0.9423574358312593, 1.130828922997511, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2PRV failed\n");
        errorCount++;
    }

    MRPswitch(v3_1, 1, v3_2);
    v3Set(0.2, -0.25, 0.3, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2PRV failed\n");
        errorCount++;
    }
    MRPswitch(v3_1, 0.4, v3_2);
    v3Set(-1.038961038961039, 1.298701298701299, -1.558441558441558, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2PRV failed\n");
        errorCount++;
    }
    v3Set(0.0 ,0.0, 0.0, v3_1);
    MRP2PRV(v3_1, v3_2);
    v3Set(0.0, 0.0, 0.0, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2PRV failed\n");
        errorCount++;
    }
    v3Set(1.0 ,0.0, 0.0, v3_1);
    MRP2PRV(v3_1, v3_2);
    v3Set(M_PI, 0.0, 0.0, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("MRP2PRV failed\n");
        errorCount++;
    }


    if(!isEqual(wrapToPi(1.2), 1.2, accuracy)) {
        printf("wrapToPi(1.2) failed\n");
        errorCount++;
    }
    if(!isEqual(wrapToPi(4.2), -2.083185307179586, accuracy)) {
        printf("wrapToPi(4.2) failed\n");
        errorCount++;
    }
    if(!isEqual(wrapToPi(-4.2), 2.083185307179586, accuracy)) {
        printf("wrapToPi(-4.2) failed\n");
        errorCount++;
    }

    v3Set(0.2, -0.25, 0.3, v3_1);
    PRV2C(v3_1, C);
    v3Set(0.9249653552860658, 0.2658656942983466, 0.2715778417245783, C2[0]);
    v3Set(-0.3150687400124018, 0.9360360405717283, 0.1567425271513747, C2[1]);
    v3Set(-0.212534186867712, -0.2305470957224576, 0.9495668781430935, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("PRV2C failed\n");
        errorCount++;
    }
    PRV2EP(v3_1, v3_2);
    v4Set(0.9760338459808767, 0.09919984446969178, -0.1239998055871147, 0.1487997667045377, v3);
    if(!vIsEqual(v3_2, 4, v3, accuracy)) {
        printf("PRV2EP failed\n");
        errorCount++;
    }
    PRV2Euler121(v3_1, v3_2);
    v3Set(2.366822457545908, 0.3898519008736288, -2.164246748437291, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler121 failed\n");
        errorCount++;
    }
    PRV2Euler123(v3_1, v3_2);
    v3Set(0.2381830975647435, -0.2141676691157164, 0.3283009769818029, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler123 failed\n");
        errorCount++;
    }
    PRV2Euler131(v3_1, v3_2);
    v3Set(0.796026130751012, 0.3898519008736288, -0.5934504216423945, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler131 failed\n");
        errorCount++;
    }
    PRV2Euler132(v3_1, v3_2);
    v3Set(0.1659141638227202, 0.3205290820781828, -0.2258549616703266, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler132 failed\n");
        errorCount++;
    }
    PRV2Euler212(v3_1, v3_2);
    v3Set(-1.109161329065078, 0.3596045976550934, 0.8564261174295806, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler212 failed\n");
        errorCount++;
    }
    PRV2Euler213(v3_1, v3_2);
    v3Set(-0.2201931522496843, 0.2326398873102022, 0.2767451364802878, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler213 failed\n");
        errorCount++;
    }
    PRV2Euler231(v3_1, v3_2);
    v3Set(-0.2855829177825758, 0.269101825006778, 0.2414947191533679, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler231 failed\n");
        errorCount++;
    }
    PRV2Euler232(v3_1, v3_2);
    v3Set(0.4616349977298192, 0.3596045976550934, -0.714370209365316, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler232 failed\n");
        errorCount++;
    }
    PRV2Euler312(v3_1, v3_2);
    v3Set(0.3246867163622526, 0.1573915425330904, -0.2785654591200913, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler312 failed\n");
        errorCount++;
    }
    PRV2Euler313(v3_1, v3_2);
    v3Set(-0.7447668031423726, 0.3189446151924337, 1.047343966000315, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler313 failed\n");
        errorCount++;
    }
    PRV2Euler321(v3_1, v3_2);
    v3Set(0.2798880637473677, -0.2750321114914171, 0.1635922230133545, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler321 failed\n");
        errorCount++;
    }
    PRV2Euler323(v3_1, v3_2);
    v3Set(-2.315563129937269, 0.3189446151924337, 2.618140292795212, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Euler321 failed\n");
        errorCount++;
    }
    PRV2Gibbs(v3_1, v3_2);
    v3Set(0.1016356603597079, -0.1270445754496348, 0.1524534905395618, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2Gibbs failed\n");
        errorCount++;
    }
    PRV2MRP(v3_1, v3_2);
    v3Set(0.05020149056224809, -0.06275186320281011, 0.07530223584337212, v3);
    if(!v3IsEqual(v3_2, v3, accuracy)) {
        printf("PRV2MRP failed\n");
        errorCount++;
    }

    v4Set(0.45226701686665, 0.75377836144441, 0.15075567228888, 0.45226701686665, v3_1);
    v4Set(-0.18663083698528, 0.46657709246321, 0.83983876643378, -0.20529392068381, v3_2);
    subEP(v3_1, v3_2, v3);
    v4Set(0.3010515331052196, -0.762476312817895, -0.0422034859493331, 0.5711538431809339, v3_1);
    if(!vIsEqual(v3, 4, v3_1, accuracy)) {
        printf("subEP failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler121(v3_1, v3_2, v3);
    v3Set(2.969124082346242, 2.907100217278789, 2.423943306316236, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler121 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler123(v3_1, v3_2, v3);
    v3Set(3.116108453572625, -0.6539785291371149, -0.9652248604105184, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler123 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler131(v3_1, v3_2, v3);
    v3Set(2.969124082346242, 2.907100217278789, 2.423943306316236, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler131 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler132(v3_1, v3_2, v3);
    v3Set(2.932019083757663, 0.6246626379494424, -1.519867235625338, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler132 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler212(v3_1, v3_2, v3);
    v3Set(2.969124082346242, 2.907100217278789, 2.423943306316236, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler212 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler213(v3_1, v3_2, v3);
    v3Set(2.932019083757663, 0.6246626379494424, -1.519867235625338, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler213 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler231(v3_1, v3_2, v3);
    v3Set(3.116108453572625, -0.6539785291371149, -0.9652248604105185, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler231 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler232(v3_1, v3_2, v3);
    v3Set(2.969124082346242, 2.907100217278789, 2.423943306316236, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler232 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler312(v3_1, v3_2, v3);
    v3Set(3.116108453572625, -0.653978529137115, -0.9652248604105184, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler312 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler313(v3_1, v3_2, v3);
    v3Set(2.969124082346242, 2.907100217278789, 2.423943306316236, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler313 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler321(v3_1, v3_2, v3);
    v3Set(2.932019083757663, 0.6246626379494424, -1.519867235625338, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler321 failed\n");
        errorCount++;
    }

    v3Set(10 * D2R, 20 * D2R, 30 * D2R, v3_1);
    v3Set(-30 * D2R, 200 * D2R, 81 * D2R, v3_2);
    subEuler323(v3_1, v3_2, v3);
    v3Set(2.969124082346242, 2.907100217278789, 2.423943306316236, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subEuler323 failed\n");
        errorCount++;
    }

    v3Set(1.5, 0.5, 0.5, v3_1);
    v3Set(-0.5, 0.25, 0.15, v3_2);
    subGibbs(v3_1, v3_2, v3);
    v3Set(4.333333333333333, -0.5, 2.166666666666667, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subGibbs failed\n");
        errorCount++;
    }

    v3Set(1.5, 0.5, 0.5, v3_1);
    v3Set(-0.5, 0.25, 0.15, v3_2);
    subMRP(v3_1, v3_2, v3);
    v3Set(-0.005376344086021518,0.04301075268817203,-0.4408602150537635, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subMRP failed\n");
        errorCount++;
    }

    v3Set(0.0, 0.0, 1.0, v3_1);
    v3Set(0.0, 0.0, -1.0, v3_2);
    subMRP(v3_1, v3_2, v3);
    v3Set(0.0,0.0,0.0, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subMRP 360 subtraction failed\n");
        errorCount++;
    }

    v3Set(1.5, 0.5, 0.5, v3_1);
    v3Set(-0.5, 0.25, 0.15, v3_2);
    subPRV(v3_1, v3_2, v3);
    v3Set(1.899971363060601, 0.06138537390284331, 0.7174863730592785, v3_1);
    if(!v3IsEqual(v3, v3_1, accuracy)) {
        printf("subPRV failed\n");
        errorCount++;
    }

    v3Set(30 * D2R, 30 * D2R, 45 * D2R, v3_1);
    Euler3132C(v3_1, C);
    v3Set(0.306186217848, 0.883883476483, 0.353553390593, C2[0]);
    v3Set(-0.918558653544, 0.176776695297, 0.353553390593, C2[1]);
    v3Set(0.250000000000, -0.433012701892, 0.866025403784, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler3132C failed\n");
        errorCount++;
    }

    v3Set(60 * D2R, 30 * D2R, 45 * D2R, v3_1);
    Euler3212C(v3_1, C);
    v3Set(0.433012701892, 0.750000000000, -0.500000000000, C2[0]);
    v3Set(-0.435595740399, 0.659739608441, 0.612372435696, C2[1]);
    v3Set(0.789149130992, -0.0473671727454, 0.612372435696, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Euler3212C failed\n");
        errorCount++;
    }

    Mi(30 * D2R, 3, C);
    v3Set(0.8660254037844387, 0.4999999999999999, 0, C2[0]);
    v3Set(-0.4999999999999999, 0.8660254037844387, 0, C2[1]);
    v3Set(0, 0, 1.00000000000000, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Mi(30 deg, 3, C) failed\n");
        errorCount++;
    }

    Mi(30 * D2R, 2, C);
    v3Set(0.8660254037844387, 0, -0.4999999999999999, C2[0]);
    v3Set(0, 1.00000000000000, 0, C2[1]);
    v3Set(0.4999999999999999, 0, 0.8660254037844387, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Mi(30 deg, 2, C) failed\n");
        errorCount++;
    }

    Mi(30 * D2R, 1, C);
    v3Set(1.0000000000000000, 0, 0, C2[0]);
    v3Set(0, 0.8660254037844387, 0.4999999999999999, C2[1]);
    v3Set(0, -0.4999999999999999, 0.8660254037844387, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("Mi(30 deg, 1, C) failed\n");
        errorCount++;
    }

    v3Set(1.0, 2.0, 3.0, v3_1);
    tilde(v3_1, C);
    v3Set(0.0, -3.0, 2.0, C2[0]);
    v3Set(3.0, 0.0, -1.0, C2[1]);
    v3Set(-2.0, 1.0, 0.0, C2[2]);
    if(!m33IsEqual(C, C2, accuracy)) {
        printf("tilde() failed\n");
        errorCount++;
    }

    return errorCount;
}
