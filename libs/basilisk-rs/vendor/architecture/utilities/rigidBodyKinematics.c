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

#include "rigidBodyKinematics.h"

#include "linearAlgebra.h"
#include "astroConstants.h"
#include "architecture/utilities/bsk_Print.h"
#include <string.h>

#define nearZero 0.0000000000001

/*
 * Q = addEP(B1,B2) provides the Euler parameter vector
 * which corresponds to performing to successive
 * rotations B1 and B2.
 */
void addEP(double *b1, double *b2, double *result)
{
    result[0] = b2[0] * b1[0] - b2[1] * b1[1] - b2[2] * b1[2] - b2[3] * b1[3];
    result[1] = b2[1] * b1[0] + b2[0] * b1[1] + b2[3] * b1[2] - b2[2] * b1[3];
    result[2] = b2[2] * b1[0] - b2[3] * b1[1] + b2[0] * b1[2] + b2[1] * b1[3];
    result[3] = b2[3] * b1[0] + b2[2] * b1[1] - b2[1] * b1[2] + b2[0] * b1[3];
}

/*
 * addEuler121(E1,E2,Q) computes the overall (1-2-1) Euler
 * angle vector corresponding to two successive
 * (1-2-1) rotations E1 and E2.
 */
void addEuler121(double *e1, double *e2, double *result)
{
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double dum;
    double cp3;

    cp1 = cos(e1[1]);
    cp2 = cos(e2[1]);
    sp1 = sin(e1[1]);
    sp2 = sin(e2[1]);
    dum = e1[2] + e2[0];

    result[1] = safeAcos(cp1 * cp2 - sp1 * sp2 * cos(dum));
    cp3 = cos(result[1]);
    result[0] = wrapToPi(e1[0] + atan2(sp1 * sp2 * sin(dum), cp2 - cp3 * cp1));
    result[2] = wrapToPi(e2[2] + atan2(sp1 * sp2 * sin(dum), cp1 - cp3 * cp2));
}

/*
 * addEuler131(E1,E2,Q) computes the overall (1-3-1) Euler
 * angle vector corresponding to two successive
 * (1-3-1) rotations E1 and E2.
 */
void addEuler131(double *e1, double *e2, double *result)
{
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double dum;
    double cp3;

    cp1 = cos(e1[1]);
    cp2 = cos(e2[1]);
    sp1 = sin(e1[1]);
    sp2 = sin(e2[1]);
    dum = e1[2] + e2[0];

    result[1] = safeAcos(cp1 * cp2 - sp1 * sp2 * cos(dum));
    cp3 = cos(result[1]);
    result[0] = wrapToPi(e1[0] + atan2(sp1 * sp2 * sin(dum), cp2 - cp3 * cp1));
    result[2] = wrapToPi(e2[2] + atan2(sp1 * sp2 * sin(dum), cp1 - cp3 * cp2));
}

/*
 * addEuler123(E1,E2,Q) computes the overall (1-2-3) Euler
 * angle vector corresponding to two successive
 * (1-2-3) rotations E1 and E2.
 */
void addEuler123(double *e1, double *e2, double *result)
{
    double C1[3][3];
    double C2[3][3];
    double C[3][3];

    Euler1232C(e1, C1);
    Euler1232C(e2, C2);
    m33MultM33(C2, C1, C);
    C2Euler123(C, result);
}

/*
 * addEuler132(E1,E2,Q) computes the overall (1-3-2) Euler
 * angle vector corresponding to two successive
 * (1-3-2) rotations E1 and E2.
 */
void addEuler132(double *e1, double *e2, double *result)
{
    double C1[3][3];
    double C2[3][3];
    double C[3][3];

    Euler1322C(e1, C1);
    Euler1322C(e2, C2);
    m33MultM33(C2, C1, C);
    C2Euler132(C, result);
}

/*
 * addEuler212(E1,E2,Q) computes the overall (2-1-2) Euler
 * angle vector corresponding to two successive
 * (2-1-2) rotations E1 and E2.
 */
void addEuler212(double *e1, double *e2, double *result)
{
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double dum;
    double cp3;

    cp1 = cos(e1[1]);
    cp2 = cos(e2[1]);
    sp1 = sin(e1[1]);
    sp2 = sin(e2[1]);
    dum = e1[2] + e2[0];

    result[1] = safeAcos(cp1 * cp2 - sp1 * sp2 * cos(dum));
    cp3 = cos(result[1]);
    result[0] = wrapToPi(e1[0] + atan2(sp1 * sp2 * sin(dum), cp2 - cp3 * cp1));
    result[2] = wrapToPi(e2[2] + atan2(sp1 * sp2 * sin(dum), cp1 - cp3 * cp2));
}

/*
 * addEuler213(E1,E2,Q) computes the overall (2-1-3) Euler
 * angle vector corresponding to two successive
 * (2-1-3) rotations E1 and E2.
 */
void addEuler213(double *e1, double *e2, double *result)
{
    double C1[3][3];
    double C2[3][3];
    double C[3][3];

    Euler2132C(e1, C1);
    Euler2132C(e2, C2);
    m33MultM33(C2, C1, C);
    C2Euler213(C, result);
}

/*
 * addEuler231(E1,E2,Q) computes the overall (2-3-1) Euler
 * angle vector corresponding to two successive
 * (2-3-1) rotations E1 and E2.
 */
void addEuler231(double *e1, double *e2, double *result)
{
    double C1[3][3];
    double C2[3][3];
    double C[3][3];

    Euler2312C(e1, C1);
    Euler2312C(e2, C2);
    m33MultM33(C2, C1, C);
    C2Euler231(C, result);
}

/*
 * addEuler232(E1,E2,Q) computes the overall (2-3-2) Euler
 * angle vector corresponding to two successive
 * (2-3-2) rotations E1 and E2.
 */
void addEuler232(double *e1, double *e2, double *result)
{
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double dum;
    double cp3;

    cp1 = cos(e1[1]);
    cp2 = cos(e2[1]);
    sp1 = sin(e1[1]);
    sp2 = sin(e2[1]);
    dum = e1[2] + e2[0];

    result[1] = safeAcos(cp1 * cp2 - sp1 * sp2 * cos(dum));
    cp3 = cos(result[1]);
    result[0] = wrapToPi(e1[0] + atan2(sp1 * sp2 * sin(dum), cp2 - cp3 * cp1));
    result[2] = wrapToPi(e2[2] + atan2(sp1 * sp2 * sin(dum), cp1 - cp3 * cp2));
}

/*
 * addEuler312(E1,E2,Q) computes the overall (3-1-2) Euler
 * angle vector corresponding to two successive
 * (3-1-2) rotations E1 and E2.
 */
void addEuler312(double *e1, double *e2, double *result)
{
    double C1[3][3];
    double C2[3][3];
    double C[3][3];

    Euler3122C(e1, C1);
    Euler3122C(e2, C2);
    m33MultM33(C2, C1, C);
    C2Euler312(C, result);
}

/*
 * addEuler313(E1,E2,Q) computes the overall (3-1-3) Euler
 * angle vector corresponding to two successive
 * (3-1-3) rotations E1 and E2.
 */
void addEuler313(double *e1, double *e2, double *result)
{
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double dum;
    double cp3;

    cp1 = cos(e1[1]);
    cp2 = cos(e2[1]);
    sp1 = sin(e1[1]);
    sp2 = sin(e2[1]);
    dum = e1[2] + e2[0];

    result[1] = safeAcos(cp1 * cp2 - sp1 * sp2 * cos(dum));
    cp3 = cos(result[1]);
    result[0] = wrapToPi(e1[0] + atan2(sp1 * sp2 * sin(dum), cp2 - cp3 * cp1));
    result[2] = wrapToPi(e2[2] + atan2(sp1 * sp2 * sin(dum), cp1 - cp3 * cp2));
}

/*
 * addEuler321(E1,E2,Q) computes the overall (3-2-1) Euler
 * angle vector corresponding to two successive
 * (3-2-1) rotations E1 and E2.
 */
void addEuler321(double *e1, double *e2, double *result)
{
    double C1[3][3];
    double C2[3][3];
    double C[3][3];

    Euler3212C(e1, C1);
    Euler3212C(e2, C2);
    m33MultM33(C2, C1, C);
    C2Euler321(C, result);
}

/*
 * addEuler323(E1,E2,Q) computes the overall (3-2-3) Euler
 * angle vector corresponding to two successive
 * (3-2-3) rotations E1 and E2.
 */
void addEuler323(double *e1, double *e2, double *result)
{
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double dum;
    double cp3;

    cp1 = cos(e1[1]);
    cp2 = cos(e2[1]);
    sp1 = sin(e1[1]);
    sp2 = sin(e2[1]);
    dum = e1[2] + e2[0];

    result[1] = safeAcos(cp1 * cp2 - sp1 * sp2 * cos(dum));
    cp3 = cos(result[1]);
    result[0] = wrapToPi(e1[0] + atan2(sp1 * sp2 * sin(dum), cp2 - cp3 * cp1));
    result[2] = wrapToPi(e2[2] + atan2(sp1 * sp2 * sin(dum), cp1 - cp3 * cp2));
}

/*
 * Q = addGibbs(Q1,Q2) provides the Gibbs vector
 * which corresponds to performing to successive
 * rotations Q1 and Q2.
 */
void addGibbs(double *q1, double *q2, double *result)
{
    double v1[3];
    double v2[3];

    v3Cross(q1, q2, v1);
    v3Add(q2, v1, v2);
    v3Add(q1, v2, v1);
    v3Scale(1. / (1. - v3Dot(q1, q2)), v1, result);
}

/*
 * addMRP(Q1,Q2,Q) provides the MRP vector
 * which corresponds to performing to successive
 * rotations Q1 and Q2.
 */
void addMRP(double *q1, double *q2, double *result)
{
    double v1[3];
    double v2[3];
    double s1[3];
    double det;
    double mag;

    v3Copy(q1, s1);
    det = (1 + v3Dot(s1, s1)*v3Dot(q2, q2) - 2 * v3Dot(s1, q2));

    if (fabs(det) < 0.1) {
        mag = v3Dot(s1, s1);
        v3Scale(-1./mag, s1, s1);
        det = (1 + v3Dot(s1, s1)*v3Dot(q2, q2) - 2 * v3Dot(s1, q2));
    }

    v3Cross(s1, q2, v1);
    v3Scale(2., v1, v2);
    v3Scale(1 - v3Dot(q2, q2), s1, result);
    v3Add(result, v2, result);
    v3Scale(1 - v3Dot(s1, s1), q2, v1);
    v3Add(result, v1, result);
    v3Scale(1 / det, result, result);

    /* map MRP to inner set */
    mag = v3Dot(result, result);
    if (mag > 1.0){
        v3Scale(-1./mag, result, result);
    }
}

/*
 * addPRV(Q1,Q2,Q) provides the principal rotation vector
 * which corresponds to performing to successive
 * prinicipal rotations Q1 and Q2.
 */
void addPRV(double *qq1, double *qq2, double *result)
{
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double p;
    double sp;
    double e1[3];
    double e2[3];
    double compSum[3];
    double q1[4];
    double q2[4];
    
    v3Add(qq1, qq2, compSum);

    if((v3Norm(qq1) < 1.0E-7 || v3Norm(qq2) < 1.0E-7))
    {
        v3Add(qq1, qq2, result);
        return;
    }

    PRV2elem(qq1, q1);
    PRV2elem(qq2, q2);
    cp1 = cos(q1[0] / 2.);
    cp2 = cos(q2[0] / 2.);
    sp1 = sin(q1[0] / 2.);
    sp2 = sin(q2[0] / 2.);
    v3Set(q1[1], q1[2], q1[3], e1);
    v3Set(q2[1], q2[2], q2[3], e2);

    p = 2 * safeAcos(cp1 * cp2 - sp1 * sp2 * v3Dot(e1, e2));
    if(fabs(p) < 1.0E-13)
    {
        v3SetZero(result);
        return;
    }
    sp = sin(p / 2.);
    v3Scale(cp1 * sp2, e2, q1);
    v3Scale(cp2 * sp1, e1, q2);
    v3Add(q1, q2, result);
    v3Cross(e1, e2, q1);
    v3Scale(sp1 * sp2, q1, q2);
    v3Add(result, q2, result);
    v3Scale(p / sp, result, result);
}

/*
 * BinvEP(Q,B) returns the 3x4 matrix which relates
 * the derivative of Euler parameter vector Q to the
 * body angular velocity vector w.
 * w = 2 [B(Q)]^(-1) dQ/dt
 */
void BinvEP(double *q, double B[3][4])
{
    B[0][0] = -q[1];
    B[0][1] = q[0];
    B[0][2] = q[3];
    B[0][3] = -q[2];
    B[1][0] = -q[2];
    B[1][1] = -q[3];
    B[1][2] = q[0];
    B[1][3] = q[1];
    B[2][0] = -q[3];
    B[2][1] = q[2];
    B[2][2] = -q[1];
    B[2][3] = q[0];
}

/*
 * BinvEuler121(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (1-2-1) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler121(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c2;
    B[0][1] = 0;
    B[0][2] = 1;
    B[1][0] = s2 * s3;
    B[1][1] = c3;
    B[1][2] = 0;
    B[2][0] = s2 * c3;
    B[2][1] = -s3;
    B[2][2] = 0;
}

/*
 * BinvEuler123(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (1-2-3) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler123(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c2 * c3;
    B[0][1] = s3;
    B[0][2] = 0;
    B[1][0] = -c2 * s3;
    B[1][1] = c3;
    B[1][2] = 0;
    B[2][0] = s2;
    B[2][1] = 0;
    B[2][2] = 1;
}

/*
 * BinvEuler131(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (1-3-1) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler131(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c2;
    B[0][1] = 0;
    B[0][2] = 1;
    B[1][0] = -s2 * c3;
    B[1][1] = s3;
    B[1][2] = 0;
    B[2][0] = s2 * s3;
    B[2][1] = c3;
    B[2][2] = 0;
}

/*
 * BinvEuler132(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (1-3-2) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler132(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c2 * c3;
    B[0][1] = -s3;
    B[0][2] = 0;
    B[1][0] = -s2;
    B[1][1] = 0;
    B[1][2] = 1;
    B[2][0] = c2 * s3;
    B[2][1] = c3;
    B[2][2] = 0;
}

/*
 * BinvEuler212(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (2-1-2) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler212(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = s2 * s3;
    B[0][1] = c3;
    B[0][2] = 0;
    B[1][0] = c2;
    B[1][1] = 0;
    B[1][2] = 1;
    B[2][0] = -s2 * c3;
    B[2][1] = s3;
    B[2][2] = 0;
}

/*
 * BinvEuler213(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (2-1-3) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler213(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c2 * s3;
    B[0][1] = c3;
    B[0][2] = 0;
    B[1][0] = c2 * c3;
    B[1][1] = -s3;
    B[1][2] = 0;
    B[2][0] = -s2;
    B[2][1] = 0;
    B[2][2] = 1;
}

/*
 * BinvEuler231(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (2-3-1) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler231(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = s2;
    B[0][1] = 0;
    B[0][2] = 1;
    B[1][0] = c2 * c3;
    B[1][1] = s3;
    B[1][2] = 0;
    B[2][0] = -c2 * s3;
    B[2][1] = c3;
    B[2][2] = 0;
}

/*
 * BinvEuler232(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (2-3-2) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler232(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = s2 * c3;
    B[0][1] = -s3;
    B[0][2] = 0;
    B[1][0] = c2;
    B[1][1] = 0;
    B[1][2] = 1;
    B[2][0] = s2 * s3;
    B[2][1] = c3;
    B[2][2] = 0;
}

/*
 * BinvEuler323(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (3-2-3) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler323(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = -s2 * c3;
    B[0][1] = s3;
    B[0][2] = 0;
    B[1][0] = s2 * s3;
    B[1][1] = c3;
    B[1][2] = 0;
    B[2][0] = c2;
    B[2][1] = 0;
    B[2][2] = 1;
}

/*
 * BinvEuler313(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (3-1-3) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler313(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = s2 * s3;
    B[0][1] = c3;
    B[0][2] = 0;
    B[1][0] = s2 * c3;
    B[1][1] = -s3;
    B[1][2] = 0;
    B[2][0] = c2;
    B[2][1] = 0;
    B[2][2] = 1;
}

/*
 * BinvEuler321(Q,B) returns the 3x3 matrix which relates
 * the derivative of the (3-2-1) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler321(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = -s2;
    B[0][1] = 0;
    B[0][2] = 1;
    B[1][0] = c2 * s3;
    B[1][1] = c3;
    B[1][2] = 0;
    B[2][0] = c2 * c3;
    B[2][1] = -s3;
    B[2][2] = 0;
}

/*
 * BinvEuler312(Q) returns the 3x3 matrix which relates
 * the derivative of the (3-2-3) Euler angle vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvEuler312(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = -c2 * s3;
    B[0][1] = c3;
    B[0][2] = 0;
    B[1][0] = s2;
    B[1][1] = 0;
    B[1][2] = 1;
    B[2][0] = c2 * c3;
    B[2][1] = s3;
    B[2][2] = 0;
}

/*
 * BinvGibbs(Q,B) returns the 3x3 matrix which relates
 * the derivative of Gibbs vector Q to the
 * body angular velocity vector w.
 *
 * w = 2 [B(Q)]^(-1) dQ/dt
 */
void BinvGibbs(double *q, double B[3][3])
{
    B[0][0] = 1;
    B[0][1] = q[2];
    B[0][2] = -q[1];
    B[1][0] = -q[2];
    B[1][1] = 1;
    B[1][2] = q[0];
    B[2][0] = q[1];
    B[2][1] = -q[0];
    B[2][2] = 1;
    m33Scale(1. / (1 + v3Dot(q, q)), B, B);
}

/*
* BinvMRP(Q,B) returns the 3x3 matrix which relates
* the derivative of MRP vector Q to the
* body angular velocity vector w.
*
* w = 4 [B(Q)]^(-1) dQ/dt
*/
void BinvMRP(double *q, double B[3][3])
{
    double s2;

    s2 = v3Dot(q, q);
    B[0][0] = 1 - s2 + 2 * q[0] * q[0];
    B[0][1] = 2 * (q[0] * q[1] + q[2]);
    B[0][2] = 2 * (q[0] * q[2] - q[1]);
    B[1][0] = 2 * (q[1] * q[0] - q[2]);
    B[1][1] = 1 - s2 + 2 * q[1] * q[1];
    B[1][2] = 2 * (q[1] * q[2] + q[0]);
    B[2][0] = 2 * (q[2] * q[0] + q[1]);
    B[2][1] = 2 * (q[2] * q[1] - q[0]);
    B[2][2] = 1 - s2 + 2 * q[2] * q[2];
    m33Scale(1. / (1 + s2) / (1 + s2), B, B);
}

/*
 * BinvPRV(Q,B) returns the 3x3 matrix which relates
 * the derivative of principal rotation vector Q to the
 * body angular velocity vector w.
 *
 * w = [B(Q)]^(-1) dQ/dt
 */
void BinvPRV(double *q, double B[3][3])
{
    double p;
    double c1;
    double c2;

    p = sqrt(v3Dot(q, q));
    c1 = (1 - cos(p)) / p / p;
    c2 = (p - sin(p)) / p / p / p;

    B[0][0] = 1 - c2 * (q[1] * q[1] + q[2] * q[2]);
    B[0][1] = c1 * q[2] + c2 * q[0] * q[1];
    B[0][2] = -c1 * q[1] + c2 * q[0] * q[2];
    B[1][0] = -c1 * q[2] + c2 * q[0] * q[1];
    B[1][1] = 1 - c2 * (q[0] * q[0] + q[2] * q[2]);
    B[1][2] = c1 * q[0] + c2 * q[1] * q[2];
    B[2][0] = c1 * q[1] + c2 * q[2] * q[0];
    B[2][1] = -c1 * q[0] + c2 * q[2] * q[1];
    B[2][2] = 1 - c2 * (q[0] * q[0] + q[1] * q[1]);
}

/*
 * BmatEP(Q,B) returns the 4x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * Euler parameter vector Q.
 *
 * dQ/dt = 1/2 [B(Q)] w
 */
void BmatEP(double *q, double B[4][3])
{
    B[0][0] = -q[1];
    B[0][1] = -q[2];
    B[0][2] = -q[3];
    B[1][0] = q[0];
    B[1][1] = -q[3];
    B[1][2] = q[2];
    B[2][0] = q[3];
    B[2][1] = q[0];
    B[2][2] = -q[1];
    B[3][0] = -q[2];
    B[3][1] = q[1];
    B[3][2] = q[0];
}

/*
 * BmatEuler121(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (1-2-1) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler121(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = 0;
    B[0][1] = s3;
    B[0][2] = c3;
    B[1][0] = 0;
    B[1][1] = s2 * c3;
    B[1][2] = -s2 * s3;
    B[2][0] = s2;
    B[2][1] = -c2 * s3;
    B[2][2] = -c2 * c3;
    m33Scale(1. / s2, B, B);
}

/*
 * BmatEuler131(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (1-3-1) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler131(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = 0;
    B[0][1] = -c3;
    B[0][2] = s3;
    B[1][0] = 0;
    B[1][1] = s2 * s3;
    B[1][2] = s2 * c3;
    B[2][0] = s2;
    B[2][1] = c2 * c3;
    B[2][2] = -c2 * s3;
    m33Scale(1. / s2, B, B);
}

/*
 * BmatEuler123(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (1-2-3) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler123(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c3;
    B[0][1] = -s3;
    B[0][2] = 0;
    B[1][0] = c2 * s3;
    B[1][1] = c2 * c3;
    B[1][2] = 0;
    B[2][0] = -s2 * c3;
    B[2][1] = s2 * s3;
    B[2][2] = c2;
    m33Scale(1. / c2, B, B);
}

/*
 * BmatEuler132(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (1-3-2) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler132(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c3;
    B[0][1] = 0;
    B[0][2] = s3;
    B[1][0] = -c2 * s3;
    B[1][1] = 0;
    B[1][2] = c2 * c3;
    B[2][0] = s2 * c3;
    B[2][1] = c2;
    B[2][2] = s2 * s3;
    m33Scale(1. / c2, B, B);
}

/*
 * BmatEuler212(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (2-1-2) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler212(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = s3;
    B[0][1] = 0;
    B[0][2] = -c3;
    B[1][0] = s2 * c3;
    B[1][1] = 0;
    B[1][2] = s2 * s3;
    B[2][0] = -c2 * s3;
    B[2][1] = s2;
    B[2][2] = c2 * c3;
    m33Scale(1. / s2, B, B);
}

/*
 * BmatEuler213(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (2-1-3) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler213(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = s3;
    B[0][1] = c3;
    B[0][2] = 0;
    B[1][0] = c2 * c3;
    B[1][1] = -c2 * s3;
    B[1][2] = 0;
    B[2][0] = s2 * s3;
    B[2][1] = s2 * c3;
    B[2][2] = c2;
    m33Scale(1. / c2, B, B);
}

/*
 * BmatEuler231(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (2-3-1) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler231(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = 0;
    B[0][1] = c3;
    B[0][2] = -s3;
    B[1][0] = 0;
    B[1][1] = c2 * s3;
    B[1][2] = c2 * c3;
    B[2][0] = c2;
    B[2][1] = -s2 * c3;
    B[2][2] = s2 * s3;
    m33Scale(1. / c2, B, B);
}

/*
 * B = BmatEuler232(Q) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (2-3-2) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler232(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = c3;
    B[0][1] = 0;
    B[0][2] = s3;
    B[1][0] = -s2 * s3;
    B[1][1] = 0;
    B[1][2] = s2 * c3;
    B[2][0] = -c2 * c3;
    B[2][1] = s2;
    B[2][2] = -c2 * s3;
    m33Scale(1. / s2, B, B);
}

/*
 * BmatEuler312(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (3-1-2) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler312(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = -s3;
    B[0][1] = 0;
    B[0][2] = c3;
    B[1][0] = c2 * c3;
    B[1][1] = 0;
    B[1][2] = c2 * s3;
    B[2][0] = s2 * s3;
    B[2][1] = c2;
    B[2][2] = -s2 * c3;
    m33Scale(1. / c2, B, B);
}

/*
 * BmatEuler313(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (3-1-3) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler313(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = s3;
    B[0][1] = c3;
    B[0][2] = 0;
    B[1][0] = c3 * s2;
    B[1][1] = -s3 * s2;
    B[1][2] = 0;
    B[2][0] = -s3 * c2;
    B[2][1] = -c3 * c2;
    B[2][2] = s2;
    m33Scale(1. / s2, B, B);
}

/*
 * BmatEuler321(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (3-2-1) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler321(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0][0] = 0;
    B[0][1] = s3;
    B[0][2] = c3;
    B[1][0] = 0;
    B[1][1] = c2 * c3;
    B[1][2] = -c2 * s3;
    B[2][0] = c2;
    B[2][1] = s2 * s3;
    B[2][2] = s2 * c3;
    m33Scale(1. / c2, B, B);
}

/*
 * BmatEuler323(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * (3-2-3) Euler angle vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatEuler323(double *q, double B[3][3])
{
    double s2;
    double c2;
    double s3;
    double c3;

    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);
    
    B[0][0] = -c3;
    B[0][1] = s3;
    B[0][2] = 0;
    B[1][0] = s2 * s3;
    B[1][1] = s2 * c3;
    B[1][2] = 0;
    B[2][0] = c2 * c3;
    B[2][1] = -c2 * s3;
    B[2][2] = s2;
    m33Scale(1. / s2, B, B);
}

/*
 * BmatGibbs(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * Gibbs vector Q.
 *
 * dQ/dt = 1/2 [B(Q)] w
 */
void BmatGibbs(double *q, double B[3][3])
{
    B[0][0] = 1 + q[0] * q[0];
    B[0][1] = q[0] * q[1] - q[2];
    B[0][2] = q[0] * q[2] + q[1];
    B[1][0] = q[1] * q[0] + q[2];
    B[1][1] = 1 + q[1] * q[1];
    B[1][2] = q[1] * q[2] - q[0];
    B[2][0] = q[2] * q[0] - q[1];
    B[2][1] = q[2] * q[1] + q[0];
    B[2][2] = 1 + q[2] * q[2];
}

/*
 * BmatMRP(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * MRP vector Q.
 *
 * dQ/dt = 1/4 [B(Q)] w
 */
void BmatMRP(double *q, double B[3][3])
{
    double s2;

    s2 = v3Dot(q, q);
    B[0][0] = 1 - s2 + 2 * q[0] * q[0];
    B[0][1] = 2 * (q[0] * q[1] - q[2]);
    B[0][2] = 2 * (q[0] * q[2] + q[1]);
    B[1][0] = 2 * (q[1] * q[0] + q[2]);
    B[1][1] = 1 - s2 + 2 * q[1] * q[1];
    B[1][2] = 2 * (q[1] * q[2] - q[0]);
    B[2][0] = 2 * (q[2] * q[0] - q[1]);
    B[2][1] = 2 * (q[2] * q[1] + q[0]);
    B[2][2] = 1 - s2 + 2 * q[2] * q[2];
}

/*
 * BdotmatMRP(Q,dQ,B) returns the 3x3 matrix derivative of 
 * the BmatMRP matrix, and it is used to relate the 
 * body angular acceleration vector dw to the second order 
 * derivative of the MRP vector Q.
 *
 * (d^2Q)/(dt^2) = 1/4 ( [B(Q)] dw + [Bdot(Q,dQ)] w )
 */
void BdotmatMRP(double *q, double *dq, double B[3][3])
{
    double s;

    s = -2 * v3Dot(q, dq);
    B[0][0] = s + 4 * ( q[0] * dq[0] );
    B[0][1] = 2 * (-dq[2] + q[0] * dq[1] + dq[0] * q[1] );
    B[0][2] = 2 * ( dq[1] + q[0] * dq[2] + dq[0] * q[2] );
    B[1][0] = 2 * ( dq[2] + q[0] * dq[1] + dq[0] * q[1] );
    B[1][1] = s + 4 * ( q[1] * dq[1] );
    B[1][2] = 2 * (-dq[0] + q[1] * dq[2] + dq[1] * q[2] );
    B[2][0] = 2 * (-dq[1] + q[0] * dq[2] + dq[0] * q[2] );
    B[2][1] = 2 * ( dq[0] + q[1] * dq[2] + dq[1] * q[2] );
    B[2][2] = s + 4 * ( q[2] * dq[2] );
}

/*
 * BmatPRV(Q,B) returns the 3x3 matrix which relates the
 * body angular velocity vector w to the derivative of
 * principal rotation vector Q.
 *
 * dQ/dt = [B(Q)] w
 */
void BmatPRV(double *q, double B[3][3])
{
    double p;
    double c;
    p = v3Norm(q);
    c = 1. / p / p * (1. - p / 2. / tan(p / 2.));
    B[0][0] = 1 - c * (q[1] * q[1] + q[2] * q[2]);
    B[0][1] = -q[2] / 2 + c * (q[0] * q[1]);
    B[0][2] = q[1] / 2 + c * (q[0] * q[2]);
    B[1][0] = q[2] / 2 + c * (q[0] * q[1]);
    B[1][1] = 1 - c * (q[0] * q[0] + q[2] * q[2]);
    B[1][2] = -q[0] / 2 + c * (q[1] * q[2]);
    B[2][0] = -q[1] / 2 + c * (q[0] * q[2]);
    B[2][1] = q[0] / 2 + c * (q[1] * q[2]);
    B[2][2] = 1 - c * (q[0] * q[0] + q[1] * q[1]);
}

/*
 * C2EP(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding 4x1 Euler parameter vector Q,
 * where the first component of Q is the non-dimensional
 * Euler parameter Beta_0 >= 0. Transformation is done
 * using the Stanley method.
 *
 */
void C2EP(double C[3][3], double b[4])
{
    double tr;
    double b2[4];
    double max;
    int i;
    int j;

    tr = C[0][0] + C[1][1] + C[2][2];
    b2[0] = (1 + tr) / 4.;
    b2[1] = (1 + 2 * C[0][0] - tr) / 4.;
    b2[2] = (1 + 2 * C[1][1] - tr) / 4.;
    b2[3] = (1 + 2 * C[2][2] - tr) / 4.;

    i = 0;
    max = b2[0];
    for(j = 1; j < 4; j++) {
        if(b2[j] > max) {
            i = j;
            max = b2[j];
        }
    }

    switch(i) {
        case 0:
            b[0] = sqrt(b2[0]);
            b[1] = (C[1][2] - C[2][1]) / 4 / b[0];
            b[2] = (C[2][0] - C[0][2]) / 4 / b[0];
            b[3] = (C[0][1] - C[1][0]) / 4 / b[0];
            break;
        case 1:
            b[1] = sqrt(b2[1]);
            b[0] = (C[1][2] - C[2][1]) / 4 / b[1];
            if(b[0] < 0) {
                b[1] = -b[1];
                b[0] = -b[0];
            }
            b[2] = (C[0][1] + C[1][0]) / 4 / b[1];
            b[3] = (C[2][0] + C[0][2]) / 4 / b[1];
            break;
        case 2:
            b[2] = sqrt(b2[2]);
            b[0] = (C[2][0] - C[0][2]) / 4 / b[2];
            if(b[0] < 0) {
                b[2] = -b[2];
                b[0] = -b[0];
            }
            b[1] = (C[0][1] + C[1][0]) / 4 / b[2];
            b[3] = (C[1][2] + C[2][1]) / 4 / b[2];
            break;
        case 3:
            b[3] = sqrt(b2[3]);
            b[0] = (C[0][1] - C[1][0]) / 4 / b[3];
            if(b[0] < 0) {
                b[3] = -b[3];
                b[0] = -b[0];
            }
            b[1] = (C[2][0] + C[0][2]) / 4 / b[3];
            b[2] = (C[1][2] + C[2][1]) / 4 / b[3];
            break;
    }
}

/*
 * C2Euler121(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (1-2-1) Euler angle set.
 */
void C2Euler121(double C[3][3], double *q)
{
    q[0] = atan2(C[0][1], -C[0][2]);
    q[1] = safeAcos(C[0][0]);
    q[2] = atan2(C[1][0], C[2][0]);
}

/*
 * C2Euler123(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (1-2-3) Euler angle set.
 */
void C2Euler123(double C[3][3], double *q)
{
    q[0] = atan2(-C[2][1], C[2][2]);
    q[1] = safeAsin(C[2][0]);
    q[2] = atan2(-C[1][0], C[0][0]);
}

/*
 * C2Euler131(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (1-3-1) Euler angle set.
 */
void C2Euler131(double C[3][3], double *q)
{
    q[0] = atan2(C[0][2], C[0][1]);
    q[1] = safeAcos(C[0][0]);
    q[2] = atan2(C[2][0], -C[1][0]);
}

/*
 * C2Euler132(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (1-3-2) Euler angle set.
 */
void C2Euler132(double C[3][3], double *q)
{
    q[0] = atan2(C[1][2], C[1][1]);
    q[1] = safeAsin(-C[1][0]);
    q[2] = atan2(C[2][0], C[0][0]);
}

/*
 * C2Euler212(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (2-1-2) Euler angle set.
 */
void C2Euler212(double C[3][3], double *q)
{
    q[0] = atan2(C[1][0], C[1][2]);
    q[1] = safeAcos(C[1][1]);
    q[2] = atan2(C[0][1], -C[2][1]);
}

/*
 * C2Euler213(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (2-1-3) Euler angle set.
 */
void C2Euler213(double C[3][3], double *q)
{
    q[0] = atan2(C[2][0], C[2][2]);
    q[1] = safeAsin(-C[2][1]);
    q[2] = atan2(C[0][1], C[1][1]);
}

/*
 * C2Euler231(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (2-3-1) Euler angle set.
 */
void C2Euler231(double C[3][3], double *q)
{
    q[0] = atan2(-C[0][2], C[0][0]);
    q[1] = safeAsin(C[0][1]);
    q[2] = atan2(-C[2][1], C[1][1]);
}

/*
 * C2Euler232(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (2-3-2) Euler angle set.
 */
void C2Euler232(double C[3][3], double *q)
{
    q[0] = atan2(C[1][2], -C[1][0]);
    q[1] = safeAcos(C[1][1]);
    q[2] = atan2(C[2][1], C[0][1]);
}

/*
 * C2Euler312(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (3-1-2) Euler angle set.
 */
void C2Euler312(double C[3][3], double *q)
{
    q[0] = atan2(-C[1][0], C[1][1]);
    q[1] = safeAsin(C[1][2]);
    q[2] = atan2(-C[0][2], C[2][2]);
}

/*
 * C2Euler313(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (3-1-3) Euler angle set.
 */
void C2Euler313(double C[3][3], double *q)
{
    q[0] = atan2(C[2][0], -C[2][1]);
    q[1] = safeAcos(C[2][2]);
    q[2] = atan2(C[0][2], C[1][2]);
}

/*
 * C2Euler321(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (3-2-1) Euler angle set.
 */
void C2Euler321(double C[3][3], double *q)
{
    q[0] = atan2(C[0][1], C[0][0]);
    q[1] = safeAsin(-C[0][2]);
    q[2] = atan2(C[1][2], C[2][2]);
}

/*
 * C2Euler323(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding (3-2-3) Euler angle set.
 */
void C2Euler323(double C[3][3], double *q)
{
    q[0] = atan2(C[2][1], C[2][0]);
    q[1] = safeAcos(C[2][2]);
    q[2] = atan2(C[1][2], -C[0][2]);
}

/*
 * C2Gibbs(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding 3x1 Gibbs vector Q.
 */
void C2Gibbs(double C[3][3], double *q)
{
    double b[4];

    C2EP(C, b);

    q[0] = b[1] / b[0];
    q[1] = b[2] / b[0];
    q[2] = b[3] / b[0];
}

/*
 * C2MRP(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding 3x1 MRP vector Q where the
 * MRP vector is chosen such that |Q| <= 1.
 */
void C2MRP(double C[3][3], double *q)
{
    double b[4];
    
    v4SetZero(b);
    b[0] = 1.0;
    C2EP(C, b);

    q[0] = b[1] / (1 + b[0]);
    q[1] = b[2] / (1 + b[0]);
    q[2] = b[3] / (1 + b[0]);
}

/*
 * C2PRV(C,Q) translates the 3x3 direction cosine matrix
 * C into the corresponding 3x1 principal rotation vector Q,
 * where the first component of Q is the principal rotation angle
 * phi (0<= phi <= Pi)
 */
void C2PRV(double C[3][3], double *q)
{
    double beta[4];

    C2EP(C,beta);
    EP2PRV(beta,q);
}

/*
 * dEP(Q,W,dq) returns the Euler parameter derivative
 * for a given Euler parameter vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt = 1/2 [B(Q)] w
 */
void dEP(double *q, double *w, double *dq)
{
    double B[4][3];
    int i;
    int j;

    BmatEP(q, B);
    m33MultV3(B, w, dq);
    for(i = 0; i < 4; i++) {
        dq[i] = 0.;
        for(j = 0; j < 3; j++) {
            dq[i] += B[i][j] * w[j];
        }
    }
    v3Scale(.5, dq, dq);
    dq[3] = 0.5 * dq[3];
}

/*
 * dEuler121(Q,W,dq) returns the (1-2-1) Euler angle derivative
 * vector for a given (1-2-1) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler121(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler121(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler123(Q,W,dq) returns the (1-2-3) Euler angle derivative
 * vector for a given (1-2-3) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler123(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler123(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler131(Q,W,dq) returns the (1-3-1) Euler angle derivative
 * vector for a given (1-3-1) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler131(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler131(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler132(Q,W,dq) returns the (1-3-2) Euler angle derivative
 * vector for a given (1-3-2) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler132(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler132(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler212(Q,W,dq) returns the (2-1-2) Euler angle derivative
 * vector for a given (2-1-2) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler212(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler212(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler213(Q,W,dq) returns the (2-1-3) Euler angle derivative
 * vector for a given (2-1-3) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler213(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler213(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler231(Q,W,dq) returns the (2-3-1) Euler angle derivative
 * vector for a given (2-3-1) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler231(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler231(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler232(Q,W,dq) returns the (2-3-2) Euler angle derivative
 * vector for a given (2-3-2) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler232(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler232(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler312(Q,W,dq) returns the (3-1-2) Euler angle derivative
 * vector for a given (3-1-2) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler312(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler312(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler313(Q,W,dq) returns the (3-1-3) Euler angle derivative
 * vector for a given (3-1-3) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler313(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler313(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler321(Q,W,dq) returns the (3-2-1) Euler angle derivative
 * vector for a given (3-2-1) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler321(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler321(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dEuler323(Q,W,dq) returns the (3-2-3) Euler angle derivative
 * vector for a given (3-2-3) Euler angle vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dEuler323(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatEuler323(q, B);
    m33MultV3(B, w, dq);
}

/*
 * dGibbs(Q,W,dq) returns the Gibbs derivative
 * for a given Gibbs vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt = 1/2 [B(Q)] w
 */
void dGibbs(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatGibbs(q, B);
    m33MultV3(B, w, dq);
    v3Scale(0.5, dq, dq);
}

/*
 * dMRP(Q,W,dq) returns the MRP derivative
 * for a given MRP vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt = 1/4 [B(Q)] w
 */
void dMRP(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatMRP(q, B);
    m33MultV3(B, w, dq);
    v3Scale(0.25, dq, dq);
}

/*
 * dMRP2omega(Q,dQ,W) returns the angular rate W
 * for a given MRP vector Q and 
 * MRP derivative dQ.
 * 
 * w = 4 [B(Q)]^(-1) dQ/dt
 */
void dMRP2Omega(double *q, double *dq, double *w)
{
    double B[3][3];

    BinvMRP(q, B);
    m33MultV3(B, dq, w);
    v3Scale(4, w, w);
}

/*
 * ddMRP(Q,dQ,W,dW) returns the second order MRP derivative
 * for a given MRP vector Q, first MRP derivative dQ, body angular
 * velocity vector w and body angular acceleration vector dw.
 * 
 * (d^2Q)/(dt^2) = 1/4 ( [B(Q)] dw + [Bdot(Q,dQ)] w )
 */
void ddMRP(double *q, double *dq, double *w, double *dw, double *ddq)
{
    double B[3][3], Bdot[3][3];
    double s1[3], s2[3];
    int i;

    BmatMRP(q, B);
    BdotmatMRP(q, dq, Bdot);
    m33MultV3(B, dw, s1);
    m33MultV3(Bdot, w, s2);
    for(i = 0; i < 3; i++) {
        ddq[i] = 0.25 * ( s1[i] + s2[i] );
    }
}

/*
 * ddMRP2omegaDot(Q,dQ,ddQ) returns the angular rate W
 * for a given MRP vector Q and 
 * MRP derivative dQ.
 * 
 * dW/dt = 4 [B(Q)]^(-1) ( ddQ - [Bdot(Q,dQ)] [B(Q)]^(-1) dQ )
 */
void ddMRP2dOmega(double *q, double *dq, double *ddq, double *dw)
{
    double B[3][3], Bdot[3][3];
    double s1[3], s2[3], s3[3];
    int i;

    BinvMRP(q, B);
    BdotmatMRP(q, dq, Bdot);
    m33MultV3(B, dq, s1);
    m33MultV3(Bdot, s1, s2);
    for(i = 0; i < 3; i++) {
        s3[i] = ddq[i] - s2[i];
    }
    m33MultV3(B, s3, dw);
    v3Scale(4, dw, dw);
}

/*
 * dPRV(Q,W,dq) returns the PRV derivative
 * for a given PRV vector Q and body
 * angular velocity vector w.
 *
 * dQ/dt =  [B(Q)] w
 */
void dPRV(double *q, double *w, double *dq)
{
    double B[3][3];

    BmatPRV(q, B);
    m33MultV3(B, w, dq);
}

/*
 * elem2PRV(R,Q) translates a prinicpal rotation
 * element set R into the corresponding principal
 * rotation vector Q.
 */
void elem2PRV(double *r, double *q)
{
    q[0] = r[1] * r[0];
    q[1] = r[2] * r[0];
    q[2] = r[3] * r[0];
}

/*
 * EP2C(Q,C) returns the direction cosine
 * matrix in terms of the 4x1 Euler parameter vector
 * Q.  The first element is the non-dimensional Euler
 * parameter, while the remain three elements form
 * the Eulerparameter vector.
 */
void EP2C(double *q, double C[3][3])
{
    double q0;
    double q1;
    double q2;
    double q3;

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    C[0][0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
    C[0][1] = 2 * (q1 * q2 + q0 * q3);
    C[0][2] = 2 * (q1 * q3 - q0 * q2);
    C[1][0] = 2 * (q1 * q2 - q0 * q3);
    C[1][1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
    C[1][2] = 2 * (q2 * q3 + q0 * q1);
    C[2][0] = 2 * (q1 * q3 + q0 * q2);
    C[2][1] = 2 * (q2 * q3 - q0 * q1);
    C[2][2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
}

/*
 * EP2Euler121(Q,E) translates the Euler parameter
 * vector Q into the corresponding (1-2-1) Euler angle
 * vector E.
 */
void EP2Euler121(double *q, double *e)
{
    double t1;
    double t2;

    t1 = atan2(q[3], q[2]);
    t2 = atan2(q[1], q[0]);

    e[0] = t1 + t2;
    e[1] = 2 * safeAcos(sqrt(q[0] * q[0] + q[1] * q[1]));
    e[2] = t2 - t1;
}

/*
 * EP2Euler123(Q,E) translates the Euler parameter vector
 * Q into the corresponding (1-2-3) Euler angle set.
 */
void EP2Euler123(double *q, double *e)
{
    double q0;
    double q1;
    double q2;
    double q3;

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    e[0] = atan2(-2 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
    e[1] = safeAsin(2 * (q1 * q3 + q0 * q2));
    e[2] = atan2(-2 * (q1 * q2 - q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);
}

/*
 * EP2Euler131(Q,E) translates the Euler parameter
 * vector Q into the corresponding (1-3-1) Euler angle
 * vector E.
 */
void EP2Euler131(double *q, double *e)
{
    double t1;
    double t2;

    t1 = atan2(q[2], q[3]);
    t2 = atan2(q[1], q[0]);

    e[0] = t2 - t1;
    e[1] = 2 * safeAcos(sqrt(q[0] * q[0] + q[1] * q[1]));
    e[2] = t2 + t1;
}

/*
 * EP2Euler132(Q,E) translates the Euler parameter vector
 * Q into the corresponding (1-3-2) Euler angle set.
 */
void EP2Euler132(double *q, double *e)
{
    double q0;
    double q1;
    double q2;
    double q3;

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    e[0] = atan2(2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3);
    e[1] = safeAsin(-2 * (q1 * q2 - q0 * q3));
    e[2] = atan2(2 * (q1 * q3 + q0 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);
}

/*
 * EP2Euler212(Q,E) translates the Euler parameter
 * vector Q into the corresponding (2-1-2) Euler angle
 * vector E.
 */
void EP2Euler212(double *q, double *e)
{
    double t1;
    double t2;

    t1 = atan2(q[3], q[1]);
    t2 = atan2(q[2], q[0]);

    e[0] = t2 - t1;
    e[1] = 2 * safeAcos(sqrt(q[0] * q[0] + q[2] * q[2]));
    e[2] = t2 + t1;
}

/*
 * EP2Euler213(Q,E) translates the Euler parameter vector
 * Q into the corresponding (2-1-3) Euler angle set.
 */
void EP2Euler213(double *q, double *e)
{
    double q0;
    double q1;
    double q2;
    double q3;

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    e[0] = atan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
    e[1] = safeAsin(-2 * (q2 * q3 - q0 * q1));
    e[2] = atan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3);
}

/*
 * EP2Euler231(Q,E) translates the Euler parameter vector
 * Q into the corresponding (2-3-1) Euler angle set.
 */
void EP2Euler231(double *q, double *e)
{
    double q0;
    double q1;
    double q2;
    double q3;

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    e[0] = atan2(-2 * (q1 * q3 - q0 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);
    e[1] = safeAsin(2 * (q1 * q2 + q0 * q3));
    e[2] = atan2(-2 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3);
}

/*
 * EP2Euler232(Q,E) translates the Euler parameter
 * vector Q into the corresponding (2-3-2) Euler angle
 * vector E.
 */
void EP2Euler232(double *q, double *e)
{
    double t1;
    double t2;

    t1 = atan2(q[1], q[3]);
    t2 = atan2(q[2], q[0]);

    e[0] = t1 + t2;
    e[1] = 2 * safeAcos(sqrt(q[0] * q[0] + q[2] * q[2]));
    e[2] = t2 - t1;
}

/*
 * EP2Euler312(Q,E) translates the Euler parameter vector
 * Q into the corresponding (3-1-2) Euler angle set.
 */
void EP2Euler312(double *q, double *e)
{
    double q0;
    double q1;
    double q2;
    double q3;

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    e[0] = atan2(-2 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3);
    e[1] = safeAsin(2 * (q2 * q3 + q0 * q1));
    e[2] = atan2(-2 * (q1 * q3 - q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
}

/*
 * EP2Euler313(Q,E) translates the Euler parameter
 * vector Q into the corresponding (3-1-3) Euler angle
 * vector E.
 */
void EP2Euler313(double *q, double *e)
{
    double t1;
    double t2;

    t1 = atan2(q[2], q[1]);
    t2 = atan2(q[3], q[0]);

    e[0] = t1 + t2;
    e[1] = 2 * safeAcos(sqrt(q[0] * q[0] + q[3] * q[3]));
    e[2] = t2 - t1;
}

/*
 * EP2Euler321(Q,E) translates the Euler parameter vector
 * Q into the corresponding (3-2-1) Euler angle set.
 */
void EP2Euler321(double *q, double *e)
{
    double q0;
    double q1;
    double q2;
    double q3;

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    e[0] = atan2(2 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);
    e[1] = safeAsin(-2 * (q1 * q3 - q0 * q2));
    e[2] = atan2(2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
}

/*
 * EP2Euler323(Q,E) translates the Euler parameter
 * vector Q into the corresponding (3-2-3) Euler angle
 * vector E.
 */
void EP2Euler323(double *q, double *e)
{
    double t1;
    double t2;

    t1 = atan2(q[1], q[2]);
    t2 = atan2(q[3], q[0]);

    e[0] = t2 - t1;
    e[1] = 2 * safeAcos(sqrt(q[0] * q[0] + q[3] * q[3]));
    e[2] = t2 + t1;
}

/*
 * EP2Gibbs(Q1,Q) translates the Euler parameter vector Q1
 * into the Gibbs vector Q.
 */
void EP2Gibbs(double *q1, double *q)
{
    q[0] = q1[1] / q1[0];
    q[1] = q1[2] / q1[0];
    q[2] = q1[3] / q1[0];
}

/*
 * EP2MRP(Q1,Q) translates the Euler parameter vector Q1
 * into the MRP vector Q.
 */
void EP2MRP(double *q1, double *q)
{
    if (q1[0] >= 0){
        q[0] = q1[1] / (1 + q1[0]);
        q[1] = q1[2] / (1 + q1[0]);
        q[2] = q1[3] / (1 + q1[0]);
    } else {
        q[0] = -q1[1] / (1 - q1[0]);
        q[1] = -q1[2] / (1 - q1[0]);
        q[2] = -q1[3] / (1 - q1[0]);
    }
}

/*
 * EP2PRV(Q1,Q) translates the Euler parameter vector Q1
 * into the principal rotation vector Q.
 */
void EP2PRV(double *q1, double *q)
{
    double p;
    double sp;

    p = 2 * safeAcos(q1[0]);
    sp = sin(p / 2);
    if (fabs(sp) < nearZero) {
        q[0] = 0.0;
        q[1] = 0.0;
        q[2] = 0.0;
        return;
    }
    q[0] = q1[1] / sp * p;
    q[1] = q1[2] / sp * p;
    q[2] = q1[3] / sp * p;
}

/*
 *  Euler1(X,M)  Elementary rotation matrix
 * Returns the elementary rotation matrix about the
 * first body axis.
 */
void Euler1(double x, double m[3][3])
{
    m33SetIdentity(m);
    m[1][1] = cos(x);
    m[1][2] = sin(x);
    m[2][1] = -m[1][2];
    m[2][2] = m[1][1];
}

/*
 *  Euler2(X,M)  Elementary rotation matrix
 * Returns the elementary rotation matrix about the
 * second body axis.
 */
void Euler2(double x, double m[3][3])
{
    m33SetIdentity(m);
    m[0][0] = cos(x);
    m[0][2] = -sin(x);
    m[2][0] = -m[0][2];
    m[2][2] = m[0][0];
}

/*
 *  Euler3(X,M)  Elementary rotation matrix
 * Returns the elementary rotation matrix about the
 * third body axis.
 */
void Euler3(double x, double m[3][3])
{
    m33SetIdentity(m);
    m[0][0] = cos(x);
    m[0][1] = sin(x);
    m[1][0] = -m[0][1];
    m[1][1] = m[0][0];
}

/*
 * Euler1212C(Q,C) returns the direction cosine
 * matrix in terms of the 1-2-1 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler1212C(double *q, double C[3][3])
{
    double st1;
    double ct1;
    double st2;
    double ct2;
    double st3;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct2;
    C[0][1] = st1 * st2;
    C[0][2] = -ct1 * st2;
    C[1][0] = st2 * st3;
    C[1][1] = ct1 * ct3 - ct2 * st1 * st3;
    C[1][2] = ct3 * st1 + ct1 * ct2 * st3;
    C[2][0] = ct3 * st2;
    C[2][1] = -ct2 * ct3 * st1 - ct1 * st3;
    C[2][2] = ct1 * ct2 * ct3 - st1 * st3;
}

/*
 * Euler1212EP(E,Q) translates the 121 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler1212EP(double *e, double *q)
{
    double e1;
    double e2;
    double e3;

    e1 = e[0] / 2;
    e2 = e[1] / 2;
    e3 = e[2] / 2;

    q[0] = cos(e2) * cos(e1 + e3);
    q[1] = cos(e2) * sin(e1 + e3);
    q[2] = sin(e2) * cos(e1 - e3);
    q[3] = sin(e2) * sin(e1 - e3);
}

/*
 * Euler1212Gibbs(E,Q) translates the (1-2-1) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler1212Gibbs(double *e, double *q)
{
    double ep[4];

    Euler1212EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler1212MRP(E,Q) translates the (1-2-1) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler1212MRP(double *e, double *q)
{
    double ep[4];

    Euler1212EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler1212PRV(E,Q) translates the (1-2-1) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler1212PRV(double *e, double *q)
{
    double ep[4];

    Euler1212EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler1232C(Q,C) returns the direction cosine
 * matrix in terms of the 1-2-3 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler1232C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct2 * ct3;
    C[0][1] = ct3 * st1 * st2 + ct1 * st3;
    C[0][2] = st1 * st3 - ct1 * ct3 * st2;
    C[1][0] = -ct2 * st3;
    C[1][1] = ct1 * ct3 - st1 * st2 * st3;
    C[1][2] = ct3 * st1 + ct1 * st2 * st3;
    C[2][0] = st2;
    C[2][1] = -ct2 * st1;
    C[2][2] = ct1 * ct2;
}

/*
 * Euler1232EP(E,Q) translates the 123 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler1232EP(double *e, double *q)
{
    double c1;
    double c2;
    double c3;
    double s1;
    double s2;
    double s3;

    c1 = cos(e[0] / 2);
    s1 = sin(e[0] / 2);
    c2 = cos(e[1] / 2);
    s2 = sin(e[1] / 2);
    c3 = cos(e[2] / 2);
    s3 = sin(e[2] / 2);

    q[0] = c1 * c2 * c3 - s1 * s2 * s3;
    q[1] = s1 * c2 * c3 + c1 * s2 * s3;
    q[2] = c1 * s2 * c3 - s1 * c2 * s3;
    q[3] = c1 * c2 * s3 + s1 * s2 * c3;
}

/*
 * Euler1232Gibbs(E,Q) translates the (1-2-3) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler1232Gibbs(double *e, double *q)
{
    double ep[4];

    Euler1232EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler1232MRP(E,Q) translates the (1-2-3) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler1232MRP(double *e, double *q)
{
    double ep[4];

    Euler1232EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler1232PRV(E,Q) translates the (1-2-3) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler1232PRV(double *e, double *q)
{
    double ep[4];

    Euler1232EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler1312C(Q,C) returns the direction cosine
 * matrix in terms of the 1-3-1 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler1312C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct2;
    C[0][1] = ct1 * st2;
    C[0][2] = st1 * st2;
    C[1][0] = -ct3 * st2;
    C[1][1] = ct1 * ct2 * ct3 - st1 * st3;
    C[1][2] = ct2 * ct3 * st1 + ct1 * st3;
    C[2][0] = st2 * st3;
    C[2][1] = -ct3 * st1 - ct1 * ct2 * st3;
    C[2][2] = ct1 * ct3 - ct2 * st1 * st3;
}

/*
 * Euler1312EP(E,Q) translates the 131 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler1312EP(double *e, double *q)
{
    double e1;
    double e2;
    double e3;

    e1 = e[0] / 2;
    e2 = e[1] / 2;
    e3 = e[2] / 2;

    q[0] = cos(e2) * cos(e1 + e3);
    q[1] = cos(e2) * sin(e1 + e3);
    q[2] = sin(e2) * sin(-e1 + e3);
    q[3] = sin(e2) * cos(-e1 + e3);
}

/*
 * Euler1312Gibbs(E,Q) translates the (1-3-1) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler1312Gibbs(double *e, double *q)
{
    double ep[4];

    Euler1312EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler1312MRP(E,Q) translates the (1-3-1) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler1312MRP(double *e, double *q)
{
    double ep[4];

    Euler1312EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler1312PRV(E,Q) translates the (1-3-1) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler1312PRV(double *e, double *q)
{
    double ep[4];

    Euler1312EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler1322C(Q,C) returns the direction cosine
 * matrix in terms of the 1-3-2 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler1322C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct2 * ct3;
    C[0][1] = ct1 * ct3 * st2 + st1 * st3;
    C[0][2] = ct3 * st1 * st2 - ct1 * st3;
    C[1][0] = -st2;
    C[1][1] = ct1 * ct2;
    C[1][2] = ct2 * st1;
    C[2][0] = ct2 * st3;
    C[2][1] = -ct3 * st1 + ct1 * st2 * st3;
    C[2][2] = ct1 * ct3 + st1 * st2 * st3;
}

/*
 * Euler1322EP(E,Q) translates the 132 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler1322EP(double *e, double *q)
{
    double c1;
    double c2;
    double c3;
    double s1;
    double s2;
    double s3;

    c1 = cos(e[0] / 2);
    s1 = sin(e[0] / 2);
    c2 = cos(e[1] / 2);
    s2 = sin(e[1] / 2);
    c3 = cos(e[2] / 2);
    s3 = sin(e[2] / 2);

    q[0] = c1 * c2 * c3 + s1 * s2 * s3;
    q[1] = s1 * c2 * c3 - c1 * s2 * s3;
    q[2] = c1 * c2 * s3 - s1 * s2 * c3;
    q[3] = c1 * s2 * c3 + s1 * c2 * s3;
}

/*
 * Euler1322Gibbs(E,Q) translates the (1-3-2) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler1322Gibbs(double *e, double *q)
{
    double ep[4];

    Euler1322EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler1322MRP(E,Q) translates the (1-3-2) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler1322MRP(double *e, double *q)
{
    double ep[4];

    Euler1322EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler1322PRV(E,Q) translates the (1-3-2) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler1322PRV(double *e, double *q)
{
    double ep[4];

    Euler1322EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler2122C(Q,C) returns the direction cosine
 * matrix in terms of the 2-1-2 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler2122C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct1 * ct3 - ct2 * st1 * st3;
    C[0][1] = st2 * st3;
    C[0][2] = -ct3 * st1 - ct1 * ct2 * st3;
    C[1][0] = st1 * st2;
    C[1][1] = ct2;
    C[1][2] = ct1 * st2;
    C[2][0] = ct2 * ct3 * st1 + ct1 * st3;
    C[2][1] = -ct3 * st2;
    C[2][2] = ct1 * ct2 * ct3 - st1 * st3;
}

/*
 * Euler2122EP(E,Q) translates the 212 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler2122EP(double *e, double *q)
{
    double e1;
    double e2;
    double e3;

    e1 = e[0] / 2;
    e2 = e[1] / 2;
    e3 = e[2] / 2;

    q[0] = cos(e2) * cos(e1 + e3);
    q[1] = sin(e2) * cos(-e1 + e3);
    q[2] = cos(e2) * sin(e1 + e3);
    q[3] = sin(e2) * sin(-e1 + e3);
}

/*
 * Euler2122Gibbs(E,Q) translates the (2-1-2) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler2122Gibbs(double *e, double *q)
{
    double ep[4];

    Euler2122EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler2122MRP(E,Q) translates the (2-1-2) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler2122MRP(double *e, double *q)
{
    double ep[4];

    Euler2122EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler2122PRV(E,Q) translates the (2-1-2) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler2122PRV(double *e, double *q)
{
    double ep[4];

    Euler2122EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler2132C(Q,C) returns the direction cosine
 * matrix in terms of the 2-1-3 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler2132C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct1 * ct3 + st1 * st2 * st3;
    C[0][1] = ct2 * st3;
    C[0][2] = -ct3 * st1 + ct1 * st2 * st3;
    C[1][0] = ct3 * st1 * st2 - ct1 * st3;
    C[1][1] = ct2 * ct3;
    C[1][2] = ct1 * ct3 * st2 + st1 * st3;
    C[2][0] = ct2 * st1;
    C[2][1] = -st2;
    C[2][2] = ct1 * ct2;
}

/*
 * Euler2132EP(E,Q) translates the 213 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler2132EP(double *e, double *q)
{
    double c1;
    double c2;
    double c3;
    double s1;
    double s2;
    double s3;

    c1 = cos(e[0] / 2);
    s1 = sin(e[0] / 2);
    c2 = cos(e[1] / 2);
    s2 = sin(e[1] / 2);
    c3 = cos(e[2] / 2);
    s3 = sin(e[2] / 2);

    q[0] = c1 * c2 * c3 + s1 * s2 * s3;
    q[1] = c1 * s2 * c3 + s1 * c2 * s3;
    q[2] = s1 * c2 * c3 - c1 * s2 * s3;
    q[3] = c1 * c2 * s3 - s1 * s2 * c3;
}

/*
 * Euler2132Gibbs(E,Q) translates the (2-1-3) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler2132Gibbs(double *e, double *q)
{
    double ep[4];

    Euler2132EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler2132MRP(E,Q) translates the (2-1-3) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler2132MRP(double *e, double *q)
{
    double ep[4];

    Euler2132EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler2132PRV(E,Q) translates the (2-1-3) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler2132PRV(double *e, double *q)
{
    double ep[4];

    Euler2132EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler2312C(Q,C) returns the direction cosine
 * matrix in terms of the 2-3-1 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler2312C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct1 * ct2;
    C[0][1] = st2;
    C[0][2] = -ct2 * st1;
    C[1][0] = -ct1 * ct3 * st2 + st1 * st3;
    C[1][1] = ct2 * ct3;
    C[1][2] = ct3 * st1 * st2 + ct1 * st3;
    C[2][0] = ct3 * st1 + ct1 * st2 * st3;
    C[2][1] = -ct2 * st3;
    C[2][2] = ct1 * ct3 - st1 * st2 * st3;
}

/*
 * Euler2312EP(E,Q) translates the 231 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler2312EP(double *e, double *q)
{
    double c1;
    double c2;
    double c3;
    double s1;
    double s2;
    double s3;

    c1 = cos(e[0] / 2);
    s1 = sin(e[0] / 2);
    c2 = cos(e[1] / 2);
    s2 = sin(e[1] / 2);
    c3 = cos(e[2] / 2);
    s3 = sin(e[2] / 2);

    q[0] = c1 * c2 * c3 - s1 * s2 * s3;
    q[1] = c1 * c2 * s3 + s1 * s2 * c3;
    q[2] = s1 * c2 * c3 + c1 * s2 * s3;
    q[3] = c1 * s2 * c3 - s1 * c2 * s3;
}

/*
 * Euler2312Gibbs(E,Q) translates the (2-3-1) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler2312Gibbs(double *e, double *q)
{
    double ep[4];

    Euler2312EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler2312MRP(E,Q) translates the (2-3-1) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler2312MRP(double *e, double *q)
{
    double ep[4];

    Euler2312EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler2312PRV(E,Q) translates the (2-3-1) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler2312PRV(double *e, double *q)
{
    double ep[4];

    Euler2312EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler2322C(Q) returns the direction cosine
 * matrix in terms of the 2-3-2 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler2322C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct1 * ct2 * ct3 - st1 * st3;
    C[0][1] = ct3 * st2;
    C[0][2] = -ct2 * ct3 * st1 - ct1 * st3;
    C[1][0] = -ct1 * st2;
    C[1][1] = ct2;
    C[1][2] = st1 * st2;
    C[2][0] = ct3 * st1 + ct1 * ct2 * st3;
    C[2][1] = st2 * st3;
    C[2][2] = ct1 * ct3 - ct2 * st1 * st3;
}

/*
* Euler2322EP(E,Q) translates the 232 Euler angle
* vector E into the Euler parameter vector Q.
*/
void Euler2322EP(double *e, double *q)
{
    double e1;
    double e2;
    double e3;

    e1 = e[0] / 2;
    e2 = e[1] / 2;
    e3 = e[2] / 2;

    q[0] = cos(e2) * cos(e1 + e3);
    q[1] = sin(e2) * sin(e1 - e3);
    q[2] = cos(e2) * sin(e1 + e3);
    q[3] = sin(e2) * cos(e1 - e3);
}

/*
 * Euler2322Gibbs(E) translates the (2-3-2) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler2322Gibbs(double *e, double *q)
{
    double ep[4];

    Euler2322EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler2322MRP(E,Q) translates the (2-3-2) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler2322MRP(double *e, double *q)
{
    double ep[4];

    Euler2322EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler2322PRV(E,Q) translates the (2-3-2) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler2322PRV(double *e, double *q)
{
    double ep[4];

    Euler2322EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler3122C(Q,C) returns the direction cosine
 * matrix in terms of the 1-2-3 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler3122C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct1 * ct3 - st1 * st2 * st3;
    C[0][1] = ct3 * st1 + ct1 * st2 * st3;
    C[0][2] = -ct2 * st3;
    C[1][0] = -ct2 * st1;
    C[1][1] = ct1 * ct2;
    C[1][2] = st2;
    C[2][0] = ct3 * st1 * st2 + ct1 * st3;
    C[2][1] = st1 * st3 - ct1 * ct3 * st2;
    C[2][2] = ct2 * ct3;
}

/*
 * Euler3122EP(E,Q) translates the 312 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler3122EP(double *e, double *q)
{
    double c1;
    double c2;
    double c3;
    double s1;
    double s2;
    double s3;

    c1 = cos(e[0] / 2);
    s1 = sin(e[0] / 2);
    c2 = cos(e[1] / 2);
    s2 = sin(e[1] / 2);
    c3 = cos(e[2] / 2);
    s3 = sin(e[2] / 2);

    q[0] = c1 * c2 * c3 - s1 * s2 * s3;
    q[1] = c1 * s2 * c3 - s1 * c2 * s3;
    q[2] = c1 * c2 * s3 + s1 * s2 * c3;
    q[3] = s1 * c2 * c3 + c1 * s2 * s3;
}

/*
 * Euler3122Gibbs(E,Q) translates the (3-1-2) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler3122Gibbs(double *e, double *q)
{
    double ep[4];

    Euler3122EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler3122MRP(E,Q) translates the (3-1-2) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler3122MRP(double *e, double *q)
{
    double ep[4];

    Euler3122EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler3122PRV(E,Q) translates the (3-1-2) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler3122PRV(double *e, double *q)
{
    double ep[4];

    Euler3122EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler3132C(Q,C) returns the direction cosine
 * matrix in terms of the 3-1-3 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler3132C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct3 * ct1 - st3 * ct2 * st1;
    C[0][1] = ct3 * st1 + st3 * ct2 * ct1;
    C[0][2] = st3 * st2;
    C[1][0] = -st3 * ct1 - ct3 * ct2 * st1;
    C[1][1] = -st3 * st1 + ct3 * ct2 * ct1;
    C[1][2] = ct3 * st2;
    C[2][0] = st2 * st1;
    C[2][1] = -st2 * ct1;
    C[2][2] = ct2;
}

/*
 * Euler3132EP(E,Q) translates the 313 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler3132EP(double *e, double *q)
{
    double e1;
    double e2;
    double e3;

    e1 = e[0] / 2;
    e2 = e[1] / 2;
    e3 = e[2] / 2;

    q[0] = cos(e2) * cos(e1 + e3);
    q[1] = sin(e2) * cos(e1 - e3);
    q[2] = sin(e2) * sin(e1 - e3);
    q[3] = cos(e2) * sin(e1 + e3);
}

/*
 * Euler3132Gibbs(E,Q) translates the (3-1-3) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler3132Gibbs(double *e, double *q)
{
    double ep[4];

    Euler3132EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler3132MRP(E,Q) translates the (3-1-3) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler3132MRP(double *e, double *q)
{
    double ep[4];

    Euler3132EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler3132PRV(E,Q) translates the (3-1-3) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler3132PRV(double *e, double *q)
{
    double ep[4];

    Euler3132EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler3212C(Q,C) returns the direction cosine
 * matrix in terms of the 3-2-1 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler3212C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct2 * ct1;
    C[0][1] = ct2 * st1;
    C[0][2] = -st2;
    C[1][0] = st3 * st2 * ct1 - ct3 * st1;
    C[1][1] = st3 * st2 * st1 + ct3 * ct1;
    C[1][2] = st3 * ct2;
    C[2][0] = ct3 * st2 * ct1 + st3 * st1;
    C[2][1] = ct3 * st2 * st1 - st3 * ct1;
    C[2][2] = ct3 * ct2;
}

/*
 * Euler3212EPE,Q) translates the 321 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler3212EP(double *e, double *q)
{
    double c1;
    double c2;
    double c3;
    double s1;
    double s2;
    double s3;

    c1 = cos(e[0] / 2);
    s1 = sin(e[0] / 2);
    c2 = cos(e[1] / 2);
    s2 = sin(e[1] / 2);
    c3 = cos(e[2] / 2);
    s3 = sin(e[2] / 2);

    q[0] = c1 * c2 * c3 + s1 * s2 * s3;
    q[1] = c1 * c2 * s3 - s1 * s2 * c3;
    q[2] = c1 * s2 * c3 + s1 * c2 * s3;
    q[3] = s1 * c2 * c3 - c1 * s2 * s3;
}

/*
 * Euler3212Gibbs(E,Q) translates the (3-2-1) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler3212Gibbs(double *e, double *q)
{
    double ep[4];

    Euler3212EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler3212MRP(E,Q) translates the (3-2-1) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler3212MRP(double *e, double *q)
{
    double ep[4];

    Euler3212EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler3212PRV(E,Q) translates the (3-2-1) Euler
 * angle vector E into the principal rotation vector Q.
 */
void Euler3212PRV(double *e, double *q)
{
    double ep[4];

    Euler3212EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Euler3232C(Q,C) returns the direction cosine
 * matrix in terms of the 3-2-3 Euler angles.
 * Input Q must be a 3x1 vector of Euler angles.
 */
void Euler3232C(double *q, double C[3][3])
{
    double st1;
    double st2;
    double st3;
    double ct1;
    double ct2;
    double ct3;

    st1 = sin(q[0]);
    ct1 = cos(q[0]);
    st2 = sin(q[1]);
    ct2 = cos(q[1]);
    st3 = sin(q[2]);
    ct3 = cos(q[2]);

    C[0][0] = ct1 * ct2 * ct3 - st1 * st3;
    C[0][1] = ct2 * ct3 * st1 + ct1 * st3;
    C[0][2] = -ct3 * st2;
    C[1][0] = -ct3 * st1 - ct1 * ct2 * st3;
    C[1][1] = ct1 * ct3 - ct2 * st1 * st3;
    C[1][2] = st2 * st3;
    C[2][0] = ct1 * st2;
    C[2][1] = st1 * st2;
    C[2][2] = ct2;
}

/*
 * Euler3232EP(E,Q) translates the 323 Euler angle
 * vector E into the Euler parameter vector Q.
 */
void Euler3232EP(double *e, double *q)
{
    double e1;
    double e2;
    double e3;

    e1 = e[0] / 2;
    e2 = e[1] / 2;
    e3 = e[2] / 2;

    q[0] = cos(e2) * cos(e1 + e3);
    q[1] = sin(e2) * sin(-e1 + e3);
    q[2] = sin(e2) * cos(-e1 + e3);
    q[3] = cos(e2) * sin(e1 + e3);
}

/*
 * Euler3232Gibbs(E,Q) translates the (3-2-3) Euler
 * angle vector E into the Gibbs vector Q.
 */
void Euler3232Gibbs(double *e, double *q)
{
    double ep[4];

    Euler3232EP(e, ep);
    EP2Gibbs(ep, q);
}

/*
 * Euler3232MRP(E,Q) translates the (3-2-3) Euler
 * angle vector E into the MRP vector Q.
 */
void Euler3232MRP(double *e, double *q)
{
    double ep[4];

    Euler3232EP(e, ep);
    EP2MRP(ep, q);
}

/*
 * Euler3232PRV(E,Q) translates the (3-2-3) Euler
 * angle vector Q1 into the principal rotation vector Q.
 */
void Euler3232PRV(double *e, double *q)
{
    double ep[4];

    Euler3232EP(e, ep);
    EP2PRV(ep, q);
}

/*
 * Gibbs2C(Q,C) returns the direction cosine
 * matrix in terms of the 3x1 Gibbs vector Q.
 */
void Gibbs2C(double *q, double C[3][3])
{
    double q1;
    double q2;
    double q3;
    double d1;

    q1 = q[0];
    q2 = q[1];
    q3 = q[2];

    d1 = v3Dot(q, q);
    C[0][0] = 1 + 2 * q1 * q1 - d1;
    C[0][1] = 2 * (q1 * q2 + q3);
    C[0][2] = 2 * (q1 * q3 - q2);
    C[1][0] = 2 * (q2 * q1 - q3);
    C[1][1] = 1 + 2 * q2 * q2 - d1;
    C[1][2] = 2 * (q2 * q3 + q1);
    C[2][0] = 2 * (q3 * q1 + q2);
    C[2][1] = 2 * (q3 * q2 - q1);
    C[2][2] = 1 + 2 * q3 * q3 - d1;
    m33Scale(1. / (1 + d1), C, C);
}

/*
 * Gibbs2EP(Q1,Q) translates the Gibbs vector Q1
 * into the Euler parameter vector Q.
 */
void Gibbs2EP(double *q1, double *q)
{
    q[0] = 1 / sqrt(1 + v3Dot(q1, q1));
    q[1] = q1[0] * q[0];
    q[2] = q1[1] * q[0];
    q[3] = q1[2] * q[0];
}

/*
 * Gibbs2Euler121(Q,E) translates the Gibbs
 * vector Q into the (1-2-1) Euler angle vector E.
 */
void Gibbs2Euler121(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler121(ep, e);
}

/*
 * Gibbs2Euler123(Q,E) translates the Gibbs
 * vector Q into the (1-2-3) Euler angle vector E.
 */
void Gibbs2Euler123(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler123(ep, e);
}

/*
 * Gibbs2Euler131(Q,E) translates the Gibbs
 * vector Q into the (1-3-1) Euler angle vector E.
 */
void Gibbs2Euler131(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler131(ep, e);
}

/*
 * Gibbs2Euler132(Q,E) translates the Gibbs
 * vector Q into the (1-3-2) Euler angle vector E.
 */
void Gibbs2Euler132(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler132(ep, e);
}

/*
 * Gibbs2Euler212(Q,E) translates the Gibbs
 * vector Q into the (2-1-2) Euler angle vector E.
 */
void Gibbs2Euler212(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler212(ep, e);
}

/*
 * Gibbs2Euler213(Q,E) translates the Gibbs
 * vector Q into the (2-1-3) Euler angle vector E.
 */
void Gibbs2Euler213(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler213(ep, e);
}

/*
 * Gibbs2Euler231(Q,E) translates the Gibbs
 * vector Q into the (2-3-1) Euler angle vector E.
 */
void Gibbs2Euler231(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler231(ep, e);
}

/*
 * Gibbs2Euler232(Q,E) translates the Gibbs
 * vector Q into the (2-3-2) Euler angle vector E.
 */
void Gibbs2Euler232(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler232(ep, e);
}

/*
 * Gibbs2Euler312(Q,E) translates the Gibbs
 * vector Q into the (3-1-2) Euler angle vector E.
 */
void Gibbs2Euler312(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler312(ep, e);
}

/*
 * Gibbs2Euler313(Q,E) translates the Gibbs
 * vector Q into the (3-1-3) Euler angle vector E.
 */
void Gibbs2Euler313(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler313(ep, e);
}

/*
 * Gibbs2Euler321(Q,E) translates the Gibbs
 * vector Q into the (3-2-1) Euler angle vector E.
 */
void Gibbs2Euler321(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler321(ep, e);
}

/*
 * Gibbs2Euler323(Q,E) translates the Gibbs
 * vector Q into the (3-2-3) Euler angle vector E.
 */
void Gibbs2Euler323(double *q, double *e)
{
    double ep[4];

    Gibbs2EP(q, ep);
    EP2Euler323(ep, e);
}

/*
 * Gibbs2MRP(Q1,Q) translates the Gibbs vector Q1
 * into the MRP vector Q.
 */
void Gibbs2MRP(double *q1, double *q)
{
    v3Scale(1.0 / (1 + sqrt(1 + v3Dot(q1, q1))), q1, q);
}

/*
 * Gibbs2PRV(Q1,Q) translates the Gibbs vector Q1
 * into the principal rotation vector Q.
 */
void Gibbs2PRV(double *q1, double *q)
{
    double tp;
    double p;

    tp = sqrt(v3Dot(q1, q1));
    p = 2 * atan(tp);
    if (tp < nearZero) {
        q[0] = 0.0;
        q[1] = 0.0;
        q[2] = 0.0;
        return;
    }
    q[0] = q1[0] / tp * p;
    q[1] = q1[1] / tp * p;
    q[2] = q1[2] / tp * p;
}

/*
 * MRP2C(Q,C) returns the direction cosine
 * matrix in terms of the 3x1 MRP vector Q.
 */
void MRP2C(double *q, double C[3][3])
{
    double q1;
    double q2;
    double q3;
    double S;
    double d1;
    double d;

    q1 = q[0];
    q2 = q[1];
    q3 = q[2];

    d1 = v3Dot(q, q);
    S = 1 - d1;
    d = (1 + d1) * (1 + d1);
    C[0][0] = 4 * (2 * q1 * q1 - d1) + S * S;
    C[0][1] = 8 * q1 * q2 + 4 * q3 * S;
    C[0][2] = 8 * q1 * q3 - 4 * q2 * S;
    C[1][0] = 8 * q2 * q1 - 4 * q3 * S;
    C[1][1] = 4 * (2 * q2 * q2 - d1) + S * S;
    C[1][2] = 8 * q2 * q3 + 4 * q1 * S;
    C[2][0] = 8 * q3 * q1 + 4 * q2 * S;
    C[2][1] = 8 * q3 * q2 - 4 * q1 * S;
    C[2][2] = 4 * (2 * q3 * q3 - d1) + S * S;
    m33Scale(1. / d, C, C);
}

/*
 * MRP2EP(Q1,Q) translates the MRP vector Q1
 * into the Euler parameter vector Q.
 */
void MRP2EP(double *q1, double *q)
{
    double ps;

    ps = 1 + v3Dot(q1, q1);
    q[0] = (1 - v3Dot(q1, q1)) / ps;
    q[1] = 2 * q1[0] / ps;
    q[2] = 2 * q1[1] / ps;
    q[3] = 2 * q1[2] / ps;
}

/*
 * MRP2Euler121(Q,E) translates the MRP
 * vector Q into the (1-2-1) Euler angle vector E.
 */
void MRP2Euler121(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler121(ep, e);
}

/*
 * MRP2Euler123(Q,E) translates the MRP
 * vector Q into the (1-2-3) Euler angle vector E.
 */
void MRP2Euler123(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler123(ep, e);
}

/*
 * MRP2Euler131(Q,E) translates the MRP
 * vector Q into the (1-3-1) Euler angle vector E.
 */
void MRP2Euler131(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler131(ep, e);
}

/*
 * MRP2Euler132(Q,E) translates the MRP
 * vector Q into the (1-3-2) Euler angle vector E.
 */
void MRP2Euler132(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler132(ep, e);
}

/*
 * E = MRP2Euler212(Q) translates the MRP
 * vector Q into the (2-1-2) Euler angle vector E.
 */
void MRP2Euler212(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler212(ep, e);
}

/*
 * MRP2Euler213(Q,E) translates the MRP
 * vector Q into the (2-1-3) Euler angle vector E.
 */
void MRP2Euler213(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler213(ep, e);
}

/*
 * MRP2Euler231(Q,E) translates the MRP
 * vector Q into the (2-3-1) Euler angle vector E.
 */
void MRP2Euler231(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler231(ep, e);
}

/*
 * MRP2Euler232(Q,E) translates the MRP
 * vector Q into the (2-3-2) Euler angle vector E.
 */
void MRP2Euler232(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler232(ep, e);
}

/*
 * MRP2Euler312(Q,E) translates the MRP
 * vector Q into the (3-1-2) Euler angle vector E.
 */
void MRP2Euler312(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler312(ep, e);
}

/*
 * MRP2Euler313(Q,E) translates the MRP
 * vector Q into the (3-1-3) Euler angle vector E.
 */
void MRP2Euler313(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler313(ep, e);
}

/*
 * MRP2Euler321(Q,E) translates the MRP
 * vector Q into the (3-2-1) Euler angle vector E.
 */
void MRP2Euler321(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler321(ep, e);
}

/*
 * MRP2Euler323(Q,E) translates the MRP
 * vector Q into the (3-2-3) Euler angle vector E.
 */
void MRP2Euler323(double *q, double *e)
{
    double ep[4];

    MRP2EP(q, ep);
    EP2Euler323(ep, e);
}

/*
 * MRP2Gibbs(Q1,Q) translates the MRP vector Q1
 * into the Gibbs vector Q.
 */
void MRP2Gibbs(double *q1, double *q)
{
    v3Scale(2. / (1. - v3Dot(q1, q1)), q1, q);
}

/*
 * MRP2PRV(Q1,Q) translates the MRP vector Q1
 * into the principal rotation vector Q.
 */
void MRP2PRV(double *q1, double *q)
{
    double tp;
    double p;

    tp = sqrt(v3Dot(q1, q1));
    if(tp < nearZero)
    {
        memset(q, 0x0, 3*sizeof(double));
        return;
    }
    p = 4 * atan(tp);
    q[0] = q1[0] / tp * p;
    q[1] = q1[1] / tp * p;
    q[2] = q1[2] / tp * p;
}

/*
 * MRPswitch(Q,s2,s) checks to see if v3Norm(Q) is larger than s2.
 * If yes, then the MRP vector Q is mapped to its shadow set.
 */
void MRPswitch(double *q, double s2, double *s)
{
    double q2;

    q2 = v3Dot(q, q);
    if(q2 > s2 * s2) {
        v3Scale(-1. / q2, q, s);
    } else {
        v3Copy(q, s);
    }
}

/*
 * MRPshadow forces a switch from the current MRP to its shadow set
 */
void MRPshadow(double *qIn, double *qOut)
{
    double q2;
    
    q2 = v3Dot(qIn, qIn);
    v3Scale(-1. / q2, qIn, qOut);
    return;
}

/*
 * Makes sure that the angle x lies within +/- Pi.
 */
double wrapToPi(double x)
{
    double q;

    q = x;

    if(x >  M_PI) {
        q = x - 2 * M_PI;
    }

    if(x < -M_PI) {
        q = x + 2 * M_PI;
    }

    return q;
}

/*
 * PRV2C(Q,C) returns the direction cosine
 * matrix in terms of the 3x1 principal rotation vector
 * Q.
 */
void PRV2C(double *q, double C[3][3])
{
    double q0;
    double q1;
    double q2;
    double q3;
    double cp;
    double sp;
    double d1;
    
    if(v3Norm(q) == 0.0)
    {
        m33SetIdentity(C);
        return;
    }
    
    q0 = sqrt(v3Dot(q, q));
    q1 = q[0] / q0;
    q2 = q[1] / q0;
    q3 = q[2] / q0;

    cp = cos(q0);
    sp = sin(q0);
    d1 = 1 - cp;
    C[0][0] = q1 * q1 * d1 + cp;
    C[0][1] = q1 * q2 * d1 + q3 * sp;
    C[0][2] = q1 * q3 * d1 - q2 * sp;
    C[1][0] = q2 * q1 * d1 - q3 * sp;
    C[1][1] = q2 * q2 * d1 + cp;
    C[1][2] = q2 * q3 * d1 + q1 * sp;
    C[2][0] = q3 * q1 * d1 + q2 * sp;
    C[2][1] = q3 * q2 * d1 - q1 * sp;
    C[2][2] = q3 * q3 * d1 + cp;
}

/*
 * PRV2elem(R,Q) translates a prinicpal rotation vector R
 * into the corresponding principal rotation element set Q.
 */
void PRV2elem(double *r, double *q)
{
    q[0] = sqrt(v3Dot(r, r));
	if (q[0] < 1.0E-12)
	{
		q[1] = q[2] = q[3] = 0.0;
	}
	else
	{
		q[1] = r[0] / q[0];
		q[2] = r[1] / q[0];
		q[3] = r[2] / q[0];
	}
}

/*
 * PRV2EP(Q0,Q) translates the principal rotation vector Q1
 * into the Euler parameter vector Q.
 */
void PRV2EP(double *q0, double *q)
{
    double q1[4];
    double sp;

    PRV2elem(q0, q1);
    sp = sin(q1[0] / 2);
    q[0] = cos(q1[0] / 2);
    q[1] = q1[1] * sp;
    q[2] = q1[2] * sp;
    q[3] = q1[3] * sp;
}

/*
 * PRV2Euler121(Q,E) translates the principal rotation
 * vector Q into the (1-2-1) Euler angle vector E.
 */
void PRV2Euler121(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler121(ep, e);
}

/*
 * PRV2Euler123(Q,E) translates the principal rotation
 * vector Q into the (1-2-3) Euler angle vector E.
 */
void PRV2Euler123(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler123(ep, e);
}

/*
 * PRV2Euler131(Q,E) translates the principal rotation
 * vector Q into the (1-3-1) Euler angle vector E.
 */
void PRV2Euler131(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler131(ep, e);
}

/*
 * PRV2Euler132(Q,E) translates the principal rotation
 * vector Q into the (1-3-2) Euler angle vector E.
 */
void PRV2Euler132(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler132(ep, e);
}

/*
 * PRV2Euler212(Q,E) translates the principal rotation
 * vector Q into the (2-1-2) Euler angle vector E.
 */
void PRV2Euler212(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler212(ep, e);
}

/*
 * PRV2Euler213(Q,E) translates the principal rotation
 * vector Q into the (2-1-3) Euler angle vector E.
 */
void PRV2Euler213(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler213(ep, e);
}

/*
 * PRV2Euler231(Q) translates the principal rotation
 * vector Q into the (2-3-1) Euler angle vector E.
 */
void PRV2Euler231(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler231(ep, e);
}

/*
 * PRV2Euler232(Q,E) translates the principal rotation
 * vector Q into the (2-3-2) Euler angle vector E.
 */
void PRV2Euler232(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler232(ep, e);
}

/*
 * PRV2Euler312(Q,E) translates the principal rotation
 * vector Q into the (3-1-2) Euler angle vector E.
 */
void PRV2Euler312(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler312(ep, e);
}

/*
 * PRV2Euler313(Q,E) translates the principal rotation
 * vector Q into the (3-1-3) Euler angle vector E.
 */
void PRV2Euler313(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler313(ep, e);
}

/*
 * PRV2Euler321(Q,E) translates the principal rotation
 * vector Q into the (3-2-1) Euler angle vector E.
 */
void PRV2Euler321(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler321(ep, e);
}

/*
 * PRV2Euler323(Q,E) translates the principal rotation
 * vector Q into the (3-2-3) Euler angle vector E.
 */
void PRV2Euler323(double *q, double *e)
{
    double ep[4];

    PRV2EP(q, ep);
    EP2Euler323(ep, e);
}

/*
 * PRV2Gibbs(Q0,Q) translates the principal rotation vector Q1
 * into the Gibbs vector Q.
 */
void PRV2Gibbs(double *q0, double *q)
{
    double q1[4];
    double tp;

    PRV2elem(q0, q1);
    tp = tan(q1[0] / 2.);
    q[0] = q1[1] * tp;
    q[1] = q1[2] * tp;
    q[2] = q1[3] * tp;
}

/*
 * PRV2MRP(Q0,Q) translates the principal rotation vector Q1
 * into the MRP vector Q.
 */
void PRV2MRP(double *q0, double *q)
{
    double q1[4];
    double tp;

    PRV2elem(q0, q1);
    tp = tan(q1[0] / 4.);
    q[0] = q1[1] * tp;
    q[1] = q1[2] * tp;
    q[2] = q1[3] * tp;
}

/*
 * subEP(B1,B2,Q) provides the Euler parameter vector
 * which corresponds to relative rotation from B2
 * to B1.
 */
void subEP(double *b1, double *b2, double *q)
{
    q[0] = b2[0] * b1[0] + b2[1] * b1[1] + b2[2] * b1[2] + b2[3] * b1[3];
    q[1] = -b2[1] * b1[0] + b2[0] * b1[1] + b2[3] * b1[2] - b2[2] * b1[3];
    q[2] = -b2[2] * b1[0] - b2[3] * b1[1] + b2[0] * b1[2] + b2[1] * b1[3];
    q[3] = -b2[3] * b1[0] + b2[2] * b1[1] - b2[1] * b1[2] + b2[0] * b1[3];
}

/*
 * subEuler121(E,E1,E2) computes the relative
 * (1-2-1) Euler angle vector from E1 to E.
 */
void subEuler121(double *e, double *e1, double *e2)
{
    double cp;
    double cp1;
    double sp;
    double sp1;
    double cp2;
    double dum;

    cp = cos(e[1]);
    cp1 = cos(e1[1]);
    sp = sin(e[1]);
    sp1 = sin(e1[1]);
    dum = e[0] - e1[0];

    e2[1] = safeAcos(cp1 * cp + sp1 * sp * cos(dum));
    cp2 = cos(e2[1]);
    e2[0] = wrapToPi(-e1[2] + atan2(sp1 * sp * sin(dum), cp2 * cp1 - cp));
    e2[2] = wrapToPi(e[2] - atan2(sp1 * sp * sin(dum), cp1 - cp * cp2));
}

/*
 * subEuler123(E,E1,E2) computes the relative
 * (1-2-3) Euler angle vector from E1 to E.
 */
void subEuler123(double *e, double *e1, double *e2)
{
    double C[3][3];
    double C1[3][3];
    double C2[3][3];

    Euler1232C(e, C);
    Euler1232C(e1, C1);
    m33MultM33t(C, C1, C2);
    C2Euler123(C2, e2);
}

/*
 * subEuler131(E,E1,E2) computes the relative
 * (1-3-1) Euler angle vector from E1 to E.
 */
void subEuler131(double *e, double *e1, double *e2)
{
    double cp;
    double cp1;
    double sp;
    double sp1;
    double dum;
    double cp2;

    cp = cos(e[1]);
    cp1 = cos(e1[1]);
    sp = sin(e[1]);
    sp1 = sin(e1[1]);
    dum = e[0] - e1[0];

    e2[1] = safeAcos(cp1 * cp + sp1 * sp * cos(dum));
    cp2 = cos(e2[1]);
    e2[0] = wrapToPi(-e1[2] + atan2(sp1 * sp * sin(dum), cp2 * cp1 - cp));
    e2[2] = wrapToPi(e[2] - atan2(sp1 * sp * sin(dum), cp1 - cp * cp2));
}

/*
 * subEuler132(E,E1,E2) computes the relative
 * (1-3-2) Euler angle vector from E1 to E.
 */
void subEuler132(double *e, double *e1, double *e2)
{
    double C[3][3];
    double C1[3][3];
    double C2[3][3];

    Euler1322C(e, C);
    Euler1322C(e1, C1);
    m33MultM33t(C, C1, C2);
    C2Euler132(C2, e2);
}

/*
 * subEuler212(E,E1,E2) computes the relative
 * (2-1-2) Euler angle vector from E1 to E.
 */
void subEuler212(double *e, double *e1, double *e2)
{
    double cp;
    double cp1;
    double sp;
    double sp1;
    double dum;
    double cp2;

    cp = cos(e[1]);
    cp1 = cos(e1[1]);
    sp = sin(e[1]);
    sp1 = sin(e1[1]);
    dum = e[0] - e1[0];

    e2[1] = safeAcos(cp1 * cp + sp1 * sp * cos(dum));
    cp2 = cos(e2[1]);
    e2[0] = wrapToPi(-e1[2] + atan2(sp1 * sp * sin(dum), cp2 * cp1 - cp));
    e2[2] = wrapToPi(e[2] - atan2(sp1 * sp * sin(dum), cp1 - cp * cp2));
}

/*
 * subEuler213(E,E1,E2) computes the relative
 * (2-1-3) Euler angle vector from E1 to E.
 */
void subEuler213(double *e, double *e1, double *e2)
{
    double C[3][3];
    double C1[3][3];
    double C2[3][3];

    Euler2132C(e, C);
    Euler2132C(e1, C1);
    m33MultM33t(C, C1, C2);
    C2Euler213(C2, e2);
}

/*
 * subEuler231(E,E1,E2) computes the relative
 * (2-3-1) Euler angle vector from E1 to E.
 */
void subEuler231(double *e, double *e1, double *e2)
{
    double C[3][3];
    double C1[3][3];
    double C2[3][3];

    Euler2312C(e, C);
    Euler2312C(e1, C1);
    m33MultM33t(C, C1, C2);
    C2Euler231(C2, e2);
}

/*
 * subEuler232(E,E1,E2) computes the relative
 * (2-3-2) Euler angle vector from E1 to E.
 */
void subEuler232(double *e, double *e1, double *e2)
{
    double cp;
    double cp1;
    double sp;
    double sp1;
    double dum;
    double cp2;

    cp = cos(e[1]);
    cp1 = cos(e1[1]);
    sp = sin(e[1]);
    sp1 = sin(e1[1]);
    dum = e[0] - e1[0];

    e2[1] = safeAcos(cp1 * cp + sp1 * sp * cos(dum));
    cp2 = cos(e2[1]);
    e2[0] = wrapToPi(-e1[2] + atan2(sp1 * sp * sin(dum), cp2 * cp1 - cp));
    e2[2] = wrapToPi(e[2] - atan2(sp1 * sp * sin(dum), cp1 - cp * cp2));
}

/*
 * subEuler312(E,E1,E2) computes the relative
 * (3-1-2) Euler angle vector from E1 to E.
 */
void subEuler312(double *e, double *e1, double *e2)
{
    double C[3][3];
    double C1[3][3];
    double C2[3][3];

    Euler3122C(e, C);
    Euler3122C(e1, C1);
    m33MultM33t(C, C1, C2);
    C2Euler312(C2, e2);
}

/*
 * subEuler313(E,E1,E2) computes the relative
 * (3-1-3) Euler angle vector from E1 to E.
 */
void subEuler313(double *e, double *e1, double *e2)
{
    double cp;
    double cp1;
    double sp;
    double sp1;
    double dum;
    double cp2;

    cp = cos(e[1]);
    cp1 = cos(e1[1]);
    sp = sin(e[1]);
    sp1 = sin(e1[1]);
    dum = e[0] - e1[0];

    e2[1] = safeAcos(cp1 * cp + sp1 * sp * cos(dum));
    cp2 = cos(e2[1]);
    e2[0] = wrapToPi(-e1[2] + atan2(sp1 * sp * sin(dum), cp2 * cp1 - cp));
    e2[2] = wrapToPi(e[2] - atan2(sp1 * sp * sin(dum), cp1 - cp * cp2));
}

/*
 * subEuler321(E,E1,E2) computes the relative
 * (3-2-1) Euler angle vector from E1 to E.
 */
void subEuler321(double *e, double *e1, double *e2)
{
    double C[3][3];
    double C1[3][3];
    double C2[3][3];

    Euler3212C(e, C);
    Euler3212C(e1, C1);
    m33MultM33t(C, C1, C2);
    C2Euler321(C2, e2);
}

/*
 * subEuler323(E,E1,E2) computes the relative
 * (3-2-3) Euler angle vector from E1 to E.
 */
void subEuler323(double *e, double *e1, double *e2)
{
    double cp;
    double cp1;
    double sp;
    double sp1;
    double dum;
    double cp2;

    cp = cos(e[1]);
    cp1 = cos(e1[1]);
    sp = sin(e[1]);
    sp1 = sin(e1[1]);
    dum = e[0] - e1[0];

    e2[1] = safeAcos(cp1 * cp + sp1 * sp * cos(dum));
    cp2 = cos(e2[1]);
    e2[0] = wrapToPi(-e1[2] + atan2(sp1 * sp * sin(dum), cp2 * cp1 - cp));
    e2[2] = wrapToPi(e[2] - atan2(sp1 * sp * sin(dum), cp1 - cp * cp2));
}

/*
 * subGibbs(Q1,Q2,Q) provides the Gibbs vector
 * which corresponds to relative rotation from Q2
 * to Q1.
 */
void subGibbs(double *q1, double *q2, double *q)
{
    double d1[3];

    v3Cross(q1, q2, d1);
    v3Add(q1, d1, q);
    v3Subtract(q, q2, q);
    v3Scale(1. / (1. + v3Dot(q1, q2)), q, q);
}

/*
 * subMRP(Q1,Q2,Q) provides the MRP vector
 * which corresponds to relative rotation from Q2
 * to Q1.
 */
void subMRP(double *q1, double *q2, double *q)
{
    double d1[3];
    double s1[3];
    double det;
    double mag;

    v3Copy(q1, s1);
    det = (1. + v3Dot(s1, s1)*v3Dot(q2, q2) + 2.*v3Dot(s1, q2));
    if (fabs(det) < 0.1) {
        mag = v3Dot(s1, s1);
        v3Scale(-1.0/mag, s1, s1);
        det = (1. + v3Dot(s1, s1)*v3Dot(q2, q2) + 2.*v3Dot(s1, q2));
    }

    v3Cross(s1, q2, d1);
    v3Scale(2., d1, q);
    v3Scale(1. - v3Dot(q2, q2), s1, d1);
    v3Add(q, d1, q);
    v3Scale(1. - v3Dot(s1, s1), q2, d1);
    v3Subtract(q, d1, q);
    v3Scale(1. / det, q, q);

    /* map MRP to inner set */
    mag = v3Dot(q, q);
    if (mag > 1.0){
        v3Scale(-1./mag, q, q);
    }

}

/*
 * subPRV(Q1,Q2,Q) provides the prinipal rotation vector
 * which corresponds to relative principal rotation from Q2
 * to Q1.
 */
void subPRV(double *q10, double *q20, double *q)
{
    double q1[4];
    double q2[4];
    double cp1;
    double cp2;
    double sp1;
    double sp2;
    double e1[3];
    double e2[3];
    double p;
    double sp;

    PRV2elem(q10, q1);
    PRV2elem(q20, q2);
    cp1 = cos(q1[0] / 2.);
    cp2 = cos(q2[0] / 2.);
    sp1 = sin(q1[0] / 2.);
    sp2 = sin(q2[0] / 2.);
    v3Copy(&(q1[1]), e1);
    v3Copy(&(q2[1]), e2);

    p = 2.*safeAcos(cp1 * cp2 + sp1 * sp2 * v3Dot(e1, e2));
    sp = sin(p / 2.);

    v3Cross(e1, e2, q1);
    v3Scale(sp1 * sp2, q1, q);
    v3Scale(cp2 * sp1, e1, q1);
    v3Add(q1, q, q);
    v3Scale(cp1 * sp2, e2, q1);
    v3Subtract(q, q1, q);
    v3Scale(p / sp, q, q);
}

/*
 * Mi(theta, a, C) returns the rotation matrix corresponding
 * to a single axis rotation about axis a by the angle theta
 */
void Mi(double theta, int a, double C[3][3])
{
    double c;
    double s;

    c = cos(theta);
    s = sin(theta);

    switch(a) {
        case 1:
            C[0][0] = 1.;
            C[0][1] = 0.;
            C[0][2] = 0.;
            C[1][0] = 0.;
            C[1][1] =  c;
            C[1][2] =  s;
            C[2][0] = 0.;
            C[2][1] = -s;
            C[2][2] =  c;
            break;

        case 2:
            C[0][0] =  c;
            C[0][1] = 0.;
            C[0][2] = -s;
            C[1][0] = 0.;
            C[1][1] = 1.;
            C[1][2] = 0.;
            C[2][0] =  s;
            C[2][1] = 0.;
            C[2][2] =  c;
            break;

        case 3:
            C[0][0] =  c;
            C[0][1] =  s;
            C[0][2] = 0.;
            C[1][0] = -s;
            C[1][1] =  c;
            C[1][2] = 0.;
            C[2][0] = 0.;
            C[2][1] = 0.;
            C[2][2] = 1.;
            break;

        default:
            BSK_PRINT(MSG_ERROR, "Mi() error: incorrect axis %d selected.", a);
    }
}

/*
 * tilde(theta, mat) returns the the 3x3 cross product matrix
 */
void   tilde(double *v, double mat[3][3])
{
    m33SetZero(mat);
    mat[0][1] = -v[2];
    mat[1][0] = v[2];
    mat[0][2] = v[1];
    mat[2][0] = -v[1];
    mat[1][2] = -v[0];
    mat[2][1] = v[0];

    return;
}
