/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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


#include "oneAxisSolarArrayPoint.h"
#include "string.h"
#include <math.h>

#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/macroDefinitions.h"

const double epsilon = 1e-12;                           // module tolerance for zero

/*! This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_oneAxisSolarArrayPoint(OneAxisSolarArrayPointConfig *configData, int64_t moduleID)
{
    AttRefMsg_C_init(&configData->attRefOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_oneAxisSolarArrayPoint(OneAxisSolarArrayPointConfig *configData, uint64_t callTime, int64_t moduleID)
{
    if (!NavAttMsg_C_isLinked(&configData->attNavInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, " oneAxisSolarArrayPoint.attNavInMsg wasn't connected.");
    }

    // check how the input body heading is provided
    if (BodyHeadingMsg_C_isLinked(&configData->bodyHeadingInMsg)) {
        configData->bodyAxisInput = inputBodyHeadingMsg;
    }
    else if (v3Norm(configData->h1Hat_B) > epsilon) {
            configData->bodyAxisInput = inputBodyHeadingParameter;
    }
    else {
            _bskLog(configData->bskLogger, BSK_ERROR, " oneAxisSolarArrayPoint.bodyHeadingInMsg wasn't connected and no body heading h1Hat_B was specified.");
    }

    // check how the input inertial heading is provided
    if (InertialHeadingMsg_C_isLinked(&configData->inertialHeadingInMsg)) {
        configData->inertialAxisInput = inputInertialHeadingMsg;
        if (EphemerisMsg_C_isLinked(&configData->ephemerisInMsg)) {
            _bskLog(configData->bskLogger, BSK_WARNING, " both oneAxisSolarArrayPoint.inertialHeadingInMsg and oneAxisSolarArrayPoint.ephemerisInMsg were linked. Inertial heading is computed from oneAxisSolarArrayPoint.inertialHeadingInMsg");
        }
    }
    else if (EphemerisMsg_C_isLinked(&configData->ephemerisInMsg)) {
        if (!NavTransMsg_C_isLinked(&configData->transNavInMsg)) {
            _bskLog(configData->bskLogger, BSK_ERROR, " oneAxisSolarArrayPoint.ephemerisInMsg was specified but oneAxisSolarArrayPoint.transNavInMsg was not.");
        }
        else {
            configData->inertialAxisInput = inputEphemerisMsg;
        }
    }
    else {
        if (v3Norm(configData->hHat_N) > epsilon) {
            configData->inertialAxisInput = inputInertialHeadingParameter;
        }
        else {
            _bskLog(configData->bskLogger, BSK_ERROR, " neither oneAxisSolarArrayPoint.inertialHeadingInMsg nor oneAxisSolarArrayPoint.ephemerisInMsg were connected and no inertial heading h_N was specified.");
        }
    }

    // set updateCallCount to zero
    configData->updateCallCount = 0;
}

/*! The Update() function computes the reference MRP attitude, reference angular rate and acceleration
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_oneAxisSolarArrayPoint(OneAxisSolarArrayPointConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! create and zero the output message */
    AttRefMsgPayload attRefOut = AttRefMsg_C_zeroMsgPayload();

    /*! read and allocate the attitude navigation message */
    NavAttMsgPayload attNavIn = NavAttMsg_C_read(&configData->attNavInMsg);

    /*! get requested heading in inertial frame */
    double hReqHat_N[3];
    if (configData->inertialAxisInput == inputInertialHeadingParameter) {
        v3Normalize(configData->hHat_N, hReqHat_N);
    }
    else if (configData->inertialAxisInput == inputInertialHeadingMsg) {
        InertialHeadingMsgPayload inertialHeadingIn = InertialHeadingMsg_C_read(&configData->inertialHeadingInMsg);
        v3Normalize(inertialHeadingIn.rHat_XN_N, hReqHat_N);
    }
    else if (configData->inertialAxisInput == inputEphemerisMsg) {
        EphemerisMsgPayload ephemerisIn = EphemerisMsg_C_read(&configData->ephemerisInMsg);
        NavTransMsgPayload transNavIn = NavTransMsg_C_read(&configData->transNavInMsg);
        v3Subtract(ephemerisIn.r_BdyZero_N, transNavIn.r_BN_N, hReqHat_N);
        v3Normalize(hReqHat_N, hReqHat_N);
    }

    /*! get body frame heading */
    double hRefHat_B[3];
    if (configData->bodyAxisInput == inputBodyHeadingParameter) {
        v3Normalize(configData->h1Hat_B, hRefHat_B);
    }
    else if (configData->bodyAxisInput == inputBodyHeadingMsg) {
        BodyHeadingMsgPayload bodyHeadingIn = BodyHeadingMsg_C_read(&configData->bodyHeadingInMsg);
        v3Normalize(bodyHeadingIn.rHat_XB_B, hRefHat_B);
    }
    
    /*! define the body frame orientation DCM BN */
    double BN[3][3];
    MRP2C(attNavIn.sigma_BN, BN);

    /*! get the solar array drive direction in body frame coordinates */
    double a1Hat_B[3];
    v3Normalize(configData->a1Hat_B, a1Hat_B);

    /*! get the second body frame direction */
    double a2Hat_B[3];
    if (v3Norm(configData->a2Hat_B) > epsilon) {
        v3Normalize(configData->a2Hat_B, a2Hat_B);
    }
    else {
        v3SetZero(a2Hat_B);
    }

    /*! read Sun direction in B frame from the attNav message */
    double rHat_SB_B[3];
    v3Copy(attNavIn.vehSunPntBdy, rHat_SB_B);

    /*! map requested heading into B frame */
    double hReqHat_B[3];
    m33MultV3(BN, hReqHat_N, hReqHat_B);

    /*! compute the total rotation DCM */
    double RN[3][3];
    oasapComputeFinalRotation(configData->alignmentPriority, BN, rHat_SB_B, hRefHat_B, hReqHat_B, a1Hat_B, a2Hat_B, RN);

    /*! compute the relative rotation DCM and Sun direction in relative frame */
    double RB[3][3];
    m33MultM33t(RN, BN, RB);
    double rHat_SB_R1[3];
    m33MultV3(RB, rHat_SB_B, rHat_SB_R1);

    /*! compute reference MRP */
    double sigma_RN[3];
    C2MRP(RN, sigma_RN);

    if (v3Norm(configData->h2Hat_B) > epsilon) {
        // compute second reference frame
        oasapComputeFinalRotation(configData->alignmentPriority, BN, rHat_SB_B, configData->h2Hat_B, hReqHat_B, a1Hat_B, a2Hat_B, RN);
        
        // compute the relative rotation DCM and Sun direction in relative frame
        m33MultM33t(RN, BN, RB);
        double rHat_SB_R2[3];
        m33MultV3(RB, rHat_SB_B, rHat_SB_R2);

        double dotP_1 = v3Dot(rHat_SB_R1, a2Hat_B);
        double dotP_2 = v3Dot(rHat_SB_R2, a2Hat_B);
        if (dotP_2 > dotP_1 && fabs(dotP_2 - dotP_1) > epsilon) {
            // if second reference frame gives a better result, save this as reference MRP set
            C2MRP(RN, sigma_RN);
        }
    }
    v3Copy(sigma_RN, attRefOut.sigma_RN);

    /*! compute reference MRP derivatives via finite differences */
    // read sigma at t-1 and switch it if needed
    double sigma_RN_1[3];
    v3Copy(configData->sigma_RN_1, sigma_RN_1);
    double delSigma[3];
    v3Subtract(sigma_RN, sigma_RN_1, delSigma);
    if (v3Norm(delSigma) > 1) {
        MRPshadow(sigma_RN_1, sigma_RN_1);
    }
    // read sigma at t-2 and switch it if needed
    double sigma_RN_2[3];
    v3Copy(configData->sigma_RN_2, sigma_RN_2);
    v3Subtract(sigma_RN_1, sigma_RN_2, delSigma);
    if (v3Norm(delSigma) > 1) {
        MRPshadow(sigma_RN_2, sigma_RN_2);
    }
    // if first update call, derivatives are set to zero
    double T1Seconds;
    double T2Seconds;
    double sigmaDot_RN[3];
    double sigmaDDot_RN[3];
    if (configData->updateCallCount == 0) {
        v3SetZero(sigmaDot_RN);
        v3SetZero(sigmaDDot_RN);
        // store information for next time step
        configData->T1NanoSeconds = callTime;
        v3Copy(sigma_RN, configData->sigma_RN_1);
    }
    // if second update call, derivatives are computed with first order finite differences
    else if (configData->updateCallCount == 1) {
        T1Seconds = (configData->T1NanoSeconds - callTime) * NANO2SEC;
        for (int j = 0; j < 3; j++) {
            sigmaDot_RN[j] = (sigma_RN_1[j] - sigma_RN[j]) / T1Seconds;
        }
        v3SetZero(sigmaDDot_RN);
        // store information for next time step
        configData->T2NanoSeconds = configData->T1NanoSeconds;
        configData->T1NanoSeconds = callTime;
        v3Copy(configData->sigma_RN_1, configData->sigma_RN_2);
        v3Copy(sigma_RN, configData->sigma_RN_1);
    }
    // if third update call or higher, derivatives are computed with second order finite differences
    else {
        T1Seconds = (configData->T1NanoSeconds - callTime) * NANO2SEC;
        T2Seconds = (configData->T2NanoSeconds - callTime) * NANO2SEC;
        for (int j = 0; j < 3; j++) {
            sigmaDot_RN[j] = ((sigma_RN_1[j]*T2Seconds*T2Seconds - sigma_RN_2[j]*T1Seconds*T1Seconds) / (T2Seconds - T1Seconds) - sigma_RN[j] * (T2Seconds + T1Seconds)) / T1Seconds / T2Seconds;
            sigmaDDot_RN[j] = 2 * ((sigma_RN_1[j]*T2Seconds - sigma_RN_2[j]*T1Seconds) / (T1Seconds - T2Seconds) + sigma_RN[j]) / T1Seconds / T2Seconds;
        }
        // store information for next time step
        configData->T2NanoSeconds = configData->T1NanoSeconds;
        configData->T1NanoSeconds = callTime;
        v3Copy(configData->sigma_RN_1, configData->sigma_RN_2);
        v3Copy(sigma_RN, configData->sigma_RN_1);
    }
    configData->updateCallCount += 1;

    /*! compute angular rates and accelerations in R frame */
    double omega_RN_R[3], omegaDot_RN_R[3];
    dMRP2Omega(sigma_RN, sigmaDot_RN, omega_RN_R);
    ddMRP2dOmega(sigma_RN, sigmaDot_RN, sigmaDDot_RN, omegaDot_RN_R);

    /*! compute angular rates and accelerations in N frame and store in buffer msg */
    m33tMultV3(RN, omega_RN_R, attRefOut.omega_RN_N);
    m33tMultV3(RN, omegaDot_RN_R, attRefOut.domega_RN_N);

    /*! write output message */
    AttRefMsg_C_write(&attRefOut, &configData->attRefOutMsg, moduleID, callTime);
}

/*! This helper function computes the first rotation that aligns the body heading with the inertial heading */
void oasapComputeFirstRotation(double hRefHat_B[3], double hReqHat_B[3], double R1B[3][3])
{
    /*! compute principal rotation angle (phi) and vector (e_phi) for the first rotation */
    double phi = acos( fmin( fmax( v3Dot(hRefHat_B, hReqHat_B), -1 ), 1 ) );
    double e_phi[3];
    v3Cross(hRefHat_B, hReqHat_B, e_phi);
    // If phi = PI, e_phi can be any vector perpendicular to hRefHat_B
    if (fabs(phi-MPI) < epsilon) {
        phi = MPI;
        v3Perpendicular(hRefHat_B, e_phi);
    }
    else if (fabs(phi) < epsilon) {
        phi = 0;
    }
    // normalize e_phi
    v3Normalize(e_phi, e_phi);

    /*! define first rotation R1B */
    double PRV_phi[3];
    v3Scale(phi, e_phi, PRV_phi);
    PRV2C(PRV_phi, R1B);
}

/*! This helper function computes the second rotation that achieves the best incidence on the solar arrays maintaining the heading alignment */
void oasapComputeSecondRotation(double hRefHat_B[3], double rHat_SB_R1[3], double a1Hat_B[3], double a2Hat_B[3], double R2R1[3][3])
{
    /*! define second rotation vector to coincide with the thrust direction in B coordinates */
    double e_psi[3];
    v3Copy(hRefHat_B, e_psi);

    /*! define the coefficients of the quadratic equation A, B and C */
    double b3[3];
    v3Cross(rHat_SB_R1, e_psi, b3);
    double A = 2 * v3Dot(rHat_SB_R1, e_psi) * v3Dot(e_psi, a1Hat_B) - v3Dot(a1Hat_B, rHat_SB_R1);
    double B = 2 * v3Dot(a1Hat_B, b3);
    double C = v3Dot(a1Hat_B, rHat_SB_R1);
    double Delta = B * B - 4 * A * C;

    /*! get the body direction that must be kept close to Sun and compute the coefficients of the quadratic equation E, F and G */
    double E = 2 * v3Dot(rHat_SB_R1, e_psi) * v3Dot(e_psi, a2Hat_B) - v3Dot(a2Hat_B, rHat_SB_R1);
    double F = 2 * v3Dot(a2Hat_B, b3);
    double G = v3Dot(a2Hat_B, rHat_SB_R1);

    /*! compute exact solution or best solution depending on Delta */
    double t, t1, t2, y, y1, y2, psi;
    if (fabs(A) < epsilon) {
        if (fabs(B) < epsilon) {
            // zero-th order equation has no solution 
            // the solution of the minimum problem is psi = MPI
            psi = MPI;
        }
        else {
            // first order equation
            t = - C / B;
            psi = 2*atan(t);
        }
    }
    else {
        if (Delta < 0) {
            // second order equation has no solution 
            // the solution of the minimum problem is found
            if (fabs(B) < epsilon) {
                t = 0.0;
            }
            else {
                double q = (A-C) / B;
                t1 = (q + sqrt(q*q + 1));
                t2 = (q - sqrt(q*q + 1));
                y1 = (A*t1*t1 + B*t1 + C) / (1 + t1*t1);
                y2 = (A*t2*t2 + B*t2 + C) / (1 + t2*t2);

                // choose which returns a smaller fcn value between t1 and t2
                t = t1;
                if (fabs(y2) < fabs(y1)) {
                    t = t2;
                }
            }
            psi = 2*atan(t);
            y = (A*t*t + B*t + C) / (1 + t*t);
            // check if the absolute fcn minimum is for psi = MPI
            if (fabs(A) < fabs(y)) {
                psi = MPI;
            }
        }
        else {
            // solution of the quadratic equation
            t1 = (-B + sqrt(Delta)) / (2*A);
            t2 = (-B - sqrt(Delta)) / (2*A);

            // choose between t1 and t2 according to a2Hat
            t = t1;            
            if (fabs(v3Dot(hRefHat_B, a2Hat_B)-1) > epsilon) {
                y1 = (E*t1*t1 + F*t1 + G) / (1 + t1*t1);
                y2 = (E*t2*t2 + F*t2 + G) / (1 + t2*t2);
                if (y2 - y1 > epsilon) {
                    t = t2;
                }
            }
            psi = 2*atan(t);
        }
    }

    /*! compute second rotation R2R1 */
    double PRV_psi[3];
    v3Scale(psi, e_psi, PRV_psi);
    PRV2C(PRV_psi, R2R1);
}

/*! This helper function computes the third rotation that breaks the heading alignment if needed, to achieve maximum incidence on solar arrays */
void oasapComputeThirdRotation(int alignmentPriority, double hRefHat_B[3], double rHat_SB_R2[3], double a1Hat_B[3], double R3R2[3][3])
{
    double PRV_theta[3];

    if (alignmentPriority == prioritizeAxisAlignment) {
        for (int i = 0; i < 3; i++) {
            PRV_theta[i] = 0;
        }
    }
    else {
        double sTheta = v3Dot(rHat_SB_R2, a1Hat_B);
        double theta = asin( fmin( fmax( fabs(sTheta), -1 ), 1 ) );
        if (fabs(theta) < epsilon) {
            // if Sun direction and solar array drive are already perpendicular, third rotation is null
            for (int i = 0; i < 3; i++) {
            PRV_theta[i] = 0;
            }
        }
        else {
            // if Sun direction and solar array drive are not perpendicular, project solar array drive a1Hat_B onto perpendicular plane (aPHat_B) and compute third rotation
            double e_theta[3], aPHat_B[3];
            if (fabs(fabs(theta)-MPI/2) > epsilon) {
                for (int i = 0; i < 3; i++) {
                    aPHat_B[i] = (a1Hat_B[i] - sTheta * rHat_SB_R2[i]) / (1 - sTheta * sTheta);
                }
                v3Cross(a1Hat_B, aPHat_B, e_theta);
            }
            else {
                // rotate about the axis that minimizes variation in hRefHat_B direction
                v3Cross(rHat_SB_R2, hRefHat_B, aPHat_B);
                if (v3Norm(aPHat_B) < epsilon) {
                    v3Perpendicular(rHat_SB_R2, aPHat_B);
                }
                v3Cross(a1Hat_B, aPHat_B, e_theta);
            }
            v3Normalize(e_theta, e_theta);
            v3Scale(theta, e_theta, PRV_theta);
        }
    }

    /*! compute third rotation R3R2 */
    PRV2C(PRV_theta, R3R2);
}

/*! This helper function computes the final rotation as a product of the first three DCMs */
void oasapComputeFinalRotation(int alignmentPriority, double BN[3][3], double rHat_SB_B[3], double hRefHat_B[3], double hReqHat_B[3], double a1Hat_B[3], double a2Hat_B[3], double RN[3][3])
{
    /*! compute the first rotation DCM */
    double R1B[3][3];
    oasapComputeFirstRotation(hRefHat_B, hReqHat_B, R1B);

    /*! compute Sun direction vector in R1 frame coordinates */
    double rHat_SB_R1[3];
    m33MultV3(R1B, rHat_SB_B, rHat_SB_R1);

    /*! compute the second rotation DCM */
    double R2R1[3][3];
    oasapComputeSecondRotation(hRefHat_B, rHat_SB_R1, a1Hat_B, a2Hat_B, R2R1);

    /* compute Sun direction in R2 frame components */
    double rHat_SB_R2[3];
    m33MultV3(R2R1, rHat_SB_R1, rHat_SB_R2);

    /*! compute the third rotation DCM */
    double R3R2[3][3];
    oasapComputeThirdRotation(alignmentPriority, hRefHat_B, rHat_SB_R2, a1Hat_B, R3R2);

    /*! compute reference frames w.r.t inertial frame */
    double R1N[3][3], R2N[3][3];
    m33MultM33(R1B, BN, R1N);
    m33MultM33(R2R1, R1N, R2N);
    m33MultM33(R3R2, R2N, RN);
}

