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

#include "thrusterPlatformReference.h"
#include <math.h>

#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/astroConstants.h"

const double epsilon = 1e-12;                           // module tolerance for zero

/*! This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_thrusterPlatformReference(ThrusterPlatformReferenceConfig *configData, int64_t moduleID)
{
    HingedRigidBodyMsg_C_init(&configData->hingedRigidBodyRef1OutMsg);
    HingedRigidBodyMsg_C_init(&configData->hingedRigidBodyRef2OutMsg);
    BodyHeadingMsg_C_init(&configData->bodyHeadingOutMsg);
    CmdTorqueBodyMsg_C_init(&configData->thrusterTorqueOutMsg);
    THRConfigMsg_C_init(&configData->thrusterConfigBOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_thrusterPlatformReference(ThrusterPlatformReferenceConfig *configData, uint64_t callTime, int64_t moduleID)
{
    if (!VehicleConfigMsg_C_isLinked(&configData->vehConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, " thrusterPlatformReference.vehConfigInMsg wasn't connected.");
    }
    if (!THRConfigMsg_C_isLinked(&configData->thrusterConfigFInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, " thrusterPlatformReference.thrusterConfigFInMsg wasn't connected.");
    }
    if (RWArrayConfigMsg_C_isLinked(&configData->rwConfigDataInMsg) && RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)) {
        configData->momentumDumping = Yes;

        /*! - read in the RW configuration message */
        configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwConfigDataInMsg);
    }
    else {
        configData->momentumDumping = No;
    }

    /*! set the RW momentum integral to zero */
    v3SetZero(configData->hsInt_M);
    v3SetZero(configData->priorHs_M);
    configData->priorTime = callTime;
}


/*! This method updates the platformAngles message based on the updated information about the system center of mass
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_thrusterPlatformReference(ThrusterPlatformReferenceConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! - Create and assign message buffers */
    VehicleConfigMsgPayload    vehConfigMsgIn = VehicleConfigMsg_C_read(&configData->vehConfigInMsg);
    THRConfigMsgPayload        thrusterConfigFIn = THRConfigMsg_C_read(&configData->thrusterConfigFInMsg);
    HingedRigidBodyMsgPayload  hingedRigidBodyRef1Out = HingedRigidBodyMsg_C_zeroMsgPayload();
    HingedRigidBodyMsgPayload  hingedRigidBodyRef2Out = HingedRigidBodyMsg_C_zeroMsgPayload();
    BodyHeadingMsgPayload      bodyHeadingOut = BodyHeadingMsg_C_zeroMsgPayload();
    CmdTorqueBodyMsgPayload    thrusterTorqueOut = CmdTorqueBodyMsg_C_zeroMsgPayload();
    THRConfigMsgPayload        thrusterConfigOut = THRConfigMsg_C_zeroMsgPayload();

    /*! compute CM position w.r.t. M frame origin, in M coordinates */
    double MB[3][3];
    MRP2C(configData->sigma_MB, MB);                         // B to M DCM
    double r_CB_B[3];
    v3Copy(vehConfigMsgIn.CoM_B, r_CB_B);                    // position of C w.r.t. B in B-frame coordinates
    double r_CB_M[3];
    m33MultV3(MB, r_CB_B, r_CB_M);                           // position of C w.r.t. B in M-frame coordinates
    double r_CM_M[3];
    v3Add(r_CB_M, configData->r_BM_M, r_CM_M);               // position of C w.r.t. M in M-frame coordinates
    double r_TM_F[3];
    v3Add(configData->r_FM_F, thrusterConfigFIn.rThrust_B, r_TM_F);   // position of T w.r.t. M in F-frame coordinates
    double T_F[3];
    v3Copy(thrusterConfigFIn.tHatThrust_B, T_F);
    v3Scale(thrusterConfigFIn.maxThrust, T_F, T_F);
    
    double FM[3][3];
    tprComputeFinalRotation(r_CM_M, r_TM_F, T_F, FM);

    if (configData->momentumDumping == Yes) {
        RWSpeedMsgPayload rwSpeedMsgIn = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);

        /*! compute net RW momentum */
        double vec3[3];
        double hs_B[3];
        v3SetZero(hs_B);
        for (int i = 0; i < configData->rwConfigParams.numRW; i++) {
            v3Scale(configData->rwConfigParams.JsList[i] * rwSpeedMsgIn.wheelSpeeds[i],
                    &configData->rwConfigParams.GsMatrix_B[i * 3], vec3);
            v3Add(hs_B, vec3, hs_B);
        }
        double hs_M[3];
        m33tMultV3(MB, hs_B, hs_M);

        /*! update integral term */
        double DeltaHsInt_M[3];
        v3Add(configData->priorHs_M, hs_M, DeltaHsInt_M);
        double dt = (callTime - configData->priorTime) * NANO2SEC;
        v3Scale(0.5*dt, DeltaHsInt_M, DeltaHsInt_M);
        v3Add(configData->hsInt_M, DeltaHsInt_M, configData->hsInt_M);
        v3Copy(hs_M, configData->priorHs_M);
        configData->priorTime = callTime;

        /*! compute offset vector */
        double T_M[3];
        m33tMultV3(FM, T_F, T_M);
        double H[3];
        v3Scale(configData->K, hs_M, H);
        if (configData->Ki > 0) {
            double Hint[3];
            v3Scale(configData->Ki, configData->hsInt_M, Hint);
            v3Add(H, Hint, H);
        }
        double d_M[3];
        v3Cross(T_M, H, d_M);
        v3Scale(-1/v3Dot(T_M, T_M), d_M, d_M);

        /*! recompute thrust direction and FM matrix based on offset */
        double r_CMd_M[3];
        v3Add(r_CM_M, d_M, r_CMd_M);
        tprComputeFinalRotation(r_CMd_M, r_TM_F, T_F, FM);
    }

    double theta1 = atan2(FM[1][2], FM[1][1]);
    double theta2 = atan2(FM[2][0], FM[0][0]);

    /*! bound reference angles between limits */
    if ((configData->theta1Max > epsilon) && (theta1 > configData->theta1Max)) {
        theta1 = configData->theta1Max;
    }
    else if ((configData->theta1Max > epsilon) && (theta1 < -configData->theta1Max)) {
        theta1 = -configData->theta1Max;
    }
    if ((configData->theta2Max > epsilon) && (theta2 > configData->theta2Max)) {
        theta2 = configData->theta2Max;
    }
    else if ((configData->theta2Max > epsilon) && (theta2 < -configData->theta2Max)) {
        theta2 = -configData->theta2Max;
    }

    /*! rewrite DCM with updated angles */
    double EulerAngles123[3] = {theta1, theta2, 0.0};
    Euler1232C(EulerAngles123, FM);

    /*! extract theta1 and theta2 angles */
    hingedRigidBodyRef1Out.theta = theta1;
    hingedRigidBodyRef1Out.thetaDot = 0;
    hingedRigidBodyRef2Out.theta = theta2;
    hingedRigidBodyRef2Out.thetaDot = 0;

    /*! write output spinning body messages */
    HingedRigidBodyMsg_C_write(&hingedRigidBodyRef1Out, &configData->hingedRigidBodyRef1OutMsg, moduleID, callTime);
    HingedRigidBodyMsg_C_write(&hingedRigidBodyRef2Out, &configData->hingedRigidBodyRef2OutMsg, moduleID, callTime);

    /*! define mapping between final platform frame and body frame FB */
    double FB[3][3];
    m33MultM33(FM, MB, FB);

    /*! compute thruster direction in body frame coordinates */
    m33tMultV3(FB, T_F, bodyHeadingOut.rHat_XB_B);
    v3Normalize(bodyHeadingOut.rHat_XB_B, bodyHeadingOut.rHat_XB_B);

    /*! write output body heading message */
    BodyHeadingMsg_C_write(&bodyHeadingOut, &configData->bodyHeadingOutMsg, moduleID, callTime);

    /*! compute thruster torque on the system in body frame coordinates */
    double r_CM_F[3];
    m33MultV3(FM, r_CM_M, r_CM_F);
    double r_TC_F[3];
    v3Subtract(r_TM_F, r_CM_F, r_TC_F);
    double Torque_F[3];
    v3Cross(T_F, r_TC_F, Torque_F);    // compute the opposite of torque to compensate with the RWs
    m33tMultV3(FB, Torque_F, thrusterTorqueOut.torqueRequestBody);

    /*! write output commanded torque message */
    CmdTorqueBodyMsg_C_write(&thrusterTorqueOut, &configData->thrusterTorqueOutMsg, moduleID, callTime);

    /*! populate thrusterConfigOut */
    double r_TC_B[3];
    m33tMultV3(FB, r_TC_F, r_TC_B);
    v3Add(r_CB_B, r_TC_B, thrusterConfigOut.rThrust_B);
    m33tMultV3(FB, T_F, thrusterConfigOut.tHatThrust_B);
    v3Normalize(thrusterConfigOut.tHatThrust_B, thrusterConfigOut.tHatThrust_B);
    thrusterConfigOut.maxThrust = v3Norm(T_F);

    /*! write output thruster config msg */
    THRConfigMsg_C_write(&thrusterConfigOut, &configData->thrusterConfigBOutMsg, moduleID, callTime);
}

void tprComputeFirstRotation(double THat_F[3], double rHat_CM_F[3], double F1M[3][3])
{
    // compute principal rotation angle phi
    double phi = acos( fmin( fmax( v3Dot(THat_F, rHat_CM_F), -1 ), 1 ) );

    // compute principal rotation vector e_phi
    double e_phi[3];
    v3Cross(THat_F, rHat_CM_F, e_phi);
    // If phi = PI, e_phi can be any vector perpendicular to F_current_B
    if (fabs(phi-MPI) < epsilon) {
        phi = MPI;
        if (fabs(THat_F[0]) > epsilon) {
            e_phi[0] = -(THat_F[1]+THat_F[2]) / THat_F[0];
            e_phi[1] = 1;
            e_phi[2] = 1;
        }
        else if (fabs(THat_F[1]) > epsilon) {
            e_phi[0] = 1;
            e_phi[1] = -(THat_F[0]+THat_F[2]) / THat_F[1];
            e_phi[2] = 1;
        }
        else {
            e_phi[0] = 1;
            e_phi[1] = 1;
            e_phi[2] = -(THat_F[0]+THat_F[1]) / THat_F[2];
        }
    }
    else if (fabs(phi) < epsilon) {
        phi = 0;
    }
    // normalize e_phi
    v3Normalize(e_phi, e_phi);

    // compute intermediate rotation F1M
    double PRV_phi[3];
    v3Scale(phi, e_phi, PRV_phi);
    PRV2C(PRV_phi, F1M);
}

void tprComputeSecondRotation(double r_CM_F[3], double r_TM_F[3], double r_CT_F[3], double THat_F[3], double F2F1[3][3])
{
    // define offset vector aVec
    double aVec[3];
    v3Copy(r_TM_F, aVec);
    double a = v3Norm(aVec);

    // define offset vector aVec
    double bVec[3];
    v3Copy(r_CM_F, bVec);
    double b = v3Norm(bVec);
    
    double c1 = v3Norm(r_CT_F);

    double psi;
    if (fabs(a) < epsilon) {
        // if offset a = 0, second rotation is null
        psi = 0;
    }
    else {
        double beta = acos( -fmin( fmax( v3Dot(aVec, THat_F) / a, -1 ), 1 ) );
        double nu = acos( -fmin( fmax( v3Dot(aVec, r_CT_F) / (a*c1), -1 ), 1 ) );

        double c2 = a*cos(beta) + sqrt(b*b - a*a*sin(beta)*sin(beta));

        double cosGamma1 = (a*a + b*b - c1*c1) / (2*a*b);
        double cosGamma2 = (a*a + b*b - c2*c2) / (2*a*b);

        psi = asin( fmin( fmax( (c1*sin(nu)*cosGamma2 - c2*sin(beta)*cosGamma1)/b, -1 ), 1 ) );
    }
    
    double e_psi[3];
    v3Cross(THat_F, r_CT_F, e_psi);
    v3Normalize(e_psi, e_psi);

    // compute intermediate rotation F2F1
    double PRV_psi[3];
    v3Scale(psi, e_psi, PRV_psi);
    PRV2C(PRV_psi, F2F1);
}

void tprComputeThirdRotation(double e_theta[3], double F2M[3][3], double F3F2[3][3])
{
    double e1 = e_theta[0];  
    double e2 = e_theta[1];  
    double e3 = e_theta[2];

    double A = 2 * (F2M[1][0]*e2*e2 + F2M[0][0]*e1*e2 + F2M[2][0]*e2*e3) - F2M[1][0];
    double B = 2 * (F2M[2][0]*e1 - F2M[0][0]*e3);
    double C = F2M[1][0];
    double Delta = B*B - 4*A*C;

    /* compute exact solution or best solution depending on Delta */
    double t, t1, t2, y, y1, y2, theta;
    if (fabs(A) < epsilon) {
        if (fabs(B) < epsilon) {
            // zero-th order equation has no solution 
            // the solution of the minimum problem is theta = MPI
            theta = MPI;
        }
        else {
            // first order equation
            t = - C / B;
            theta = 2*atan(t);
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
            theta = 2*atan(t);
            y = (A*t*t + B*t + C) / (1 + t*t);
            // check if the absolute fcn minimum is for theta = MPI
            if (fabs(A) < fabs(y)) {
                theta = MPI;
            }
        }
        else {
            t1 = (-B + sqrt(Delta)) / (2*A);
            t2 = (-B - sqrt(Delta)) / (2*A);
            t = t1;
            if (fabs(t2) < fabs(t1)) {
                t = t2;
            }

            theta = 2*atan(t);
        }
    }

    // compute intermediate rotation F3F2
    double PRV_theta[3];
    v3Scale(theta, e_theta, PRV_theta);
    PRV2C(PRV_theta, F3F2);
}

void tprComputeFinalRotation(double r_CM_M[3], double r_TM_F[3], double T_F[3], double FM[3][3])
{
    /*! define unit vectors of CM direction in M coordinates and thrust direction in F coordinates */
    double rHat_CM_M[3];
    v3Normalize(r_CM_M, rHat_CM_M);
    double THat_F[3];
    v3Normalize(T_F, THat_F);
    double rHat_CM_F[3];
    v3Copy(rHat_CM_M, rHat_CM_F);        // assume zero initial rotation between F and M

    /*! compute first rotation to make T_F parallel to r_CM */
    double F1M[3][3];
    tprComputeFirstRotation(THat_F, rHat_CM_F, F1M);

    /*! rotate r_CM_F */
    double r_CM_F[3];
    m33MultV3(F1M, r_CM_M, r_CM_F);

    /*! compute position of CM w.r.t. thrust application point T */
    double r_CT_F[3];
    v3Subtract(r_CM_F, r_TM_F, r_CT_F);

    /*! compute second rotation to zero the offset between T_F and r_CT_F */
    double F2F1[3][3];
    tprComputeSecondRotation(r_CM_F, r_TM_F, r_CT_F, THat_F, F2F1);

    /*! define intermediate platform rotation F2M */
    double F2M[3][3];
    m33MultM33(F2F1, F1M, F2M);

    /*! compute third rotation to make the frame compliant with the platform constraint */
    double e_theta[3];
    m33MultV3(F2M, r_CM_M, e_theta);
    v3Normalize(e_theta, e_theta);
    double F3F2[3][3];
    tprComputeThirdRotation(e_theta, F2M, F3F2);

    /*! define final platform rotation FM */
    m33MultM33(F3F2, F2M, FM);
}
