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

#include <string.h>
#include <math.h>
#include "fswAlgorithms/attGuidance/sunSafePoint/sunSafePoint.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/astroConstants.h"

/*! This method initializes the configData for the sun safe attitude guidance.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the sun safe guidance
 @param moduleID The Basilisk module identifier
 */
void SelfInit_sunSafePoint(sunSafePointConfig *configData, int64_t moduleID)
{
    AttGuidMsg_C_init(&configData->attGuidanceOutMsg);
    
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the guidance module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Reset_sunSafePoint(sunSafePointConfig *configData, uint64_t callTime, int64_t moduleID)
{
    double v1[3];

    // check if the required input messages are included
    if (!NavAttMsg_C_isLinked(&configData->sunDirectionInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunSafePoint.sunDirectionInMsg wasn't connected.");
    }
    if (!NavAttMsg_C_isLinked(&configData->imuInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunSafePoint.imuInMsg wasn't connected.");
    }

    /* compute an Eigen axis orthogonal to sHatBdyCmd */
    if (v3Norm(configData->sHatBdyCmd)  < 0.1) {
      char info[MAX_LOGGING_LENGTH];
      sprintf(info, "The module vector sHatBdyCmd is not setup as a unit vector [%f, %f %f]",
                configData->sHatBdyCmd[0], configData->sHatBdyCmd[1], configData->sHatBdyCmd[2]);
      _bskLog(configData->bskLogger, BSK_ERROR, info);
    } else {
        v3Set(1., 0., 0., v1);
        v3Normalize(configData->sHatBdyCmd, configData->sHatBdyCmd);    /* ensure that this vector is a unit vector */
        v3Cross(configData->sHatBdyCmd, v1, configData->eHat180_B);
        if (v3Norm(configData->eHat180_B) < 0.1) {
            v3Set(0., 1., 0., v1);
            v3Cross(configData->sHatBdyCmd, v1, configData->eHat180_B);
        }
        v3Normalize(configData->eHat180_B, configData->eHat180_B);
    }

    configData->attGuidanceOutBuffer = AttGuidMsg_C_zeroMsgPayload();

    return;
}

/*! This method takes the estimated body-observed sun vector and computes the
 current attitude/attitude rate errors to pass on to control.
 @return void
 @param configData The configuration data associated with the sun safe attitude guidance
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
*/
void Update_sunSafePoint(sunSafePointConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    NavAttMsgPayload navMsg;
    double ctSNormalized;
    double sNorm;                   /*!< --- Norm of measured direction vector */
    double e_hat[3];                /*!< --- Eigen Axis */
    double omega_BN_B[3];           /*!< r/s inertial body angular velocity vector in B frame components */
    double omega_RN_B[3];           /*!< r/s local copy of the desired reference frame rate */

    NavAttMsgPayload localImuDataInBuffer;
    configData->attGuidanceOutBuffer = AttGuidMsg_C_zeroMsgPayload();

    /*! - Read the current sun body vector estimate*/
    navMsg = NavAttMsg_C_read(&configData->sunDirectionInMsg);
    localImuDataInBuffer = NavAttMsg_C_read(&configData->imuInMsg);

    v3Copy(localImuDataInBuffer.omega_BN_B, omega_BN_B);

    /*! - Compute the current error vector if it is valid*/
    sNorm = v3Norm(navMsg.vehSunPntBdy);
    if(sNorm > configData->minUnitMag)
    {
        /* a good sun direction vector is available */
        ctSNormalized = v3Dot(configData->sHatBdyCmd, navMsg.vehSunPntBdy)/sNorm;
        ctSNormalized = fabs(ctSNormalized) > 1.0 ?
        ctSNormalized/fabs(ctSNormalized) : ctSNormalized;
        configData->sunAngleErr = safeAcos(ctSNormalized);

        /*
            Compute the heading error relative to the sun direction vector 
         */
        if (configData->sunAngleErr < configData->smallAngle) {
            /* sun heading and desired body axis are essentially aligned.  Set attitude error to zero. */
             v3SetZero(configData->attGuidanceOutBuffer.sigma_BR);
        } else {
            if (M_PI - configData->sunAngleErr < configData->smallAngle) {
                /* the commanded body vector nearly is opposite the sun heading */
                v3Copy(configData->eHat180_B, e_hat);
            } else {
                /* normal case where sun and commanded body vectors are not aligned */
                v3Cross(navMsg.vehSunPntBdy, configData->sHatBdyCmd, e_hat);
            }
            v3Normalize(e_hat, configData->sunMnvrVec);
            v3Scale(tan(configData->sunAngleErr*0.25), configData->sunMnvrVec,
                    configData->attGuidanceOutBuffer.sigma_BR);
            MRPswitch(configData->attGuidanceOutBuffer.sigma_BR, 1.0, configData->attGuidanceOutBuffer.sigma_BR);
        }

        /* rate tracking error are the body rates to bring spacecraft to rest */
        v3Scale(configData->sunAxisSpinRate/sNorm, navMsg.vehSunPntBdy, omega_RN_B);
        v3Subtract(omega_BN_B, omega_RN_B, configData->attGuidanceOutBuffer.omega_BR_B);
        v3Copy(omega_RN_B, configData->attGuidanceOutBuffer.omega_RN_B);

    } else {
        /* no proper sun direction vector is available */
        v3SetZero(configData->attGuidanceOutBuffer.sigma_BR);

        /* specify a body-fixed constant search rotation rate */
        v3Subtract(omega_BN_B, configData->omega_RN_B, configData->attGuidanceOutBuffer.omega_BR_B);
        v3Copy(configData->omega_RN_B, configData->attGuidanceOutBuffer.omega_RN_B);
    }

    /* write the Guidance output message */
    AttGuidMsg_C_write(&configData->attGuidanceOutBuffer, &configData->attGuidanceOutMsg, moduleID, callTime);
    
    return;
}
