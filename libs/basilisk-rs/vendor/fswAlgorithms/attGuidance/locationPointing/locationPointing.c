/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#include "fswAlgorithms/attGuidance/locationPointing/locationPointing.h"
#include "string.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/macroDefinitions.h"
#include <math.h>

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_locationPointing(locationPointingConfig  *configData, int64_t moduleID)
{
    AttGuidMsg_C_init(&configData->attGuidOutMsg);
    AttRefMsg_C_init(&configData->attRefOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
    time varying states between function calls are reset to their default values.
    Check if required input messages are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_locationPointing(locationPointingConfig *configData, uint64_t callTime, int64_t moduleID)
{

    // check if the required message has not been connected
    if (!NavAttMsg_C_isLinked(&configData->scAttInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: locationPointing.scAttInMsg was not connected.");
    }
    if (!NavTransMsg_C_isLinked(&configData->scTransInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: locationPointing.scTransInMsg was not connected.");
    }
    int numInMsgs = GroundStateMsg_C_isLinked(&configData->locationInMsg)
                    + EphemerisMsg_C_isLinked(&configData->celBodyInMsg)
                    + NavTransMsg_C_isLinked(&configData->scTargetInMsg);

    if (numInMsgs == 0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: In the locationPointing module no target messages were not connected.");
    }
    else if (numInMsgs > 1) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: In the locationPointing module multiple target messages were connected. Defaulting to either ground location, planet location or spacecraft location, in that order.");
    }

    configData->init = 1;

    v3SetZero(configData->sigma_BR_old);
    configData->time_old = callTime;
    
    /* compute an Eigen axis orthogonal to sHatBdyCmd */
    if (v3Norm(configData->pHat_B)  < 0.1) {
      char info[MAX_LOGGING_LENGTH];
      sprintf(info, "locationPoint: vector pHat_B is not setup as a unit vector [%f, %f %f]",
                configData->pHat_B[0], configData->pHat_B[1], configData->pHat_B[2]);
      _bskLog(configData->bskLogger, BSK_ERROR, info);
    } else {
        double v1[3];
        v3Set(1., 0., 0., v1);
        v3Normalize(configData->pHat_B, configData->pHat_B);    /* ensure that this vector is a unit vector */
        v3Cross(configData->pHat_B, v1, configData->eHat180_B);
        if (v3Norm(configData->eHat180_B) < 0.1) {
            v3Set(0., 1., 0., v1);
            v3Cross(configData->pHat_B, v1, configData->eHat180_B);
        }
        v3Normalize(configData->eHat180_B, configData->eHat180_B);
    }
}


/*! This method takes the estimated body states and position relative to the ground to compute the current attitude/attitude rate errors and pass them to control.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_locationPointing(locationPointingConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /* Local copies*/
    NavAttMsgPayload scAttInMsgBuffer;  //!< local copy of input message buffer
    NavTransMsgPayload scTransInMsgBuffer;  //!< local copy of input message buffer
    GroundStateMsgPayload locationInMsgBuffer;  //!< local copy of location input message buffer
    EphemerisMsgPayload celBodyInMsgBuffer; //!< local copy of celestial body input message buffer
    NavTransMsgPayload scTargetInMsgBuffer;  //!< local copy of input message buffer of target spacecraft
    AttGuidMsgPayload attGuidOutMsgBuffer;  //!< local copy of guidance output message buffer
    AttRefMsgPayload attRefOutMsgBuffer;  //!< local copy of reference output message buffer

    double r_LS_N[3];                   /*!< Position vector of location w.r.t spacecraft CoM in inertial frame */
    double r_LS_B[3];                   /*!< Position vector of location w.r.t spacecraft CoM in body frame */
    double rHat_LS_B[3];                /*!< unit vector of location w.r.t spacecraft CoM in body frame */
    double eHat_B[3];                   /*!< --- Eigen Axis */
    double dcmBN[3][3];                 /*!< inertial spacecraft orientation DCM */
    double phi;                         /*!< principal angle between pHat and heading to location */
    double sigmaDot_BR[3];              /*!< time derivative of sigma_BR*/
    double sigma_BR[3];                 /*!< MRP of B relative to R */
    double sigma_RB[3];                 /*!< MRP of R relative to B */
    double omega_RN_B[3];               /*!< angular velocity of the R frame w.r.t the B frame in B frame components */
    double difference[3];
    double time_diff;                   /*!< module update time */
    double Binv[3][3];                  /*!< BinvMRP for dsigma_RB_R calculations*/
    double dum1;
    double r_TN_N[3];                   /*!< [m] inertial target location */
    double boreRate_B[3];               /*!< [rad/s] rotation rate about target direction */

    // zero output buffer
    attGuidOutMsgBuffer = AttGuidMsg_C_zeroMsgPayload();
    attRefOutMsgBuffer = AttRefMsg_C_zeroMsgPayload();

    // read in the input messages
    scAttInMsgBuffer = NavAttMsg_C_read(&configData->scAttInMsg);
    scTransInMsgBuffer = NavTransMsg_C_read(&configData->scTransInMsg);
    if (GroundStateMsg_C_isLinked(&configData->locationInMsg)) {
        locationInMsgBuffer = GroundStateMsg_C_read(&configData->locationInMsg);
        v3Copy(locationInMsgBuffer.r_LN_N, r_TN_N);
    }
    else if (EphemerisMsg_C_isLinked(&configData->celBodyInMsg)) {
        celBodyInMsgBuffer = EphemerisMsg_C_read(&configData->celBodyInMsg);
        v3Copy(celBodyInMsgBuffer.r_BdyZero_N, r_TN_N);
    } else {
        scTargetInMsgBuffer = NavTransMsg_C_read(&configData->scTargetInMsg);
        v3Copy(scTargetInMsgBuffer.r_BN_N, r_TN_N);
    }

    /* calculate r_LS_N*/
    v3Subtract(r_TN_N, scTransInMsgBuffer.r_BN_N, r_LS_N);

    /* principle rotation angle to point pHat at location */
    MRP2C(scAttInMsgBuffer.sigma_BN, dcmBN);
    m33MultV3(dcmBN, r_LS_N, r_LS_B);
    v3Normalize(r_LS_B, rHat_LS_B);
    dum1 = v3Dot(configData->pHat_B, rHat_LS_B);
    if (fabs(dum1) > 1.0) {
        dum1 = dum1 / fabs(dum1);
    }
    phi = safeAcos(dum1);

    /* calculate sigma_BR */
    if (phi < configData->smallAngle) {
        /* body axis and desired inertial pointing direction are essentially aligned.  Set attitude error to zero. */
         v3SetZero(sigma_BR);
    } else {
        if (M_PI - phi < configData->smallAngle) {
            /* the commanded body vector nearly is opposite the desired inertial heading */
            v3Copy(configData->eHat180_B, eHat_B);
        } else {
            /* normal case where body and inertial heading vectors are not aligned */
            v3Cross(configData->pHat_B, rHat_LS_B, eHat_B);
        }
        v3Normalize(eHat_B, eHat_B);
        v3Scale(-tan(phi / 4.), eHat_B, sigma_BR);
    }
    v3Copy(sigma_BR, attGuidOutMsgBuffer.sigma_BR);

    // compute sigma_RN
    v3Scale(-1.0, sigma_BR, sigma_RB);
    addMRP(scAttInMsgBuffer.sigma_BN, sigma_RB, attRefOutMsgBuffer.sigma_RN);
    
    /* use sigma_BR to compute d(sigma_BR)/dt if at least two data points */
    if (configData->init < 1) {
        // module update time
        time_diff = (callTime - configData->time_old)*NANO2SEC;

        // calculate d(sigma_BR)/dt
        v3Subtract(sigma_BR, configData->sigma_BR_old, difference);
        v3Scale(1.0/(time_diff), difference, sigmaDot_BR);

        // calculate BinvMRP
        BinvMRP(sigma_BR, Binv);
        
        // compute omega_BR_B
        v3Scale(4.0, sigmaDot_BR, sigmaDot_BR);
        m33MultV3(Binv, sigmaDot_BR, attGuidOutMsgBuffer.omega_BR_B);

    } else {
        configData->init -= 1;
    }

    if (configData->useBoresightRateDamping) {
        v3Scale(v3Dot(scAttInMsgBuffer.omega_BN_B, rHat_LS_B), rHat_LS_B, boreRate_B);
        v3Add(attGuidOutMsgBuffer.omega_BR_B, boreRate_B, attGuidOutMsgBuffer.omega_BR_B);
    }

    // compute omega_RN_B
    v3Subtract(scAttInMsgBuffer.omega_BN_B, attGuidOutMsgBuffer.omega_BR_B, omega_RN_B);

    // convert to omega_RN_N
    m33tMultV3(dcmBN, omega_RN_B, attRefOutMsgBuffer.omega_RN_N);

    // copy current attitude states into prior state buffers
    v3Copy(sigma_BR, configData->sigma_BR_old);

    // update former module call time
    configData->time_old = callTime;

    // write to the output messages
    AttGuidMsg_C_write(&attGuidOutMsgBuffer, &configData->attGuidOutMsg, moduleID, callTime);
    AttRefMsg_C_write(&attRefOutMsgBuffer, &configData->attRefOutMsg, moduleID, callTime);
}

