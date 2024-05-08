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
#include "fswAlgorithms/attGuidance/attTrackingError/attTrackingError.h"
#include "fswAlgorithms/fswUtilities/fswDefinitions.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"


/*! This method initializes the configData for this module.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the attitude tracking error module
 @param moduleID The ID associated with the configData
 */
void SelfInit_attTrackingError(attTrackingErrorConfig *configData, int64_t moduleID)
{
    AttGuidMsg_C_init(&configData->attGuidOutMsg);
}


/*! This method performs a complete reset of the module. Local module variables that retain time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the attitude tracking error module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_attTrackingError(attTrackingErrorConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!AttRefMsg_C_isLinked(&configData->attRefInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: attTrackingError.attRefInMsg wasn't connected.");
    }
    if (!NavAttMsg_C_isLinked(&configData->attNavInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: attTrackingError.attNavInMsg wasn't connected.");
    }

    return;
}

/*! The Update method performs reads the Navigation message (containing the spacecraft attitude information), and the Reference message (containing the desired attitude). It computes the attitude error and writes it in the Guidance message.
 @return void
 @param configData The configuration data associated with the attitude tracking error module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Update_attTrackingError(attTrackingErrorConfig *configData, uint64_t callTime, int64_t moduleID)
{
    AttRefMsgPayload ref;                      /* reference guidance message */
    NavAttMsgPayload nav;                      /* navigation message */
    AttGuidMsgPayload attGuidOut;              /* Guidance message */

    /*! - Read the input messages */
    attGuidOut = AttGuidMsg_C_zeroMsgPayload();

    ref = AttRefMsg_C_read(&configData->attRefInMsg);
    nav = NavAttMsg_C_read(&configData->attNavInMsg);

    computeAttitudeError(configData->sigma_R0R, nav, ref, &attGuidOut);

    AttGuidMsg_C_write(&attGuidOut, &configData->attGuidOutMsg, moduleID, callTime);

    return;
}

/*! This method performs the attitude computations in order to extract the error.
 @return void
 @param sigma_R0R Reference frame state
 @param nav The spacecraft attitude information
 @param ref The reference attitude
 @param attGuidOut Output attitude guidance message
 */
void computeAttitudeError(double sigma_R0R[3], NavAttMsgPayload nav, AttRefMsgPayload ref, AttGuidMsgPayload *attGuidOut){
    double      sigma_RR0[3];               /* MRP from the original reference frame R0 to the corrected reference frame R */
    double      sigma_RN[3];                /* MRP from inertial to updated reference frame */
    double      dcm_BN[3][3];               /* DCM from inertial to body frame */

    /*! - compute the initial reference frame orientation that takes the corrected body frame into account */
    v3Scale(-1.0, sigma_R0R, sigma_RR0);
    addMRP(ref.sigma_RN, sigma_RR0, sigma_RN);

    subMRP(nav.sigma_BN, sigma_RN, attGuidOut->sigma_BR);               /*! - compute attitude error */

    MRP2C(nav.sigma_BN, dcm_BN);                                /* [BN] */
    m33MultV3(dcm_BN, ref.omega_RN_N, attGuidOut->omega_RN_B);              /*! - compute reference omega in body frame components */

    v3Subtract(nav.omega_BN_B, attGuidOut->omega_RN_B, attGuidOut->omega_BR_B);     /*! - delta_omega = omega_B - [BR].omega.r */

    m33MultV3(dcm_BN, ref.domega_RN_N, attGuidOut->domega_RN_B);            /*! - compute reference d(omega)/dt in body frame components */

}
