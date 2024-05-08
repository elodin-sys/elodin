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

#include "thrusterPlatformState.h"
#include <math.h>
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"

/*! This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_thrusterPlatformState(thrusterPlatformStateConfig *configData, int64_t moduleID)
{
    THRConfigMsg_C_init(&configData->thrusterConfigBOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_thrusterPlatformState(thrusterPlatformStateConfig *configData, uint64_t callTime, int64_t moduleID)
{
    if (!THRConfigMsg_C_isLinked(&configData->thrusterConfigFInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, " thrusterPlatformState.thrusterConfigFInMsg wasn't connected.");
    }
    if (!HingedRigidBodyMsg_C_isLinked(&configData->hingedRigidBody1InMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, " thrusterPlatformState.hingedRigidBody1InMsg wasn't connected.");
    }
    if (!HingedRigidBodyMsg_C_isLinked(&configData->hingedRigidBody2InMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, " thrusterPlatformState.hingedRigidBody2InMsg wasn't connected.");
    }
}


/*! This method updates the platformAngles message based on the updated information about the system center of mass
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_thrusterPlatformState(thrusterPlatformStateConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! - Create and assign message buffers */
    THRConfigMsgPayload        thrusterConfigFIn   = THRConfigMsg_C_read(&configData->thrusterConfigFInMsg);
    HingedRigidBodyMsgPayload  hingedRigidBody1In = HingedRigidBodyMsg_C_read(&configData->hingedRigidBody1InMsg);
    HingedRigidBodyMsgPayload  hingedRigidBody2In = HingedRigidBodyMsg_C_read(&configData->hingedRigidBody2InMsg);
    THRConfigMsgPayload        thrusterConfigBOut  = THRConfigMsg_C_zeroMsgPayload();

    /*! compute CM position w.r.t. M frame origin, in M coordinates */
    double MB[3][3];
    MRP2C(configData->sigma_MB, MB);                                 // B to M DCM
    double r_TM_F[3];
    v3Add(configData->r_FM_F, thrusterConfigFIn.rThrust_B, r_TM_F);   // position of T w.r.t. M in F-frame coordinates
    double T_F[3];
    v3Copy(thrusterConfigFIn.tHatThrust_B, T_F);
    v3Scale(thrusterConfigFIn.maxThrust, T_F, T_F);

    /*! extract theta1 and theta2 angles and compute FB DCM */
    double EulerAngles123[3] = {hingedRigidBody1In.theta, hingedRigidBody2In.theta, 0.0};
    double FM[3][3];
    Euler1232C(EulerAngles123, FM);
    double FB[3][3];
    m33MultM33(FM, MB, FB);

    /*! populate output msg */
    double r_TM_B[3];
    m33tMultV3(FB, r_TM_F, r_TM_B);
    double r_BM_B[3];
    m33tMultV3(MB, configData->r_BM_M, r_BM_B);
    v3Subtract(r_TM_B, r_BM_B, thrusterConfigBOut.rThrust_B);
    m33tMultV3(FB, thrusterConfigFIn.tHatThrust_B, thrusterConfigBOut.tHatThrust_B);
    thrusterConfigBOut.maxThrust = thrusterConfigFIn.maxThrust;

    /*! write output thruster config msg */
    THRConfigMsg_C_write(&thrusterConfigBOut, &configData->thrusterConfigBOutMsg, moduleID, callTime);
}

