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


#include "mtbMomentumManagementSimple.h"
#include "architecture/utilities/linearAlgebra.h"
#include <stdio.h>

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_mtbMomentumManagementSimple(mtbMomentumManagementSimpleConfig  *configData, int64_t moduleID)
{
    /*
     * Initialize the output message.
     */
    CmdTorqueBodyMsg_C_init(&configData->tauMtbRequestOutMsg);
    
    return;
}


/*! This method performs a complete reset of the module.  Local module variables that retain
    time varying states between function calls are reset to their default values.
    Check if required input messages are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_mtbMomentumManagementSimple(mtbMomentumManagementSimpleConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*
     * Check if the required input messages are connected.
     */
    if (!RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbMomentumManagement.rwParamsInMsg is not connected.");
    }
    if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbMomentumManagement.rwSpeedsInMsg is not connected.");
    }
    
    /*! - Read in the reaction wheels input configuration message. This gives us the transformation from
         from the wheel space to the Body frame through GsMatrix_B.*/
    configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwParamsInMsg);
    
    /*
     * Compute the transpose of GsMatrix_B, which is an array of the spin
     * axis of the reaction wheels. By transposing it we get the transformation
     * from the wheel space to the Body frame, Gs.
     */
    mTranspose(configData->rwConfigParams.GsMatrix_B, configData->rwConfigParams.numRW, 3, configData->Gs);
    
    /*
     * Sanity check configs.
     */
    if (configData->Kp < 0.0)
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: k < 0.0");
    
    return;
}


/*! This routine calculate the current desired torque in the Body frame to meet the momentum target.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_mtbMomentumManagementSimple(mtbMomentumManagementSimpleConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*
     * Initialize local variables.
     */
    double hWheels_B[3] = {0.0, 0.0, 0.0};                      // the net momentum of the reaction wheels in the body frame
    double hWheels_W[MAX_EFF_CNT];                              // array of individual wheel momentum values
    vSetZero(hWheels_W, configData->rwConfigParams.numRW);
    
    /*
     * Read the input message and initialize output message.
     */
    RWSpeedMsgPayload rwSpeedsInMsgBuffer = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);
    CmdTorqueBodyMsgPayload tauMtbRequestOutMsgBuffer = CmdTorqueBodyMsg_C_zeroMsgPayload(&configData->tauMtbRequestOutMsg);
    
    /*! - Compute wheel momentum in Body frame components by calculating it first in the wheel frame and then
         transforming it from the wheel space into the body frame using Gs.*/
    vElementwiseMult(rwSpeedsInMsgBuffer.wheelSpeeds, configData->rwConfigParams.numRW, configData->rwConfigParams.JsList, hWheels_W);
    mMultV(configData->Gs, 3, configData->rwConfigParams.numRW, hWheels_W, hWheels_B);
    
    /*! - Compute the feedback torque command by multiplying the wheel momentum in the Body frame by the proportional
         momentum gain Kp. Note that this module is currently targeting a wheel momentum in the Body frame of zero and
         hWheels_B is the momentum feedback error and needs to be multiplied by a negative sign.*/
    v3Scale(-configData->Kp, hWheels_B, tauMtbRequestOutMsgBuffer.torqueRequestBody);
    
    /*! - Write the output message. This is the torque we are requesting the torque bars to produce in the Body frame.
         Note that depending on the torque rod/magentic field geometry, torque rod saturation limts, unknown alignments,
         and imperfect sensor readings, this torque may not be perfectly produced.*/
    CmdTorqueBodyMsg_C_write(&tauMtbRequestOutMsgBuffer, &configData->tauMtbRequestOutMsg, moduleID, callTime);
}
