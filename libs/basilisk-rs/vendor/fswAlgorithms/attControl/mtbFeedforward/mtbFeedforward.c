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


#include "fswAlgorithms/attControl/mtbFeedforward/mtbFeedforward.h"
#include "string.h"
#include "architecture/utilities/linearAlgebra.h"

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_mtbFeedforward(mtbFeedforwardConfig  *configData, int64_t moduleID)
{
    /*
     * Initialize the output message.
     */
    CmdTorqueBodyMsg_C_init(&configData->vehControlOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
    time varying states between function calls are reset to their default values.
    Check if required input messages are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_mtbFeedforward(mtbFeedforwardConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*
     * Check if the required input messages are connected.
     */
    if (!MTBCmdMsg_C_isLinked(&configData->dipoleRequestMtbInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbFeedForward.dipoleRequestMtbInMsg is not connected.");
    }
    if (!CmdTorqueBodyMsg_C_isLinked(&configData->vehControlInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbFeedForward.vehControlInMsg is not connected.");
    }
    if (!TAMSensorBodyMsg_C_isLinked(&configData->tamSensorBodyInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbFeedForward.tamSensorBodyInMsg is not connected.");
    }
    if (!MTBArrayConfigMsg_C_isLinked(&configData->mtbArrayConfigParamsInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbFeedForward.mtbArrayConfigParamsInMsg is not connected.");
    }
    
    /*! - Read in the torque rod input configuration message. This gives us the transformation from the
         torque rod space the the Body frame.*/
    configData->mtbArrayConfigParams = MTBArrayConfigMsg_C_read(&configData->mtbArrayConfigParamsInMsg);
}


/*! Computes the feedforward torque rod torque.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_mtbFeedforward(mtbFeedforwardConfig *configData, uint64_t callTime, int64_t moduleID)
{

    /*
     * Initialize local variables.
     */
    double mtbDipoleCmd_B[3] = {0.0, 0.0, 0.0};     // the commanded dipole in the Body frame
    double tauMtbFF_B[3] = {0.0, 0.0, 0.0};         // the torque rod feedforward term in the Body frame
    
    /*
     * Read the input messages and initialize output message.
     */
    MTBCmdMsgPayload dipoleRequestMtbInMsgBuffer = MTBCmdMsg_C_read(&configData->dipoleRequestMtbInMsg);
    TAMSensorBodyMsgPayload tamSensorBodyInMsgBuffer = TAMSensorBodyMsg_C_read(&configData->tamSensorBodyInMsg);
    CmdTorqueBodyMsgPayload vehControlOutMsgBuffer = CmdTorqueBodyMsg_C_read(&configData->vehControlInMsg);

    /*! -  Compute net torque produced on the vehicle from the torque bars.*/
    mMultV(configData->mtbArrayConfigParams.GtMatrix_B, 3, configData->mtbArrayConfigParams.numMTB, dipoleRequestMtbInMsgBuffer.mtbDipoleCmds, mtbDipoleCmd_B);
    v3Cross(mtbDipoleCmd_B, tamSensorBodyInMsgBuffer.tam_B, tauMtbFF_B);
    
    /*! -  Negate the net rod torque to spin wheels in appropriate direction. */
    v3Subtract(vehControlOutMsgBuffer.torqueRequestBody, tauMtbFF_B, vehControlOutMsgBuffer.torqueRequestBody);

    /*! - Write output message. This used as a feedforward term to the attiude controller.*/
    CmdTorqueBodyMsg_C_write(&vehControlOutMsgBuffer, &configData->vehControlOutMsg, moduleID, callTime);
}
