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


#include "fswAlgorithms/effectorInterfaces/torque2Dipole/torque2Dipole.h"
#include "string.h"
#include "architecture/utilities/linearAlgebra.h"

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_torque2Dipole(torque2DipoleConfig  *configData, int64_t moduleID)
{
    /*
     * Initialize the output message.
     */
    DipoleRequestBodyMsg_C_init(&configData->dipoleRequestOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
    time varying states between function calls are reset to their default values.
    Check if required input messages are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_torque2Dipole(torque2DipoleConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*
     * Check if the required input messages are connected.
     */
    if (!TAMSensorBodyMsg_C_isLinked(&configData->tamSensorBodyInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: torque2Dipole.tamSensorBodyInMsg is not connected.");
    }
    if (!CmdTorqueBodyMsg_C_isLinked(&configData->tauRequestInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: torque2Dipole.tauRequestInMsg is not connected.");
    }
}


/*! This method transforms the requested torque from the torque rods into a Body frame requested dipole from the torque rods.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_torque2Dipole(torque2DipoleConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*
     * Initialize local variables.
     */
    double bFieldNormSqrd = 0.0;        // the norm squared of the local magnetic field vector
    
    /*
     * Read the input messages and initialize output message.
     */
    TAMSensorBodyMsgPayload tamSensorBodyInMsgBuffer = TAMSensorBodyMsg_C_read(&configData->tamSensorBodyInMsg);
    CmdTorqueBodyMsgPayload tauRequestInMsgBuffer = CmdTorqueBodyMsg_C_read(&configData->tauRequestInMsg);
    DipoleRequestBodyMsgPayload dipoleRequestOutMsgBuffer = DipoleRequestBodyMsg_C_zeroMsgPayload();
    
    /*! - Transform the requested Body frame torque into a requested Body frame dipole protecting against a bogus
         magnetic field value. */
    bFieldNormSqrd = v3Dot(tamSensorBodyInMsgBuffer.tam_B, tamSensorBodyInMsgBuffer.tam_B);
    if (bFieldNormSqrd > DB0_EPS)
    {
        v3Cross(tamSensorBodyInMsgBuffer.tam_B, tauRequestInMsgBuffer.torqueRequestBody, dipoleRequestOutMsgBuffer.dipole_B);
        v3Scale(1 / bFieldNormSqrd, dipoleRequestOutMsgBuffer.dipole_B, dipoleRequestOutMsgBuffer.dipole_B);
    }
    
    /*! - Write output message. This is the Body frame requested dipole from the torque rods.*/
    DipoleRequestBodyMsg_C_write(&dipoleRequestOutMsgBuffer, &configData->dipoleRequestOutMsg, moduleID, callTime);
}
