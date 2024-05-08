/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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

#include "fswAlgorithms/sensorInterfaces/scanningInstrumentController/scanningInstrumentController.h"
#include "architecture/utilities/linearAlgebra.h"

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_scanningInstrumentController(scanningInstrumentControllerConfig  *configData, int64_t moduleID)
{
    DeviceCmdMsg_C_init(&configData->deviceCmdOutMsg);
}

/*! This method checks if required input messages (accessInMsg and attGuidInMsg) are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_scanningInstrumentController(scanningInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!AccessMsg_C_isLinked(&configData->accessInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: scanningInstrumentController.accessInMsg was not connected.");
    }
    if (!AttGuidMsg_C_isLinked(&configData->attGuidInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: scanningInstrumentController.attGuidInMsg was not connected.");
    }
}

/*! This method checks the status of the device and if there is access to target, as well if the magnitude of the attitude 
error and attitude rate are within the tolerance. If so, the instrument is turned on, otherwise it is turned off.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_scanningInstrumentController(scanningInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID)
{
    double sigma_BR_norm; //!< Norm of sigma_BR
    double omega_BR_norm; //!< Norm of omega_BR

    AccessMsgPayload accessInMsgBuffer;  //!< local copy of message buffer
    AttGuidMsgPayload attGuidInMsgBuffer;  //!< local copy of message buffer
    DeviceStatusMsgPayload deviceStatusInMsgBuffer;  //!< local copy of message buffer
    DeviceCmdMsgPayload deviceCmdOutMsgBuffer;  //!< local copy of message buffer

    // always zero the output message buffers before assigning values
    deviceCmdOutMsgBuffer = DeviceCmdMsg_C_zeroMsgPayload();

    // read in the input messages
    accessInMsgBuffer = AccessMsg_C_read(&configData->accessInMsg);
    attGuidInMsgBuffer = AttGuidMsg_C_read(&configData->attGuidInMsg);

    // Read in the device status message if it is linked
    if (DeviceStatusMsg_C_isLinked(&configData->deviceStatusInMsg)) {
        deviceStatusInMsgBuffer = DeviceStatusMsg_C_read(&configData->deviceStatusInMsg);
        configData->controllerStatus = deviceStatusInMsgBuffer.deviceStatus;
    }

    // Compute the norms of the attitude and rate errors
    sigma_BR_norm = v3Norm(attGuidInMsgBuffer.sigma_BR);
    omega_BR_norm = v3Norm(attGuidInMsgBuffer.omega_BR_B);
    
    // If the controller is active
    if (configData->controllerStatus) {
        /* If the attitude error is less than the tolerance, the groundLocation is accessible, and (if enabled) the rate
        error is less than the tolerance, turn on the instrument and set the imaged indicator to 1*/
        if ((sigma_BR_norm <= configData->attErrTolerance)
            && (!configData->useRateTolerance || (omega_BR_norm <= configData->rateErrTolerance)) // Check rate tolerance if useRateTolerance enabled
            && (accessInMsgBuffer.hasAccess))
        {
            deviceCmdOutMsgBuffer.deviceCmd = 1;
        }
    }

    // write to the output messages
    DeviceCmdMsg_C_write(&deviceCmdOutMsgBuffer, &configData->deviceCmdOutMsg, moduleID, callTime);
}