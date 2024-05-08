/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

/*
    simpleInstrumentController Module

 */

#include "fswAlgorithms/sensorInterfaces/simpleInstrumentController/simpleInstrumentController.h"
#include "architecture/utilities/linearAlgebra.h"
#include <stdio.h>

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_simpleInstrumentController(simpleInstrumentControllerConfig *configData, int64_t moduleID)
{
    configData->imaged = 0;
    configData->controllerStatus = 1;
    DeviceCmdMsg_C_init(&configData->deviceCmdOutMsg);
}

/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_simpleInstrumentController(simpleInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!AccessMsg_C_isLinked(&configData->locationAccessInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: simpleInstrumentController.locationAccessInMsg wasn't connected.");
    }
    if (!AttGuidMsg_C_isLinked(&configData->attGuidInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: simpleInstrumentController.attGuidInMsg wasn't connected.");
    }

    // reset the imaged variable to zero
    configData->imaged = 0;
}

/*! Add a description of what this main Update() routine does for this module
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_simpleInstrumentController(simpleInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID)
{
    double sigma_BR_norm; //!< Norm of sigma_BR
    double omega_BR_norm; //!< Norm of omega_BR

    /* Local copies of the msg buffers*/
    AccessMsgPayload accessInMsgBuffer;  //!< local copy of input message buffer
    AttGuidMsgPayload attGuidInMsgBuffer;  //!< local copy of output message buffer
    DeviceStatusMsgPayload deviceStatusInMsgBuffer; //!< local copy of input message buffer
    DeviceCmdMsgPayload deviceCmdOutMsgBuffer;  //!< local copy of output message buffer

    // zero output buffer
    deviceCmdOutMsgBuffer = DeviceCmdMsg_C_zeroMsgPayload();

    // read in the input messages
    accessInMsgBuffer = AccessMsg_C_read(&configData->locationAccessInMsg);
    attGuidInMsgBuffer = AttGuidMsg_C_read(&configData->attGuidInMsg);

    // read in the device cmd message if it is connected
    if (DeviceStatusMsg_C_isLinked(&configData->deviceStatusInMsg)) {
        deviceStatusInMsgBuffer = DeviceStatusMsg_C_read(&configData->deviceStatusInMsg);
        configData->controllerStatus = deviceStatusInMsgBuffer.deviceStatus;
    }

    // Compute the norms of the attitude and rate errors
    sigma_BR_norm = v3Norm(attGuidInMsgBuffer.sigma_BR);
    omega_BR_norm = v3Norm(attGuidInMsgBuffer.omega_BR_B);

    // If the controller is active
    if (configData->controllerStatus) {
        // If the target has not been imaged
        if (!configData->imaged) {
            /* If the attitude error is less than the tolerance, the groundLocation is accessible, and (if enabled) the rate
            error is less than the tolerance, turn on the instrument and set the imaged indicator to 1*/
            if ((sigma_BR_norm <= configData->attErrTolerance)
                && (!configData->useRateTolerance || (omega_BR_norm <= configData->rateErrTolerance)) // Check rate tolerance if useRateTolerance enabled
                && (accessInMsgBuffer.hasAccess))
            {
                deviceCmdOutMsgBuffer.deviceCmd = 1;
                configData->imaged = 1;
                // Otherwise, turn off the instrument
            }
        }
    }

    // write to the output messages
    DeviceCmdMsg_C_write(&deviceCmdOutMsgBuffer, &(configData->deviceCmdOutMsg), moduleID, callTime);

    return;
}
