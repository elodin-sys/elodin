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

#include "fswAlgorithms/effectorInterfaces/errorConversion/sunSafeACS.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <string.h>
#include <math.h>

/*! This method initializes the configData for the sun safe ACS control.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the sun safe control
 @param moduleID The ID associated with the configData
 */
void SelfInit_sunSafeACS(sunSafeACSConfig *configData, int64_t moduleID)
{
    THRArrayOnTimeCmdMsg_C_init(&configData->thrData.thrOnTimeOutMsg);
}

/*! This method resets the module.
 @return void
 @param configData The configuration data associated with the sun safe ACS control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_sunSafeACS(sunSafeACSConfig *configData, uint64_t callTime,
                        int64_t moduleID)
{
    // check if the required input messages are included
    if (!CmdTorqueBodyMsg_C_isLinked(&configData->cmdTorqueBodyInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunSafeACS.cmdTorqueBodyInMsg wasn't connected.");
    }
}


/*! This method takes the estimated body-observed sun vector and computes the
 current attitude/attitude rate errors to pass on to control.
 @return void
 @param configData The configuration data associated with the sun safe ACS control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_sunSafeACS(sunSafeACSConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    CmdTorqueBodyMsgPayload cntrRequest;

    /*! - Read the input parsed CSS sensor data message*/
    cntrRequest = CmdTorqueBodyMsg_C_read(&configData->cmdTorqueBodyInMsg);
    computeSingleThrustBlock(&(configData->thrData), callTime,
                             &cntrRequest, moduleID);
    
    return;
}
