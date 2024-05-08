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

#include "fswAlgorithms/sensorInterfaces/IMUSensorData/imuComm.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>

/*! This method initializes the configData for theIMU sensor interface.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the IMU sensor interface
 @param moduleID The ID associated with the configData
 */
void SelfInit_imuProcessTelem(IMUConfigData *configData, int64_t moduleID)
{
    IMUSensorBodyMsg_C_init(&configData->imuSensorOutMsg);
    
}


/*! This method resets the module.
 @return void
 @param configData The configuration data associated with the OD filter
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_imuProcessTelem(IMUConfigData *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!IMUSensorMsg_C_isLinked(&configData->imuComInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: imuComm.imuComInMsg wasn't connected.");
    }
}

/*! This method takes the raw sensor data from the coarse sun sensors and
 converts that information to the format used by the IMU nav.
 @return void
 @param configData The configuration data associated with the IMU interface
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_imuProcessTelem(IMUConfigData *configData, uint64_t callTime, int64_t moduleID)
{
    IMUSensorMsgPayload LocalInput;

    // read imu com msg
    LocalInput = IMUSensorMsg_C_read(&configData->imuComInMsg);
    m33MultV3(RECAST3X3 configData->dcm_BP, LocalInput.DVFramePlatform,
              configData->outMsgBuffer.DVFrameBody);
    m33MultV3(RECAST3X3 configData->dcm_BP, LocalInput.AccelPlatform,
              configData->outMsgBuffer.AccelBody);
    m33MultV3(RECAST3X3 configData->dcm_BP, LocalInput.DRFramePlatform,
              configData->outMsgBuffer.DRFrameBody);
    m33MultV3(RECAST3X3 configData->dcm_BP, LocalInput.AngVelPlatform,
              configData->outMsgBuffer.AngVelBody);
    
    IMUSensorBodyMsg_C_write(&configData->outMsgBuffer, &configData->imuSensorOutMsg, moduleID, callTime);
    
    return;
}
