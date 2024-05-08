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

#include "fswAlgorithms/sensorInterfaces/STSensorData/stComm.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>

/*! This method initializes the configData for the ST sensor interface.
 @return void
 @param configData The configuration data associated with the ST sensor interface
 @param moduleID The ID associated with the configData
 */
void SelfInit_stProcessTelem(STConfigData *configData, int64_t moduleID)
{
    STAttMsg_C_init(&configData->stAttOutMsg);
}



/*! This method resets the module.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_stProcessTelem(STConfigData *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!STSensorMsg_C_isLinked(&configData->stSensorInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: stComm.stSensorInMsg wasn't connected.");
    }
}

/*! This method takes the raw sensor data from the star tracker and
 converts that information to the format used by the ST nav.
 @return void
 @param configData The configuration data associated with the ST interface
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_stProcessTelem(STConfigData *configData, uint64_t callTime, int64_t moduleID)
{
    double dcm_CN[3][3];            /* dcm, inertial to case frame */
    double dcm_BN[3][3];            /* dcm, inertial to body frame */
    STSensorMsgPayload localInput;

    // read input msg
    localInput = STSensorMsg_C_read(&configData->stSensorInMsg);

    EP2C(localInput.qInrtl2Case, dcm_CN);
    m33MultM33(RECAST3X3 configData->dcm_BP, dcm_CN, dcm_BN);

    C2MRP(dcm_BN, configData->attOutBuffer.MRP_BdyInrtl);
    configData->attOutBuffer.timeTag = localInput.timeTag;

    STAttMsg_C_write(&configData->attOutBuffer, &configData->stAttOutMsg, moduleID, callTime);
    
    return;
}
