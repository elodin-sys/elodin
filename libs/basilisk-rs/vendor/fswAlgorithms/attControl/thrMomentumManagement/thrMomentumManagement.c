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
/*
    Thruster RW Momentum Management
 
 */

#include "fswAlgorithms/attControl/thrMomentumManagement/thrMomentumManagement.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"
#include <string.h>


/*!
 \verbatim embed:rst
    This method initializes the configData for this module.  It creates a single output message of type
    :ref:`CmdTorqueBodyMsgPayload`.
 \endverbatim
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
*/
void SelfInit_thrMomentumManagement(thrMomentumManagementConfig *configData, int64_t moduleID)
{
    CmdTorqueBodyMsg_C_init(&configData->deltaHOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Reset_thrMomentumManagement(thrMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!RWArrayConfigMsg_C_isLinked(&configData->rwConfigDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: thrMomentumManagement.rwConfigDataInMsg wasn't connected.");
    }
    if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: thrMomentumManagement.rwSpeedsInMsg wasn't connected.");
    }

    /*! - read in the RW configuration message */
    configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwConfigDataInMsg);

    /*! - reset the momentum dumping request flag */
    configData->initRequest = 1;
}

/*! The RW momentum level is assessed to determine if a momentum dumping maneuver is required.
 This checking only happens once after the reset function is called.  To run this again afterwards,
 the reset function must be called again.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_thrMomentumManagement(thrMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID)
{
    RWSpeedMsgPayload   rwSpeedMsg;         /* Reaction wheel speed estimate message */
    CmdTorqueBodyMsgPayload controlOutMsg;  /* Control torque output message */
    double              hs;                 /* net RW cluster angular momentum magnitude */
    double              hs_B[3];            /* RW angular momentum */
    double              vec3[3];            /* temp vector */
    double              Delta_H_B[3];       /* [Nms]  net desired angular momentum change */
    int i;

    /*! - check if a momentum dumping check has been requested */
    if (configData->initRequest == 1) {

        /*! - Read the input messages */
        rwSpeedMsg = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);

        /*! - compute net RW momentum magnitude */
        v3SetZero(hs_B);
        for (i=0;i<configData->rwConfigParams.numRW;i++) {
            v3Scale(configData->rwConfigParams.JsList[i]*rwSpeedMsg.wheelSpeeds[i],&configData->rwConfigParams.GsMatrix_B[i*3],vec3);
            v3Add(hs_B, vec3, hs_B);
        }
        hs = v3Norm(hs_B);

        /*! - check if momentum dumping is required */
        if (hs < configData->hs_min) {
            /* Momentum dumping not required */
            v3SetZero(Delta_H_B);
        } else {
            v3Scale(-(hs - configData->hs_min)/hs, hs_B, Delta_H_B);
        }
        configData->initRequest = 0;


        /*! - write out the output message */
        controlOutMsg = CmdTorqueBodyMsg_C_zeroMsgPayload();
        v3Copy(Delta_H_B, controlOutMsg.torqueRequestBody);

        CmdTorqueBodyMsg_C_write(&controlOutMsg, &configData->deltaHOutMsg, moduleID, callTime);

    }

    return;
}
