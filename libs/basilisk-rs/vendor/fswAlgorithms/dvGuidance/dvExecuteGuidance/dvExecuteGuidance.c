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

#include "fswAlgorithms/dvGuidance/dvExecuteGuidance/dvExecuteGuidance.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>
#include <math.h>

/*! This method initializes the configData for the nominal delta-V maneuver guidance.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the delta-V maneuver guidance
 @param moduleID The ID associated with the configData
 */
void SelfInit_dvExecuteGuidance(dvExecuteGuidanceConfig *configData, int64_t moduleID)
{
    DvExecutionDataMsg_C_init(&configData->burnExecOutMsg);
    THRArrayOnTimeCmdMsg_C_init(&configData->thrCmdOutMsg);
}


/*! @brief This resets the module.
 @return void
 @param configData The configuration data associated with this module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The unique module identifier
 */
void Reset_dvExecuteGuidance(dvExecuteGuidanceConfig *configData, uint64_t callTime,
                       int64_t moduleID)
{
    // check if the required input messages are included
    if (!NavTransMsg_C_isLinked(&configData->navDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: dvExecuteGuidance.navDataInMsg wasn't connected.");
    }
    if (!DvBurnCmdMsg_C_isLinked(&configData->burnDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: dvExecuteGuidance.burnDataInMsg wasn't connected.");
    }
}




/*! This method takes its own internal variables and creates an output attitude 
    command to use for burn execution.  It also flags whether the burn should 
    be happening or not.
 @return void
 @param configData The configuration data associated with the delta-V maneuver guidance
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_dvExecuteGuidance(dvExecuteGuidanceConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    double burnAccum[3];
    double dvExecuteMag;
	double burnTime;
    double dvMag;

    NavTransMsgPayload navData;
    DvBurnCmdMsgPayload localBurnData;
    DvExecutionDataMsgPayload localExeData;
    THRArrayOnTimeCmdMsgPayload effCmd;

    // read in messages
    navData = NavTransMsg_C_read(&configData->navDataInMsg);
    localBurnData = DvBurnCmdMsg_C_read(&configData->burnDataInMsg);

    burnTime = ((int64_t) callTime - (int64_t) localBurnData.burnStartTime)*1.0E-9;
    v3SetZero(burnAccum);
    if((configData->burnExecuting == 0 && burnTime >= 0.0)
        && configData->burnComplete != 1)
    {
        configData->burnExecuting = 1;
        v3Copy(navData.vehAccumDV, configData->dvInit);
        configData->burnComplete = 0;
    }

    if(configData->burnExecuting)
    {
        v3Subtract(navData.vehAccumDV, configData->dvInit, burnAccum);
    }
    
    dvMag = v3Norm(localBurnData.dvInrtlCmd);
    dvExecuteMag = v3Norm(burnAccum);
    configData->burnComplete = configData->burnComplete == 1 ||
        dvExecuteMag > dvMag;
    configData->burnExecuting = configData->burnComplete != 1 &&
        configData->burnExecuting == 1;
    
    if(configData->burnComplete)
    {
        effCmd = THRArrayOnTimeCmdMsg_C_zeroMsgPayload();
        THRArrayOnTimeCmdMsg_C_write(&effCmd, &configData->thrCmdOutMsg, moduleID, callTime);
    }

    localExeData = DvExecutionDataMsg_C_zeroMsgPayload();
    localExeData.burnComplete = configData->burnComplete;
    localExeData.burnExecuting = configData->burnExecuting;
    DvExecutionDataMsg_C_write(&localExeData, &configData->burnExecOutMsg, moduleID, callTime);
    
    return;
}

