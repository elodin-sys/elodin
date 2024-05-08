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
    Control Torque Low Pass Filter Module
 
 */

/* modify the path to reflect the new module names */
#include "fswAlgorithms/attControl/lowPassFilterTorqueCommand/lowPassFilterTorqueCommand.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/macroDefinitions.h"
#include "fswAlgorithms/fswUtilities/fswDefinitions.h"
#include "math.h"



/*! This method initializes the configData for this module.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_lowPassFilterTorqueCommand(lowPassFilterTorqueCommandConfig *configData, int64_t moduleID)
{
    /*! - Initialize output message for module */
    CmdTorqueBodyMsg_C_init(&configData->cmdTorqueOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the MRP steering control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Reset_lowPassFilterTorqueCommand(lowPassFilterTorqueCommandConfig *configData, uint64_t callTime, int64_t moduleID)
{
    int i;

    configData->reset  = BOOL_TRUE;         /* reset the first run flag */

    // check if the required input message is included
    if (!CmdTorqueBodyMsg_C_isLinked(&configData->cmdTorqueInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: lowPassFilterTorqueCommand.cmdTorqueInMsg wasn't connected.");
    }

    for (i=0;i<NUM_LPF;i++) {
        v3SetZero(configData->Lr[i]);
        v3SetZero(configData->LrF[i]);
    }
}

/*! This method takes the attitude and rate errors relative to the Reference frame, as well as
    the reference frame angular rates and acceleration, and computes the required control torque Lr.
 @return void
 @param configData The configuration data associated with the MRP Steering attitude control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_lowPassFilterTorqueCommand(lowPassFilterTorqueCommandConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    double      v3[3];                      /*!<      3d vector sub-result */
    int         i;
    CmdTorqueBodyMsgPayload controlOut;             /*!< -- Control output message */

    /* zero the output message */
    controlOut = CmdTorqueBodyMsg_C_zeroMsgPayload();

    /* - Read the input messages */
    CmdTorqueBodyMsgPayload msgBuffer = CmdTorqueBodyMsg_C_read(&configData->cmdTorqueInMsg);
    v3Copy(msgBuffer.torqueRequestBody, configData->Lr[0]);

    /*
        check if the filter states must be reset
     */
    if (configData->reset) {
        /* populate the filter history with 1st input */
        for (i=1;i<NUM_LPF;i++) {
            v3Copy(configData->Lr[0], configData->Lr[i]);
        }

        /* zero the history of filtered outputs */
        for (i=0;i<NUM_LPF;i++) {
            v3SetZero(configData->LrF[i]);
        }

        /* compute h times the prewarped critical filter frequency */
        configData->hw = tan(configData->wc * configData->h / 2.0)*2.0;

        /* determine 1st order low-pass filter coefficients */
        configData->a[0] = 2.0 + configData->hw;
        configData->a[1] = 2.0 - configData->hw;
        configData->b[0] = configData->hw;
        configData->b[1] = configData->hw;

        /* turn off first run flag */
        configData->reset = BOOL_FALSE;

    }
    
    /*
        regular filter run
     */

    v3SetZero(configData->LrF[0]);
    for (i=0;i<NUM_LPF;i++) {
        v3Scale(configData->b[i], configData->Lr[i], v3);
        v3Add(v3, configData->LrF[0], configData->LrF[0]);
    }
    for (i=1;i<NUM_LPF;i++) {
        v3Scale(configData->a[i], configData->LrF[i], v3);
        v3Add(v3, configData->LrF[0], configData->LrF[0]);
    }
    v3Scale(1.0/configData->a[0], configData->LrF[0], configData->LrF[0]);


    /* reset the filter state history */
    for (i=1;i<NUM_LPF;i++) {
        v3Copy(configData->Lr[NUM_LPF-1-i],  configData->Lr[NUM_LPF-i]);
        v3Copy(configData->LrF[NUM_LPF-1-i], configData->LrF[NUM_LPF-i]);
    }

    /*
        store the output message 
     */
    v3Copy(configData->LrF[0], controlOut.torqueRequestBody);
    CmdTorqueBodyMsg_C_write(&controlOut, &configData->cmdTorqueOutMsg, moduleID, callTime);
    
    return;
}
