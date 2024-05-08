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
    Rate Converter message

    Note:   this module reads in a message of type ImuSensorBodyMsgPayload, extracts the body rate vector information,
            and adds this info to a msg of type NavAttMsgPayload.
    Author: Hanspeter Schaub
    Date:   June 30, 2018
 
 */

#include <string.h>
#include "rateMsgConverter.h"
#include "architecture/utilities/linearAlgebra.h"

/*! This method initializes the configData for this module.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The Basilisk module identifier
 */
void SelfInit_rateMsgConverter(rateMsgConverterConfig *configData, int64_t moduleID)
{
    NavAttMsg_C_init(&configData->navRateOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Reset_rateMsgConverter(rateMsgConverterConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!IMUSensorBodyMsg_C_isLinked(&configData->imuRateInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rateMsgConverter.imuRateInMsg wasn't connected.");
    }
}

/*! This method performs a time step update of the module.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Update_rateMsgConverter(rateMsgConverterConfig *configData, uint64_t callTime, int64_t moduleID)
{
    IMUSensorBodyMsgPayload inMsg;
    NavAttMsgPayload outMsg;
    
    /*! - read in the message of type IMUSensorBodyMsgPayload */
    inMsg = IMUSensorBodyMsg_C_read(&configData->imuRateInMsg);
    
    /*! - create a zero message of type NavAttMsgPayload which has the rate vector from the input message */
    outMsg = NavAttMsg_C_zeroMsgPayload();
    v3Copy(inMsg.AngVelBody, outMsg.omega_BN_B);
    
    /*! - write output message */
    NavAttMsg_C_write(&outMsg, &configData->navRateOutMsg, moduleID, callTime);
    
    return;
}
