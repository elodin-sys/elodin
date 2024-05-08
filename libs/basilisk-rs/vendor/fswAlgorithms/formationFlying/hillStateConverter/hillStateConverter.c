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

/* modify the path to reflect the new module names */
#include "hillStateConverter.h"
#include "string.h"

// Internal utilities
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/orbitalMotion.h"


/*!
 \verbatim embed:rst
    This method initializes the module's hillStateOutMsg.
 \endverbatim
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_hillStateConverter(HillStateConverterConfig *configData, int64_t moduleID)
{
    HillRelStateMsg_C_init(&configData->hillStateOutMsg);
}

/*! This message checks to see that both of the input translational state messages were connected; if not, it errors.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_hillStateConverter(HillStateConverterConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!NavTransMsg_C_isLinked(&configData->chiefStateInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: hillStateConverter.chiefStateInMsg wasn't connected.");
    }
    if (!NavTransMsg_C_isLinked(&configData->depStateInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: hillStateConverter.depStateInMsg wasn't connected.");
    }
}

/*! Computes the relative state of the deputy vs the chief in chief Hill-frame coordinates and writes an output message.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_hillStateConverter(HillStateConverterConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! - Read the input messages */
    NavTransMsgPayload chiefStateIn;
    NavTransMsgPayload depStateIn;
    HillRelStateMsgPayload hillStateOut;
    chiefStateIn = NavTransMsg_C_read(&configData->chiefStateInMsg);
    depStateIn = NavTransMsg_C_read(&configData->depStateInMsg);
    
    hillStateOut = HillRelStateMsg_C_zeroMsgPayload();
    
    /*! - Add the module specific code */
    rv2hill(chiefStateIn.r_BN_N, chiefStateIn.v_BN_N,
            depStateIn.r_BN_N,  depStateIn.v_BN_N,
            hillStateOut.r_DC_H, hillStateOut.v_DC_H);

    /*! - write the module output message */
    HillRelStateMsg_C_write(&hillStateOut, &configData->hillStateOutMsg, moduleID, callTime);

    return;
}
