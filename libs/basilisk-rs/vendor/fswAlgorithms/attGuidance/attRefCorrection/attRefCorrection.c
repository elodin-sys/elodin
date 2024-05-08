/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#include "fswAlgorithms/attGuidance/attRefCorrection/attRefCorrection.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "string.h"

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_attRefCorrection(attRefCorrectionConfig  *configData, int64_t moduleID)
{
    AttRefMsg_C_init(&configData->attRefOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
    time varying states between function calls are reset to their default values.
    Check if required input messages are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_attRefCorrection(attRefCorrectionConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!AttRefMsg_C_isLinked(&configData->attRefInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: attRefCorrection.attRefInMsg was not connected.");
    }
}


/*! Corrects the reference attitude message by a fixed rotation
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_attRefCorrection(attRefCorrectionConfig *configData, uint64_t callTime, int64_t moduleID)
{
    AttRefMsgPayload attRefMsgBuffer;     //!< local copy of message buffer
    double sigma_BBc[3];                    //!< MRP from corrected body frame to body frame

    // read in the input messages
    attRefMsgBuffer = AttRefMsg_C_read(&configData->attRefInMsg);

    // compute corrected reference orientation
    v3Scale(-1.0, configData->sigma_BcB, sigma_BBc);
    addMRP(attRefMsgBuffer.sigma_RN, sigma_BBc, attRefMsgBuffer.sigma_RN);

    // write to the output messages
    AttRefMsg_C_write(&attRefMsgBuffer, &configData->attRefOutMsg, moduleID, callTime);
}

