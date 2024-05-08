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

#include "navAggregate.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"
#include <string.h>
#include <stdio.h>

/*! This method initializes the configData for the nav aggregation algorithm.
    It initializes the output messages in the messaging system.
 @return void
 @param configData The configuration data associated with the Nav aggregation interface
 @param moduleID The Basilisk module identifier
 */
void SelfInit_aggregateNav(NavAggregateData *configData, int64_t moduleID)
{
    NavAttMsg_C_init(&configData->navAttOutMsg);
    NavTransMsg_C_init(&configData->navTransOutMsg);
}


/*! This resets the module to original states.
 @return void
 @param configData The configuration data associated with this module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_aggregateNav(NavAggregateData *configData, uint64_t callTime, int64_t moduleID)
{

    /*! - ensure incoming message counters are not larger than MAX_AGG_NAV_MSG */
    if (configData->attMsgCount > MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        snprintf(info, MAX_LOGGING_LENGTH, "The attitude message count %d is larger than allowed (%d). Setting count to max value.",
                  configData->attMsgCount, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->attMsgCount = MAX_AGG_NAV_MSG;
    }
    if (configData->transMsgCount > MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The translation message count %d is larger than allowed (%d). Setting count to max value.",
                  configData->transMsgCount, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->transMsgCount = MAX_AGG_NAV_MSG;
    }

    /*! - loop over the number of attitude input messages and make sure they are linked */
    for(uint32_t i=0; i<configData->attMsgCount; i=i+1)
    {
        if (!NavAttMsg_C_isLinked(&configData->attMsgs[i].navAttInMsg)) {
            _bskLog(configData->bskLogger, BSK_ERROR, "An attitude input message name was not linked.  Be sure that attMsgCount is set properly.");
        }
    }
    /*! - loop over the number of translational input messages and make sure they are linked */
    for(uint32_t i=0; i<configData->transMsgCount; i=i+1)
    {
        if (!NavTransMsg_C_isLinked(&configData->transMsgs[i].navTransInMsg)) {
            _bskLog(configData->bskLogger, BSK_ERROR, "A translation input message name was not specified.  Be sure that transMsgCount is set properly.");
        }
    }

    /*! - ensure the attitude message index locations are less than MAX_AGG_NAV_MSG */
    if (configData->attTimeIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The attTimeIdx variable %d is too large. Must be less than %d. Setting index to max value.",
              configData->attTimeIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->attTimeIdx = MAX_AGG_NAV_MSG - 1;
    }
    if (configData->attIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The attIdx variable %d is too large. Must be less than %d. Setting index to max value.",
                  configData->attIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->attIdx = MAX_AGG_NAV_MSG - 1;
    }
    if (configData->rateIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The rateIdx variable %d is too large. Must be less than %d. Setting index to max value.",
                  configData->rateIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->rateIdx = MAX_AGG_NAV_MSG - 1;
    }
    if (configData->sunIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The sunIdx variable %d is too large. Must be less than %d. Setting index to max value.",
                configData->sunIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->sunIdx = MAX_AGG_NAV_MSG - 1;
    }

    /*! - ensure the translational message index locations are less than MAX_AGG_NAV_MSG */
    if (configData->transTimeIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The transTimeIdx variable %d is too large. Must be less than %d. Setting index to max value.",
                configData->transTimeIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->transTimeIdx = MAX_AGG_NAV_MSG - 1;
    }
    if (configData->posIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The posIdx variable %d is too large. Must be less than %d. Setting index to max value.",
                  configData->posIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->posIdx = MAX_AGG_NAV_MSG - 1;
    }
    if (configData->velIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The velIdx variable %d is too large. Must be less than %d. Setting index to max value.",
                  configData->velIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->velIdx = MAX_AGG_NAV_MSG - 1;
    }
    if (configData->dvIdx >= MAX_AGG_NAV_MSG) {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The dvIdx variable %d is too large. Must be less than %d. Setting index to max value.",
                configData->dvIdx, MAX_AGG_NAV_MSG);
        _bskLog(configData->bskLogger, BSK_ERROR, info);

        configData->dvIdx = MAX_AGG_NAV_MSG - 1;
    }

    //! - zero the arrays of input messages
    for (uint32_t i=0; i< MAX_AGG_NAV_MSG; i++) {
        configData->attMsgs[i].msgStorage = NavAttMsg_C_zeroMsgPayload();
        configData->transMsgs[i].msgStorage = NavTransMsg_C_zeroMsgPayload();
    }

}


/*! This method takes the navigation message snippets created by the various
    navigation components in the FSW and aggregates them into a single complete
    navigation message.
 @return void
 @param configData The configuration data associated with the aggregate nav module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Update_aggregateNav(NavAggregateData *configData, uint64_t callTime, int64_t moduleID)
{
    uint32_t i;
    NavAttMsgPayload navAttOutMsgBuffer;     /* [-] The local storage of the outgoing attitude navibation message data*/
    NavTransMsgPayload navTransOutMsgBuffer; /* [-] The local storage of the outgoing message data*/

    /*! - zero the output message buffers */
    navAttOutMsgBuffer = NavAttMsg_C_zeroMsgPayload();
    navTransOutMsgBuffer = NavTransMsg_C_zeroMsgPayload();

    /*! - check that attitude navigation messages are present */
    if (configData->attMsgCount) {
        /*! - Iterate through all of the attitude input messages, clear local Msg buffer and archive the new nav data */
        for(i=0; i<configData->attMsgCount; i=i+1)
        {
            configData->attMsgs[i].msgStorage = NavAttMsg_C_read(&configData->attMsgs[i].navAttInMsg);
        }

        /*! - Copy out each part of the attitude source message into the target output message*/
        navAttOutMsgBuffer.timeTag = configData->attMsgs[configData->attTimeIdx].msgStorage.timeTag;
        v3Copy(configData->attMsgs[configData->attIdx].msgStorage.sigma_BN, navAttOutMsgBuffer.sigma_BN);
        v3Copy(configData->attMsgs[configData->rateIdx].msgStorage.omega_BN_B, navAttOutMsgBuffer.omega_BN_B);
        v3Copy(configData->attMsgs[configData->sunIdx].msgStorage.vehSunPntBdy, navAttOutMsgBuffer.vehSunPntBdy);

    }

    /*! - check that translation navigation messages are present */
    if (configData->transMsgCount) {
        /*! - Iterate through all of the translation input messages, clear local Msg buffer and archive the new nav data */
        for(i=0; i<configData->transMsgCount; i=i+1)
        {
            configData->transMsgs[i].msgStorage = NavTransMsg_C_read(&configData->transMsgs[i].navTransInMsg);
        }

        /*! - Copy out each part of the translation source message into the target output message*/
        navTransOutMsgBuffer.timeTag = configData->transMsgs[configData->transTimeIdx].msgStorage.timeTag;
        v3Copy(configData->transMsgs[configData->posIdx].msgStorage.r_BN_N, navTransOutMsgBuffer.r_BN_N);
        v3Copy(configData->transMsgs[configData->velIdx].msgStorage.v_BN_N, navTransOutMsgBuffer.v_BN_N);
        v3Copy(configData->transMsgs[configData->dvIdx].msgStorage.vehAccumDV, navTransOutMsgBuffer.vehAccumDV);
    }

    /*! - Write the total message out for everyone else to pick up */
    NavAttMsg_C_write(&navAttOutMsgBuffer, &configData->navAttOutMsg, moduleID, callTime);
    NavTransMsg_C_write(&navTransOutMsgBuffer, &configData->navTransOutMsg, moduleID, callTime);

    return;
}
