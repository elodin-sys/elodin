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

#ifndef _THR_MOMENTUM_MANAGEMENT_H_
#define _THR_MOMENTUM_MANAGEMENT_H_

#include <stdint.h>

#include "cMsgCInterface/RWArrayConfigMsg_C.h"
#include "cMsgCInterface/RWSpeedMsg_C.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"

#include "architecture/utilities/bskLogging.h"






/*! @brief Module configuration message definition. */
typedef struct {
    /* declare module private variables */
    int initRequest;                                    //!< [-] status flag of the momentum dumping management
    RWArrayConfigMsgPayload rwConfigParams;             //!< [-] struct to store message containing RW config parameters in body B frame

    /* declare module public variables */
    double hs_min;                                      //!< [Nms]  minimum RW cluster momentum for dumping

    /* declare module IO interfaces */
    CmdTorqueBodyMsg_C deltaHOutMsg;                    //!< The name of the output message
    RWSpeedMsg_C rwSpeedsInMsg;                         //!< [] The name for the reaction wheel speeds message
    RWArrayConfigMsg_C rwConfigDataInMsg;               //!< [-] The name of the RWA configuration message

    BSKLogger *bskLogger;                             //!< BSK Logging

}thrMomentumManagementConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_thrMomentumManagement(thrMomentumManagementConfig *configData, int64_t moduleID);
    void Update_thrMomentumManagement(thrMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_thrMomentumManagement(thrMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
