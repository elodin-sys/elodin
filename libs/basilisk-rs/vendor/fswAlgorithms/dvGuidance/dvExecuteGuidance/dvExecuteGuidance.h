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

#ifndef _DV_EXECUTE_GUIDANCE_H_
#define _DV_EXECUTE_GUIDANCE_H_

#include "cMsgCInterface/NavTransMsg_C.h"
#include "cMsgCInterface/THRArrayOnTimeCmdMsg_C.h"
#include "cMsgCInterface/DvBurnCmdMsg_C.h"
#include "cMsgCInterface/DvExecutionDataMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>



/*! @brief Top level structure for the nominal delta-V guidance */
typedef struct {
    DvExecutionDataMsg_C burnExecOutMsg; /*!< [-] The name of burn execution output message*/
    NavTransMsg_C navDataInMsg; /*!< [-] The name of the incoming attitude command */
    DvBurnCmdMsg_C burnDataInMsg;/*!< [-] Input message that configures the vehicle burn*/
    THRArrayOnTimeCmdMsg_C thrCmdOutMsg; /*!< [-] Output thruster message name */
    double dvInit[3];        /*!< (m/s) DV reading off the accelerometers at burn start*/
    uint32_t burnExecuting;  /*!< (-) Flag indicating whether the burn is in progress or not*/
    uint32_t burnComplete;   /*!< (-) Flag indicating that burn has completed successfully*/

    BSKLogger *bskLogger;   //!< BSK Logging
}dvExecuteGuidanceConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_dvExecuteGuidance(dvExecuteGuidanceConfig *configData, int64_t moduleID);
    void Update_dvExecuteGuidance(dvExecuteGuidanceConfig *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_dvExecuteGuidance(dvExecuteGuidanceConfig *configData, uint64_t callTime,
                       int64_t moduleID);

#ifdef __cplusplus
}
#endif


#endif
