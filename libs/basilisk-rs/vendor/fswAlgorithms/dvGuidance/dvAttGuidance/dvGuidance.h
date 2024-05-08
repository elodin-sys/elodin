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

#ifndef _DV_GUIDANCE_POINT_H_
#define _DV_GUIDANCE_POINT_H_

#include "cMsgCInterface/AttRefMsg_C.h"
#include "cMsgCInterface/DvBurnCmdMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>




/*! @brief Top level structure for the nominal delta-V guidance
 */
typedef struct {
    AttRefMsg_C attRefOutMsg;           //!< The name of the output message
    DvBurnCmdMsg_C burnDataInMsg;       //!< Input message that configures the vehicle burn

    BSKLogger *bskLogger;   //!< BSK Logging
}dvGuidanceConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_dvGuidance(dvGuidanceConfig *configData, int64_t moduleID);
    void Update_dvGuidance(dvGuidanceConfig *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_dvGuidance(dvGuidanceConfig *configData, uint64_t callTime,
                           int64_t moduleID);

#ifdef __cplusplus
}
#endif


#endif
