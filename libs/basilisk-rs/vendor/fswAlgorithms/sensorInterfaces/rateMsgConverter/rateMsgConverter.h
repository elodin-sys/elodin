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

#ifndef _RATE_IMU_TO_NAV_CONVERTER_H_
#define _RATE_IMU_TO_NAV_CONVERTER_H_

#include <stdint.h>

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/IMUSensorBodyMsg_C.h"

#include "architecture/utilities/bskLogging.h"



/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    /* declare module IO interfaces */
    NavAttMsg_C navRateOutMsg;                        //!< attitude output message*/
    IMUSensorBodyMsg_C imuRateInMsg;                  //!< attitude Input message*/

    BSKLogger *bskLogger;                             //!< BSK Logging
}rateMsgConverterConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_rateMsgConverter(rateMsgConverterConfig *configData, int64_t moduleID);
    void Update_rateMsgConverter(rateMsgConverterConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_rateMsgConverter(rateMsgConverterConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
