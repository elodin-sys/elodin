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

#ifndef _ST_COMM_H_
#define _ST_COMM_H_

#include "cMsgCInterface/STSensorMsg_C.h"
#include "cMsgCInterface/STAttMsg_C.h"

#include "architecture/utilities/bskLogging.h"


/*! @brief Module configuration message.  */
typedef struct {
    double dcm_BP[9];                /*!< Row major platform 2 body DCM*/
    STSensorMsg_C stSensorInMsg;  /*!< star tracker sensor input message*/
    STAttMsg_C stAttOutMsg; /*!< star tracker attitude output message */

    STAttMsgPayload attOutBuffer; /*!< Output data structure*/
    BSKLogger *bskLogger;   //!< BSK Logging
}STConfigData;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_stProcessTelem(STConfigData *configData, int64_t moduleID);
    void Reset_stProcessTelem(STConfigData *configData, uint64_t callTime, int64_t moduleID);
    void Update_stProcessTelem(STConfigData *configData, uint64_t callTime,
        int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
