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

#ifndef _RW_CONFIG_DATA_H_
#define _RW_CONFIG_DATA_H_

#include "cMsgCInterface/RWArrayConfigMsg_C.h"
#include "cMsgCInterface/RWConstellationMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>



/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    /* declare module private variables */
    RWConstellationMsgPayload rwConstellation; /*!< struct to populate input RW config parameters in structural S frame */
    RWArrayConfigMsgPayload  rwConfigParamsOut; /*!< struct to populate ouput RW config parameters in body B frame */
    /* declare module IO interfaces */
    RWConstellationMsg_C rwConstellationInMsg;          /*!< RW array input message */
    RWArrayConfigMsg_C rwParamsOutMsg;                  /*!< RW array output message */

    BSKLogger *bskLogger;   //!< BSK Logging

}rwConfigData_Config;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_rwConfigData(rwConfigData_Config*configData, int64_t moduleID);
    void Update_rwConfigData(rwConfigData_Config *configData, uint64_t callTime, int64_t moduleID);
    void Reset_rwConfigData(rwConfigData_Config *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
