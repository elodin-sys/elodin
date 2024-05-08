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

#ifndef _SUNLINE_EPHEM_FSW_MSG_H_
#define _SUNLINE_EPHEM_FSW_MSG_H_

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/NavTransMsg_C.h"
#include "cMsgCInterface/EphemerisMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>



/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    /* declare module IO interfaces */
    NavAttMsg_C navStateOutMsg;                     /*!< The name of the output message*/
    EphemerisMsg_C sunPositionInMsg;           //!< The name of the sun ephemeris input message
    NavTransMsg_C scPositionInMsg;             //!< The name of the spacecraft ephemeris input message
    NavAttMsg_C scAttitudeInMsg;               //!< The name of the spacecraft attitude input message
    
    BSKLogger *bskLogger; //!< BSK Logging

}sunlineEphemConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_sunlineEphem(sunlineEphemConfig *configData, int64_t moduleID);
    void Update_sunlineEphem(sunlineEphemConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_sunlineEphem(sunlineEphemConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
