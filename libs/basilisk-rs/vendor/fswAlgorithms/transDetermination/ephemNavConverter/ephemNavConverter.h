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

#ifndef _EPHEM_NAV_CONVERTER_H_
#define _EPHEM_NAV_CONVERTER_H_

#include "architecture/utilities/bskLogging.h"

#include "cMsgCInterface/EphemerisMsg_C.h"
#include "cMsgCInterface/NavTransMsg_C.h"


/*! @brief The configuration structure for the ephemNavConverter module.*/
typedef struct {
    NavTransMsg_C stateOutMsg; //!< [-] output navigation message for pos/vel
    EphemerisMsg_C ephInMsg; //!< ephemeris input message

    BSKLogger *bskLogger;   //!< BSK Logging
}EphemNavConverterData;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_ephemNavConverter(EphemNavConverterData *configData, int64_t moduleID);
    void Update_ephemNavConverter(EphemNavConverterData *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_ephemNavConverter(EphemNavConverterData *configData, uint64_t callTime,
                              int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
