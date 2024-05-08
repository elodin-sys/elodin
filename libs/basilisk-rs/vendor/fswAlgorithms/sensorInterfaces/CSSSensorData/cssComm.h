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

#ifndef _CSS_COMM_H_
#define _CSS_COMM_H_


#define MAX_NUM_CHEBY_POLYS 32

#include "cMsgCInterface/CSSArraySensorMsg_C.h"

#include "architecture/utilities/bskLogging.h"



/*! @brief Top level structure for the CSS sensor interface system.  Contains all parameters for the
 CSS interface*/
typedef struct {
    uint32_t  numSensors;   //!< The number of sensors we are processing
    CSSArraySensorMsg_C sensorListInMsg; //!< input message that contains CSS data
    CSSArraySensorMsg_C cssArrayOutMsg; //!< output message of corrected CSS data

    CSSArraySensorMsgPayload inputValues; //!< Input values we took off the messaging system
    double maxSensorValue; //!< Scale factor to go from sensor values to cosine
    uint32_t chebyCount; //!< Count on the number of chebyshev polynominals we have
    double kellyCheby[MAX_NUM_CHEBY_POLYS]; //!< Chebyshev polynominals to fit output to cosine
    BSKLogger *bskLogger;                             //!< BSK Logging
}CSSConfigData;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_cssProcessTelem(CSSConfigData *configData, int64_t moduleID);
    void Update_cssProcessTelem(CSSConfigData *configData, uint64_t callTime, int64_t moduleID);
    void Reset_cssProcessTelem(CSSConfigData *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
