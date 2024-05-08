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

#ifndef _CSS_WLS_EST_H_
#define _CSS_WLS_EST_H_

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/CSSConfigMsg_C.h"
#include "cMsgCInterface/CSSUnitConfigMsg_C.h"
#include "cMsgCInterface/CSSArraySensorMsg_C.h"
#include "cMsgCInterface/SunlineFilterMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>




/*! @brief Top level structure for the CSS weighted least squares estimator.
 Used to estimate the sun state in the vehicle body frame*/
typedef struct {
    CSSArraySensorMsg_C cssDataInMsg;                   //!< The name of the CSS sensor input message
    CSSConfigMsg_C cssConfigInMsg;                      //!< The name of the CSS configuration input message
    NavAttMsg_C navStateOutMsg;                         //!< The name of the navigation output message containing the estimated states
    SunlineFilterMsg_C cssWLSFiltResOutMsg;             //!< The name of the CSS filter data out message

    uint32_t numActiveCss;                              //!< [-] Number of currently active CSS sensors
    uint32_t useWeights;                                //!< Flag indicating whether or not to use weights for least squares
    uint32_t priorSignalAvailable;                      //!< Flag indicating if a recent prior heading estimate is available
    double dOld[3];                                     //!< The prior sun heading estimate
    double sensorUseThresh;                             //!< Threshold below which we discount sensors
    uint64_t priorTime;                                 //!< [ns] Last time the attitude control is called
    CSSConfigMsgPayload cssConfigInBuffer;              //!< CSS constellation configuration message buffer
    SunlineFilterMsgPayload filtStatus;                 //!< Filter message

    BSKLogger *bskLogger;                               //!< BSK Logging
}CSSWLSConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_cssWlsEst(CSSWLSConfig *configData, int64_t moduleID);
    void Update_cssWlsEst(CSSWLSConfig *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_cssWlsEst(CSSWLSConfig *configData, uint64_t callTime, int64_t moduleID);
    int computeWlsmn(int numActiveCss, double *H, double *W,
                     double *y, double x[3]);
    void computeWlsResiduals(double *cssMeas, CSSConfigMsgPayload *cssConfig,
                             double *wlsEst, double *cssResiduals);
    
#ifdef __cplusplus
}
#endif


#endif
