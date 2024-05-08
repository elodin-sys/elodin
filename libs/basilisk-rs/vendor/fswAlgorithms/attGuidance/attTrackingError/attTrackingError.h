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

#ifndef _ATT_TRACKING_ERROR_
#define _ATT_TRACKING_ERROR_

#include "cMsgCInterface/AttGuidMsg_C.h"
#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/AttRefMsg_C.h"

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"



/*!@brief Data structure for module to compute the attitude tracking error between the spacecraft attitude and the reference.
 */
typedef struct {
    /* declare module private variables */
    double sigma_R0R[3];                        //!< MRP from corrected reference frame to original reference frame R0. This is the same as [BcB] going from primary body frame B to the corrected body frame Bc
    AttGuidMsg_C attGuidOutMsg;              //!< output msg of attitude guidance
    NavAttMsg_C attNavInMsg;                 //!< input msg measured attitude
    AttRefMsg_C attRefInMsg;                 //!< input msg of reference attitude
    BSKLogger *bskLogger;                       //!< BSK Logging
}attTrackingErrorConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_attTrackingError(attTrackingErrorConfig *configData, int64_t moduleID);
    void Update_attTrackingError(attTrackingErrorConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_attTrackingError(attTrackingErrorConfig *configData, uint64_t callTime, int64_t moduleID);
    void computeAttitudeError(double sigma_R0R[3], NavAttMsgPayload nav, AttRefMsgPayload ref, AttGuidMsgPayload *attGuidOut);

#ifdef __cplusplus
}
#endif


#endif
