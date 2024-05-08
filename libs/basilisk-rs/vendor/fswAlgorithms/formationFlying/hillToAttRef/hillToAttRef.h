/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef _HILL_TO_ATT_H
#define _HILL_TO_ATT_H

#include <stdint.h>
#include <string.h>

#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/HillRelStateMsg_C.h"
#include "cMsgCInterface/AttRefMsg_C.h"
#include "cMsgCInterface/NavAttMsg_C.h"




/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    /* declare module IO interfaces */
    HillRelStateMsg_C hillStateInMsg;               //!< Provides state relative to chief
    AttRefMsg_C attRefInMsg;                        //!< (Optional) Provides basis for relative attitude
    NavAttMsg_C attNavInMsg;                        //!< (Optional) Provides basis for relative attitude
    AttRefMsg_C attRefOutMsg;                       //!< Provides the attitude reference output message. 
    BSKLogger *bskLogger;                           //!< BSK Logging

    double gainMatrix[3][6]; //!< User-configured gain matrix that maps from hill states to relative attitudes.
    double relMRPMin;        //!< Minimum value for the relative MRP components; user-configurable.
    double relMRPMax;        //!< Maximum value for the relative MRP components; user-configurable.
}HillToAttRefConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_hillToAttRef(HillToAttRefConfig *configData, int64_t moduleID);
    void Update_hillToAttRef(HillToAttRefConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_hillToAttRef(HillToAttRefConfig *configData, uint64_t callTime, int64_t moduleID);
    AttRefMsgPayload RelativeToInertialMRP(HillToAttRefConfig *configData, double relativeAtt[3], double sigma_XN[3]);
#ifdef __cplusplus
}
#endif


#endif

