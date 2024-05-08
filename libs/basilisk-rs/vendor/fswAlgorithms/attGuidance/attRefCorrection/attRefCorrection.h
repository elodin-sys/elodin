/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#ifndef ATTREFCORRECTION_H
#define ATTREFCORRECTION_H

#include <stdint.h>
#include "cMsgCInterface/AttRefMsg_C.h"
#include "architecture/utilities/bskLogging.h"

/*! @brief This module reads in the attitude reference message and adjusts it by a fixed rotation.  This allows a general body-fixed frame B to align with this corrected reference frame Rc.
 */
typedef struct {

    /* declare module IO interfaces */
    AttRefMsg_C attRefInMsg;    //!< attitude reference input message
    AttRefMsg_C attRefOutMsg;   //!< corrected attitude reference input message

    double sigma_BcB[3];        //!< MRP from from body frame B to the corrected body frame Bc

    BSKLogger *bskLogger;       //!< BSK Logging
}attRefCorrectionConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_attRefCorrection(attRefCorrectionConfig *configData, int64_t moduleID);
    void Update_attRefCorrection(attRefCorrectionConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_attRefCorrection(attRefCorrectionConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
