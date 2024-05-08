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

#ifndef _MEAN_OE_FEEDBACK_H_
#define _MEAN_OE_FEEDBACK_H_

#include <stdint.h>

#include "cMsgCInterface/CmdForceInertialMsg_C.h"
#include "cMsgCInterface/NavTransMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include "architecture/utilities/orbitalMotion.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    NavTransMsg_C chiefTransInMsg;      //!< chief orbit input message
    NavTransMsg_C deputyTransInMsg;     //!< deputy orbit input message
    CmdForceInertialMsg_C forceOutMsg;  //!< deputy control force output message

    double K[36];               //!< Lyapunov Gain (6*6)
    double targetDiffOeMean[6];   //!< target mean orbital element difference
    uint8_t oeType;            //!< 0: classic (default), 1: equinoctial
    double mu;                  //!< [m^3/s^2] gravitational constant
    double req;                 //!< [m] equatorial planet radius
    double J2;                  //!< [] J2 planet oblateness parameter
    BSKLogger *bskLogger;       //!< BSK Logging
} meanOEFeedbackConfig;

#ifdef __cplusplus
extern "C" {
#endif
void SelfInit_meanOEFeedback(meanOEFeedbackConfig *configData, int64_t moduleID);
void Update_meanOEFeedback(meanOEFeedbackConfig *configData, uint64_t callTime, int64_t moduleID);
void Reset_meanOEFeedback(meanOEFeedbackConfig *configData, uint64_t callTime, int64_t moduleID);
#ifdef __cplusplus
}
#endif

#endif
