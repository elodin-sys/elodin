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

#ifndef _SIMPLE_INSTRUMENT_CONTROLLER_H_
#define _SIMPLE_INSTRUMENT_CONTROLLER_H_

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>

#include "cMsgCInterface/AccessMsg_C.h"
#include "cMsgCInterface/AttGuidMsg_C.h"
#include "cMsgCInterface/DeviceCmdMsg_C.h"
#include "cMsgCInterface/DeviceStatusMsg_C.h"

/*! @brief Data configuration structure for the MRP feedback attitude control routine. */
typedef struct {
    /* User configurable variables */
    double attErrTolerance; //!< Normalized MRP attitude error tolerance
    unsigned int useRateTolerance; //!< Flag to enable rate error tolerance
    double rateErrTolerance; //!< Rate error tolerance in rad/s
    unsigned int imaged;    //!< Indicator for whether or not the image has already been captured
    unsigned int controllerStatus;  //!< dictates whether or not the controller should be running

    /* declare module IO interfaces */
    AccessMsg_C locationAccessInMsg;                   //!< Ground location access input message
    AttGuidMsg_C attGuidInMsg;                         //!< attitude guidance input message
    DeviceStatusMsg_C deviceStatusInMsg;                     //!< (optional) device status input message
    DeviceCmdMsg_C deviceCmdOutMsg;              //!< device status output message

    BSKLogger *bskLogger;                              //!< BSK Logging
}simpleInstrumentControllerConfig;

#ifdef __cplusplus
extern "C" {
#endif

void SelfInit_simpleInstrumentController(simpleInstrumentControllerConfig *configData, int64_t moduleID);
void Update_simpleInstrumentController(simpleInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID);
void Reset_simpleInstrumentController(simpleInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif //_SIMPLE_INSTRUMENT_CONTROLLER_H_
