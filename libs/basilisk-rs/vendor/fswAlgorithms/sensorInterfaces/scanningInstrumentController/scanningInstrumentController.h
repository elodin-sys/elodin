/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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

#ifndef SCANNINGINSTRUMENTCONTROLLER_H
#define SCANNINGINSTRUMENTCONTROLLER_H

#include <stdint.h>
#include "cMsgCInterface/AccessMsg_C.h"
#include "cMsgCInterface/AttGuidMsg_C.h"
#include "cMsgCInterface/DeviceStatusMsg_C.h"
#include "cMsgCInterface/DeviceCmdMsg_C.h"
#include "architecture/utilities/bskLogging.h"

/*! @brief Module to perform continuous instrument control
 */
typedef struct {
    double attErrTolerance; //!< Normalized MRP attitude error tolerance
    unsigned int useRateTolerance; //!< Flag to enable rate error tolerance
    double rateErrTolerance; //!< Rate error tolerance in rad/s
    unsigned int controllerStatus;  //!< dictates whether or not the controller should be running

    /* declare module IO interfaces */
    AccessMsg_C accessInMsg;  //!< Ground location access
    AttGuidMsg_C attGuidInMsg;  //!< Attitude guidance input message
    DeviceStatusMsg_C deviceStatusInMsg;  //!< Device status input message
    DeviceCmdMsg_C deviceCmdOutMsg;  //!< Device status output message

    BSKLogger *bskLogger;  //!< BSK Logging
}scanningInstrumentControllerConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_scanningInstrumentController(scanningInstrumentControllerConfig *configData, int64_t moduleID);
    void Update_scanningInstrumentController(scanningInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_scanningInstrumentController(scanningInstrumentControllerConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
