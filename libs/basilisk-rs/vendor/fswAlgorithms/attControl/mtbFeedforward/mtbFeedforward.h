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


#ifndef MTBFEEDFORWARD_H
#define MTBFEEDFORWARD_H

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
#include "cMsgCInterface/MTBCmdMsg_C.h"
#include "cMsgCInterface/TAMSensorBodyMsg_C.h"
#include "cMsgCInterface/MTBArrayConfigMsg_C.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    
    /* Inputs. */
    CmdTorqueBodyMsg_C vehControlInMsg;                 //!< input message containing the current control torque in the Body frame
    MTBCmdMsg_C dipoleRequestMtbInMsg;                  //!< input message containing the individual dipole requests for each torque bar on the vehicle
    TAMSensorBodyMsg_C tamSensorBodyInMsg;              //!< [Tesla] input message for magnetic field sensor data in the Body frame
    MTBArrayConfigMsg_C mtbArrayConfigParamsInMsg;      //!< input message containing configuration parameters for all the torque bars on the vehicle
    
    /* Outputs. */
    CmdTorqueBodyMsg_C vehControlOutMsg;                //!< output message containing the current control torque in the Body frame
    
    /* Other. */
    MTBArrayConfigMsgPayload mtbArrayConfigParams;      //!< configuration for MTB layout
    BSKLogger *bskLogger;                               //!< BSK Logging
}mtbFeedforwardConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_mtbFeedforward(mtbFeedforwardConfig *configData, int64_t moduleID);
    void Update_mtbFeedforward(mtbFeedforwardConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_mtbFeedforward(mtbFeedforwardConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
