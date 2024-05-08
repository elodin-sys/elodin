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


#ifndef MTBMOMENTUMMANAGEMENTSIMPLE_H
#define MTBMOMENTUMMANAGEMENTSIMPLE_H

#include "cMsgCInterface/RWSpeedMsg_C.h"
#include "cMsgCInterface/RWArrayConfigMsg_C.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
#include "architecture/utilities/bskLogging.h"
#include <stdio.h>
#include "architecture/utilities/macroDefinitions.h"
#include <stdint.h>

/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    /* Configs.*/
    double Kp;                                  //!<[1/s]  momentum feedback gain
    
    /* Inputs.*/
    RWArrayConfigMsg_C rwParamsInMsg;           //!< input message containing RW parameters
    RWSpeedMsg_C rwSpeedsInMsg;                 //!< input message containingRW speeds
    
    /* Outputs.*/
    CmdTorqueBodyMsg_C tauMtbRequestOutMsg;     //!< output message containing control torque in the Body frame
    
    /* Other. */
    RWArrayConfigMsgPayload rwConfigParams;     //!< configuration for RW's
    double Gs[3 * MAX_EFF_CNT];                 //!< transformation from the wheelspace to the Body frame
    BSKLogger *bskLogger;                       //!< BSK Logging
}mtbMomentumManagementSimpleConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_mtbMomentumManagementSimple(mtbMomentumManagementSimpleConfig *configData, int64_t moduleID);
    void Update_mtbMomentumManagementSimple(mtbMomentumManagementSimpleConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_mtbMomentumManagementSimple(mtbMomentumManagementSimpleConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
