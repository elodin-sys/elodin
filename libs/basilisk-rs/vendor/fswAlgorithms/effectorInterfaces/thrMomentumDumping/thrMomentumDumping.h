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

#ifndef _THR_MOMENTUM_DUMPING_H_
#define _THR_MOMENTUM_DUMPING_H_

#include <stdint.h>

#include "cMsgCInterface/THRArrayConfigMsg_C.h"
#include "cMsgCInterface/THRArrayCmdForceMsg_C.h"
#include "cMsgCInterface/THRArrayOnTimeCmdMsg_C.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"

#include "architecture/utilities/bskLogging.h"



/*! @brief thruster force momentum dumping module configuration message
 */
typedef struct {
    /* declare module private variables */
    int32_t     thrDumpingCounter;                      //!<        counter to specify after how many contro period a thruster firing should occur.
    double      Delta_p[MAX_EFF_CNT];                   //!<        vector of desired total thruster impulses
    uint64_t    lastDeltaHInMsgTime;                    //!<        time tag of the last momentum change input message 
    double      thrOnTimeRemaining[MAX_EFF_CNT];        //!<        vector of remaining thruster on times
    uint64_t    priorTime;                              //!< [ns]   Last time the attitude control is called
    int         numThrusters;                           //!<        number of thrusters installed
    double      thrMaxForce[MAX_EFF_CNT];               //!< [N]    vector of maximum thruster forces

    /* declare module public variables */
    int         maxCounterValue;                        //!<        this variable must be set to a non-zero value, indicating how many control periods to wait until the thrusters fire again to dump RW momentum
    double      thrMinFireTime;                         //!< [s]    smallest thruster firing time

    /* declare module IO interfaces */
    THRArrayOnTimeCmdMsg_C thrusterOnTimeOutMsg;        //!< thruster on time output message name
    THRArrayCmdForceMsg_C thrusterImpulseInMsg;         //!< desired thruster impulse input message name
    THRArrayConfigMsg_C thrusterConfInMsg;              //!< The name of the thruster configuration Input message
    CmdTorqueBodyMsg_C deltaHInMsg;                     //!< The name of the requested momentum change input message

    BSKLogger *bskLogger;                             //!< BSK Logging
}thrMomentumDumpingConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_thrMomentumDumping(thrMomentumDumpingConfig *configData, int64_t moduleID);
    void Update_thrMomentumDumping(thrMomentumDumpingConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_thrMomentumDumping(thrMomentumDumpingConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
