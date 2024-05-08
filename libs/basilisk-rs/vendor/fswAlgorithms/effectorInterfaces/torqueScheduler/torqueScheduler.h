/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef _TORQUE_SCHEDULER_
#define _TORQUE_SCHEDULER_

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/ArrayMotorTorqueMsg_C.h"
#include "cMsgCInterface/ArrayEffectorLockMsg_C.h"


/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    /* declare these user-defined inputs */
    int    lockFlag;                               //!< flag to control the scheduler logic
    double tSwitch;                                //!< [s] time span after t0 at which controller switches to second angle

    /* declare this quantity that is a module internal variable */
    uint64_t t0;                                   //!< [ns] epoch time where module is reset

    /* declare module IO interfaces */
    ArrayMotorTorqueMsg_C  motorTorque1InMsg;      //!< input motor torque message #1
    ArrayMotorTorqueMsg_C  motorTorque2InMsg;      //!< input motor torque message #1
    ArrayMotorTorqueMsg_C  motorTorqueOutMsg;      //!< output msg containing the motor torque to the array drive
    ArrayEffectorLockMsg_C effectorLockOutMsg;     //!< output msg containing the flag to actuate or lock the motor

    BSKLogger *bskLogger;                          //!< BSK Logging

}torqueSchedulerConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_torqueScheduler(torqueSchedulerConfig *configData, int64_t moduleID);
    void Reset_torqueScheduler(torqueSchedulerConfig *configData, uint64_t callTime, int64_t moduleID);
    void Update_torqueScheduler(torqueSchedulerConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif


#endif
