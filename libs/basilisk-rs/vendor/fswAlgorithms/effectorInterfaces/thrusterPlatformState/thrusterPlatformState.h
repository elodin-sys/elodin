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

#ifndef _THRUSTER_PLATFORM_STATE_
#define _THRUSTER_PLATFORM_STATE_

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
#include "cMsgCInterface/HingedRigidBodyMsg_C.h"
#include "cMsgCInterface/THRConfigMsg_C.h"


/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    /* declare these user-defined quantities */
    double sigma_MB[3];                                   //!< orientation of the M frame w.r.t. the B frame
    double r_BM_M[3];                                     //!< position of B frame origin w.r.t. M frame origin, in M frame coordinates
    double r_FM_F[3];                                     //!< position of F frame origin w.r.t. M frame origin, in F frame coordinates

    double K;                                             //!< momentum dumping time constant [1/s]

    /* declare module IO interfaces */
    THRConfigMsg_C            thrusterConfigFInMsg;       //!< input thruster configuration msg
    HingedRigidBodyMsg_C      hingedRigidBody1InMsg;      //!< output msg containing theta1 reference and thetaDot1 reference
    HingedRigidBodyMsg_C      hingedRigidBody2InMsg;      //!< output msg containing theta2 reference and thetaDot2 reference
    THRConfigMsg_C            thrusterConfigBOutMsg;      //!< output msg containing the thruster configuration infor in B-frame

    BSKLogger *bskLogger;                                 //!< BSK Logging

}thrusterPlatformStateConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_thrusterPlatformState(thrusterPlatformStateConfig *configData, int64_t moduleID);
    void Reset_thrusterPlatformState(thrusterPlatformStateConfig *configData, uint64_t callTime, int64_t moduleID);
    void Update_thrusterPlatformState(thrusterPlatformStateConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif


#endif
