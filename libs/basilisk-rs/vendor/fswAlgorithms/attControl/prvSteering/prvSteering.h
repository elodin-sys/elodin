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

#ifndef _PRV_STEERING_CONTROL_H_
#define _PRV_STEERING_CONTROL_H_

#include "cMsgCInterface/AttGuidMsg_C.h"
#include "cMsgCInterface/RateCmdMsg_C.h"
#include "architecture/utilities/bskLogging.h"
#include <stdint.h>


/*! module configuration message definition */
typedef struct {
    /* declare module private variables */
    double K1;                          /*!< [rad/sec] Proportional gain applied to principal rotation angle error */
    double K3;                          /*!< [rad/sec] Cubic gain applied to principal rotation angle error
                                            in steering saturation function */
    double omega_max;                   /*!< [rad/sec] Maximum rate command of steering control */

    /* declare module IO interfaces */    
    RateCmdMsg_C rateCmdOutMsg;             //!< rate command output message
    AttGuidMsg_C guidInMsg;                 //!< attitude guidance input message

    BSKLogger *bskLogger;                             //!< BSK Logging
}PrvSteeringConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_prvSteering(PrvSteeringConfig *configData, int64_t moduleID);
    void Update_prvSteering(PrvSteeringConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_prvSteering(PrvSteeringConfig *configData, uint64_t callTime, int64_t moduleID);

    void PRVSteeringLaw(PrvSteeringConfig *configData, double sigma_BR[3], double omega_ast[3], double omega_ast_p[3]);

    
#ifdef __cplusplus
}
#endif


#endif
