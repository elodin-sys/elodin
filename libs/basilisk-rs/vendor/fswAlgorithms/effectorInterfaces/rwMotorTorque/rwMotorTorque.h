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

#ifndef _RW_MOTOR_TORQUE_H_
#define _RW_MOTOR_TORQUE_H_

#include <stdint.h>

#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
#include "cMsgCInterface/ArrayMotorTorqueMsg_C.h"
#include "cMsgCInterface/RWAvailabilityMsg_C.h"
#include "cMsgCInterface/RWArrayConfigMsg_C.h"

#include "architecture/utilities/bskLogging.h"


/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    /* declare module private variables */
    double   controlAxes_B[3*3];        //!< [-] array of the control unit axes
    uint32_t numControlAxes;            //!< [-] counter indicating how many orthogonal axes are controlled
    int      numAvailRW;                //!< [-] number of reaction wheels available
    RWArrayConfigMsgPayload rwConfigParams; //!< [-] struct to store message containing RW config parameters in body B frame
    double GsMatrix_B[3*MAX_EFF_CNT];   //!< [-] The RW spin axis matrix in body frame components
    double CGs[3][MAX_EFF_CNT];         //!< [-] Projection matrix that defines the controlled body axes

    /* declare module IO interfaces */
    ArrayMotorTorqueMsg_C rwMotorTorqueOutMsg;   //!< RW motor torque output message
    CmdTorqueBodyMsg_C vehControlInMsg;  //!<  vehicle control (Lr) Input message

    RWArrayConfigMsg_C rwParamsInMsg;    //!<  RW Array input message
    RWAvailabilityMsg_C rwAvailInMsg;     //!< optional RWs availability input message

    BSKLogger *bskLogger;                             //!< BSK Logging

}rwMotorTorqueConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_rwMotorTorque(rwMotorTorqueConfig *configData, int64_t moduleID);
    void Update_rwMotorTorque(rwMotorTorqueConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_rwMotorTorque(rwMotorTorqueConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
