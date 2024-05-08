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

#ifndef MTB_MOMENTUM_MANAGEMENT_H
#define MTB_MOMENTUM_MANAGEMENT_H

#include "cMsgCInterface/MTBCmdMsg_C.h"
#include "cMsgCInterface/TAMSensorBodyMsg_C.h"
#include "cMsgCInterface/RWSpeedMsg_C.h"
#include "cMsgCInterface/RWArrayConfigMsg_C.h"
#include "cMsgCInterface/MTBArrayConfigMsg_C.h"
#include "cMsgCInterface/ArrayMotorTorqueMsg_C.h"
#include "architecture/utilities/bskLogging.h"
#include <stdio.h>
#include "architecture/utilities/macroDefinitions.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    /*
     * Configs.
     */
    double wheelSpeedBiases[MAX_EFF_CNT];           //!< [rad/s] reaction wheel speed biases
    double cGain;                                   //!<[1/s]  reaction wheel momentum feedback gain
    
    /*
     * Inputs.
     */
    RWArrayConfigMsg_C rwParamsInMsg;               //!< input message for RW parameters
    MTBArrayConfigMsg_C mtbParamsInMsg;             //!< input message for MTB layout
    TAMSensorBodyMsg_C tamSensorBodyInMsg;          //!< input message for magnetic field sensor data in the Body frame
    RWSpeedMsg_C rwSpeedsInMsg;                     //!< input message for RW speeds
    ArrayMotorTorqueMsg_C rwMotorTorqueInMsg;       //!< input message for RW motor torques
    
    /*
     * Outputs.
     */
    MTBCmdMsg_C mtbCmdOutMsg;                       //!< output message for MTB dipole commands
    ArrayMotorTorqueMsg_C rwMotorTorqueOutMsg;      //!< output message for RW motor torques
    
    /*
     * Other.
     */
    BSKLogger *bskLogger;                           //!< BSK Logging
    double tauDesiredMTB_B[3];                      //!< [N-m] desired torque produced by the magnetic torque bars in the Body frame
    double tauDesiredRW_B[3];                       //!< [N-m]  desired torque produced by the reaction wheels in the Body frame
    double hDeltaWheels_W[MAX_EFF_CNT];             //!<  [N-m-s] momentum of each wheel
    double hDeltaWheels_B[3];                       //!<  [N-m-s] momentum of reaction wheels in the Body frame
    double tauDesiredRW_W[MAX_EFF_CNT];             //!<  [N-m] Desired individual wheel torques
    double tauIdealRW_W[MAX_EFF_CNT];               //!<  [N-m-s] Ideal individual wheel torques
    double tauIdealRW_B[MAX_EFF_CNT];               //!<  [N-m-s] Ideal wheel torque in the body frame
    double wheelSpeedError_W[MAX_EFF_CNT];          //!<  [N-m-s] difference between current wheel speeds and desired wheel speeds
    RWArrayConfigMsgPayload rwConfigParams;         //!< configuration for RW's
    MTBArrayConfigMsgPayload mtbConfigParams;       //!< configuration for MTB layout
    
}mtbMomentumManagementConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_mtbMomentumManagement(mtbMomentumManagementConfig *configData, int64_t moduleID);
    void Update_mtbMomentumManagement(mtbMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_mtbMomentumManagement(mtbMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID);
    void v3TildeM(double v[3], void *result);

#ifdef __cplusplus
}
#endif


#endif
