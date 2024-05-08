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

#ifndef _RW_MOTOR_VOLTAGE_H_
#define _RW_MOTOR_VOLTAGE_H_

#include <stdint.h>

#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
#include "cMsgCInterface/ArrayMotorTorqueMsg_C.h"
#include "cMsgCInterface/RWAvailabilityMsg_C.h"
#include "cMsgCInterface/RWArrayConfigMsg_C.h"
#include "cMsgCInterface/RWSpeedMsg_C.h"
#include "cMsgCInterface/ArrayMotorVoltageMsg_C.h"

#include "architecture/utilities/bskLogging.h"


/*!@brief module configuration message
 */

typedef struct {
    /* declare module private variables */
    double VMin;                                    /*!< [V]    minimum voltage below which the torque is zero */
    double VMax;                                    /*!< [V]    maximum output voltage */
    double K;                                       /*!< [V/Nm] torque tracking gain for closed loop control.*/
    double rwSpeedOld[MAX_EFF_CNT];                 /*!< [r/s]  the RW spin rates from the prior control step */
    uint64_t priorTime;                             /*!< [ns]   Last time the module control was called */
    int    resetFlag;                               /*!< []     Flag indicating that a module reset occured */

    /* declare module IO interfaces */
    ArrayMotorVoltageMsg_C voltageOutMsg;      /*!< voltage output message*/
    ArrayMotorTorqueMsg_C torqueInMsg;      /*!< Input torque message*/
    RWArrayConfigMsg_C rwParamsInMsg;       /*!< RW array input message*/
    RWSpeedMsg_C rwSpeedInMsg;              /*!< [] The name for the reaction wheel speeds message. Must be provided to enable speed tracking loop */
    RWAvailabilityMsg_C rwAvailInMsg;       /*!< [-] The name of the RWs availability message*/

    RWArrayConfigMsgPayload rwConfigParams;         /*!< [-] struct to store message containing RW config parameters in body B frame */

    BSKLogger *bskLogger;                             //!< BSK Logging

}rwMotorVoltageConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_rwMotorVoltage(rwMotorVoltageConfig *configData, int64_t moduleID);
    void Update_rwMotorVoltage(rwMotorVoltageConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_rwMotorVoltage(rwMotorVoltageConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
